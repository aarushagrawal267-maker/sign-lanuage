"""ISL recogniser: classifier + dictionary retrieval + My signs + fusion + abstain (docs/ISL_DESIGN.md 6.3, 7.3).

Every component is optional so a pack degrades gracefully: no classifier -> retrieval only; no index ->
classifier only; neither -> only the user's own signs ("My signs") can be recognised.
"""

from __future__ import annotations

import time

import numpy as np

from app.features.build import FeatureError, build_features
from app.features.spec import FeatureConfig
from app.inference.recognizer import Candidate, Recognition
from app.isl.models.encoder import EncoderPredictor, FeatureStatsEmbedder, RawEmbedder
from app.isl.models.fusion import USER_PREFIX, FusionInputs, FusionParams, decide, fuse
from app.isl.models.index import EmbeddingIndex
from app.landmarks.raw import RawSequence
from app.models.checkpoint import Predictor
from app.packs.vocabulary import ORIGIN, Vocabulary


DEFAULT_CLASSIFIER_THRESHOLD = 0.6  # Predictor's default when a checkpoint has no recommended_threshold


def classifier_calibration(predictor) -> tuple[float, float, bool]:
    """``(temperature, accept threshold, fitted)`` of the classifier used ALONE (sliding mode, E1).

    Both come from the classifier checkpoint: ``train_classifier`` fits ``temperature`` and the classic
    ``recommended_threshold`` on the classifier's own validation part, the threshold on temperature-scaled
    probabilities. The pack's fused thresholds (``fusion.json`` ``accept``) are on another scale - the softmax of
    classifier + retrieval + prior - and must never be compared with a classifier probability (docs 15.16)."""
    meta = getattr(predictor, "meta", None) or {}
    t = meta.get("temperature")
    t = float(t) if t is not None and float(t) > 0 else 1.0
    thr = meta.get("recommended_threshold")
    return t, float(thr) if thr is not None else DEFAULT_CLASSIFIER_THRESHOLD, thr is not None


def calibrated_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax(log p / T) along the last axis: temperature-scaled class probabilities from T = 1 ones."""
    if temperature == 1.0:
        return probs
    z = np.log(np.clip(probs, 1e-12, 1.0)) / max(float(temperature), 1e-3)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class IslRecognizer:
    def __init__(self, language: str, feature_config: FeatureConfig, vocab: Vocabulary, *, lexicon=None,
                 classifier: Predictor | None = None, encoder: EncoderPredictor | None = None,
                 index: EmbeddingIndex | None = None, speller=None, fusion: FusionParams | None = None,
                 my_signs=None, profile: str = "commercial-safe"):
        self.language = language
        self.feature_config = feature_config
        self.vocab = vocab
        self.lexicon = lexicon
        self.classifier = classifier
        self.encoder = encoder
        self.index = index if encoder is not None else None
        self.speller = speller
        self.fusion = fusion or FusionParams()
        self.my_signs = my_signs
        self.profile = profile
        for name, comp in (("classifier", classifier), ("encoder", encoder)):
            if comp is not None and comp.feature_config != feature_config:
                raise ValueError(f"the {name} was trained with another feature configuration "
                                 f"({comp.feature_config.to_dict()}) than the pack ({feature_config.to_dict()})")
        self.user_embedder = RawEmbedder(encoder if encoder is not None else FeatureStatsEmbedder(feature_config),
                                         feature_config)
        self.recommended_threshold = float(self.fusion.accept.get("balanced", 0.7))
        caps = {"words", "my_signs"}
        if self.index is not None:
            caps.add("retrieval")
        if speller is not None:
            caps.add("spelling")
        self.capabilities = frozenset(caps)
        self._prior = {}

    # ------------------------------------------------------------------ helpers
    @property
    def has_global_model(self) -> bool:
        return self.classifier is not None or self.index is not None

    def _hi(self, concept_id: str) -> tuple[str | None, str | None]:
        if self.lexicon is None:
            return None, None
        return self.lexicon.text(concept_id)

    def global_search(self, z: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        return list(self.index.concept_scores(z, k)[0].items())

    def _features(self, crops: list[RawSequence]) -> np.ndarray:
        return np.stack([build_features(c, self.feature_config) for c in crops])

    @property
    def classifier_temperature(self) -> float:
        return classifier_calibration(self.classifier)[0] if self.classifier is not None else 1.0

    @property
    def classifier_threshold(self) -> float | None:
        """Accept threshold of the classifier alone (sliding mode), on :meth:`class_probs`' scale."""
        return classifier_calibration(self.classifier)[1] if self.classifier is not None else None

    def class_probs(self, crops: list[RawSequence]):
        """Closed-set probabilities of the classifier alone (sliding mode), temperature-scaled with the
        classifier's own validation temperature so they match :attr:`classifier_threshold`."""
        if self.classifier is None:
            return None
        t0 = time.perf_counter()
        try:
            feats = self._features(crops)
        except FeatureError:
            return None
        fm = (time.perf_counter() - t0) * 1000
        probs, ms = self.classifier.predict(feats)
        return calibrated_probs(probs, self.classifier_temperature), list(self.classifier.class_names), fm, ms

    def evidence(self, feats: np.ndarray, use_my_signs: bool = True) -> tuple[FusionInputs, dict]:
        """Run every component on (crops, T, D) features. Returns the fusion inputs and per-stage timings."""
        timing: dict[str, float] = {}
        inp = FusionInputs()
        if self.classifier is not None:
            probs, timing["classifier_ms"] = self.classifier.predict(feats)
            inp.cls_logp = np.log(np.clip(probs, 1e-12, 1.0))
            inp.cls_ids = list(self.classifier.class_names)
        z = None
        if self.encoder is not None:
            z, timing["encoder_ms"] = self.encoder.embed(feats)
            if self.index is not None and len(self.index):
                t0 = time.perf_counter()
                inp.ret = self.index.concept_matrix(z)
                inp.ret_ids = self.index.concepts
                timing["search_ms"] = (time.perf_counter() - t0) * 1000
        if use_my_signs and self.my_signs is not None:
            t0 = time.perf_counter()
            uidx = self.my_signs.user_index(self.user_embedder)
            if uidx is not None and len(uidx):
                zu = z if (z is not None and self.encoder is not None) else self.user_embedder.embedder.embed(feats)[0]
                inp.user = uidx.concept_matrix(zu)
                inp.user_ids = uidx.concepts
            timing["my_signs_ms"] = (time.perf_counter() - t0) * 1000
        keys = set(inp.cls_ids)
        if inp.ret is not None:
            keys |= {inp.ret_ids[i] for i in np.argsort(-inp.ret.mean(0))[: self.fusion.top_n]}
        inp.prior = {k: self._prior.setdefault(k, self.vocab.prior(k)) for k in keys}
        return inp, {"z": z, **timing}

    # ----------------------------------------------------------------- recognise
    def recognize(self, crops: list[RawSequence], top_k: int = 5, policy: str = "balanced",
                  use_my_signs: bool = True) -> Recognition:
        t0 = time.perf_counter()
        try:
            feats = self._features(crops)
        except FeatureError as e:
            return Recognition([], "abstain", "no_features", 0.0, 0, len(crops), detail={"error": str(e)})
        feat_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        inp, info = self.evidence(feats, use_my_signs)
        z = info.pop("z")
        res = fuse(inp, self.fusion)
        p = self.fusion
        decision, reason = decide(res.probs, res.agree, len(crops), p, policy, res.best_cos if res.best_cos > -1 else None)
        keys, probs = list(res.keys), np.array(res.probs)
        # My signs searched first (7.3): a confident user match that beats the dictionary is taken directly
        override = False
        if res.best_user is not None:
            ukey, ucos = res.best_user
            theta = self._user_theta(ukey)
            u_agree = int(sum(inp.user_ids[int(j)] == ukey for j in inp.user.argmax(1)))
            if (ucos >= theta and ucos >= res.best_global_cos + p.user_margin
                    and u_agree >= min(p.min_agree, len(crops))):
                j = keys.index(ukey)
                keys.insert(0, keys.pop(j))
                probs = np.concatenate([[probs[j]], np.delete(probs, j)])
                decision, reason, override = "accept", "my_signs", True
        if not override and keys and keys[0].startswith(USER_PREFIX) and decision == "accept":
            # a user sign is never accepted below its own calibrated similarity (enrolment self-test)
            decision, reason = "suggest", "low_confidence"
        if not override and not self.has_global_model:
            # only user signs: the softmax over a handful of signs says nothing about "none of them"
            ucos = res.best_user[1] if res.best_user else -1.0
            near = res.best_user is not None and ucos >= self._user_theta(res.best_user[0]) - 0.1
            decision, reason = ("suggest", "low_confidence") if near else ("abstain", "out_of_vocabulary")
        cands = self._candidates(keys, probs, res.components, z, top_k)
        inf_ms = (time.perf_counter() - t1) * 1000
        conf = cands[0].prob if cands else 0.0
        return Recognition(cands, decision, reason, float(conf), res.agree, len(crops), feat_ms, inf_ms,
                           detail={**{k: round(v, 3) for k, v in info.items()}, "best_cos": round(res.best_cos, 4),
                                   "policy": policy, "fusion_fitted": p.fitted})

    def _user_theta(self, key: str) -> float:
        p = self.fusion
        sign = self.my_signs.get(key.removeprefix(USER_PREFIX)) if self.my_signs else None
        return max(sign.theta if sign and sign.theta is not None else 1.0, p.user_theta_floor, p.s_min)

    def _candidate(self, key: str, prob: float, comps: dict, row_scores) -> Candidate:
        if key.startswith(USER_PREFIX):
            sign = self.my_signs.get(key.removeprefix(USER_PREFIX)) if self.my_signs else None
            if sign is None:
                return Candidate(key, -1, key, prob, {"en": key, "hi": None}, None, "my_signs", key, None, "My signs")
            origin = "My signs" if sign.origin == "recorded" else f"My signs: {sign.origin.removeprefix('imported:')}"
            if sign.concept_id and self.vocab.get(sign.concept_id):
                hi, rev = self._hi(sign.concept_id)
                if sign.text_hi and not hi:
                    hi, rev = sign.text_hi, "user"
                return Candidate(sign.concept_id, self.vocab.index_of(sign.concept_id), self.vocab.english(sign.concept_id),
                                 prob, {"en": self.vocab.english(sign.concept_id), "hi": hi}, rev, "my_signs",
                                 key, sign.region, origin)
            return Candidate(key, -1, sign.text_en, prob, {"en": sign.text_en, "hi": sign.text_hi},
                             "user" if sign.text_hi else None, "my_signs", key, sign.region, origin)
        in_c, in_r = key in comps.get("classifier", ()), key in comps.get("retrieval", ())
        source = "classifier+retrieval" if (in_c and in_r) else ("classifier" if in_c and not in_r else "retrieval")
        if self.index is None:
            source = "classifier"
        elif self.classifier is None:
            source = "retrieval"
        variant_id, region, origin = None, None, None
        if self.index is not None and row_scores is not None:
            row = self.index.best_row(None, key, row_scores)
            if row is not None:
                variant_id = row.get("variant_id")
                v = self.vocab.variant(key, variant_id)
                region = (v or {}).get("region")
                origin = ORIGIN.get(row.get("source", ""), row.get("source"))
        if origin is None:
            r = self.vocab.get(key)
            srcs = {v.get("source") for v in (r or {}).get("variants", [])}
            origin = "INCLUDE" if "include" in srcs and source == "classifier" else ("ISLRTC dictionary" if "islrtc" in srcs else None)
        hi, rev = self._hi(key)
        en = self.vocab.english(key)
        return Candidate(key, self.vocab.index_of(key), en, prob, {"en": en, "hi": hi}, rev, source, variant_id,
                         region, origin)

    def _candidates(self, keys: list[str], probs: np.ndarray, comps: dict, z, top_k: int) -> list[Candidate]:
        row_scores = self.index.scores(z).mean(0) if (self.index is not None and z is not None) else None
        out: list[Candidate] = []
        seen: dict[str, Candidate] = {}
        for k, pr in zip(keys[: max(top_k * 3, 15)], probs):
            c = self._candidate(k, float(pr), comps, row_scores)
            if c.concept_id in seen:  # a user variant of a dictionary concept: one row, summed probability
                prev = seen[c.concept_id]
                prev.prob += c.prob
                if c.source == "my_signs":
                    prev.source, prev.variant_id, prev.region, prev.origin = c.source, c.variant_id, c.region, c.origin
                continue
            seen[c.concept_id] = c
            out.append(c)
        return out[:top_k]

    # ------------------------------------------------------------------ spelling
    def spell(self, raw: RawSequence, k: int = 5):
        if self.speller is None:
            return None
        return self.speller.spell(raw, k=k)

    def spell_recognition(self, raw: RawSequence, top_k: int = 5) -> Recognition:
        t0 = time.perf_counter()
        sr = self.spell(raw, k=top_k)
        ms = (time.perf_counter() - t0) * 1000
        if sr is None or not sr.suggestions:
            return Recognition([], "abstain", "no_letters", 0.0, 0, 1, 0.0, ms,
                               detail={"spell": sr.to_json() if sr else None})
        cands = []
        for s in sr.suggestions:
            cid = s["concept_id"]
            if cid and self.vocab.get(cid):
                hi, rev = self._hi(cid)
                en = self.vocab.english(cid)
                cands.append(Candidate(cid, self.vocab.index_of(cid), en, s["score"], {"en": en, "hi": hi}, rev,
                                       "spelling", None, None, "Fingerspelled"))
            else:
                w = s["word"]
                cands.append(Candidate(f"spelled:{w.upper()}", -1, w, s["score"], {"en": w, "hi": None}, None,
                                       "spelling", None, None,
                                       "Fingerspelled" if s["in_lexicon"] else "Fingerspelled (not in dictionary)"))
        # The letter model is trained on very few signers: never accept silently, always let the user choose.
        return Recognition(cands, "suggest", "spelling", cands[0].prob, 1, 1, 0.0, ms, detail={"spell": sr.to_json()})
