import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import type { VocabEntry } from "../api/client";
import { useApi } from "../lib/hooks";
import { pct } from "../lib/format";
import { Badge, Card, Empty, Kv, Notice, PageHeader, Segmented, SyntheticBadge, cx, inputCls } from "../components/ui";
import { useSignLanguage } from "../state/language";
import { LanguageVocabulary } from "./LanguageVocabulary";

type Sort = "class" | "alpha" | "f1" | "data";

function f1Tone(f1: number | undefined) {
  if (f1 === undefined) return "bg-ink-700";
  if (f1 >= 0.7) return "bg-teal";
  if (f1 >= 0.4) return "bg-accent";
  return "bg-serious";
}

function DetailPanel({ e, onClose, modelName }: { e: VocabEntry; onClose: () => void; modelName?: string }) {
  return (
    <Card title={<span className="font-mono text-lg">{e.gloss.toUpperCase()}</span>} subtitle={`WLASL100 class ${e.class_index} · ASL gloss label (an English word used to name the sign)`}
      actions={<button onClick={onClose} className="rounded-md p-1 text-fg-3 hover:bg-ink-800 hover:text-fg" aria-label="Close details"><X className="size-4" /></button>}>
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-fg-2">Data</div>
          <Kv k="Annotated instances" v={e.annotated} />
          <Kv k="Valid local videos" v={e.valid} />
          <Kv k="With landmarks" v={e.with_landmarks} />
          <Kv k="Official train instances" v={e.train_official} />
          <Kv k="Signers" v={e.signers} />
          <Kv k="Dialect variations" v={e.variations} />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-fg-2">Recognition performance {modelName && <span className="text-fg-3">· {modelName}</span>}</div>
          {e.val ? (
            <>
              <Kv k="Validation precision" v={pct(e.val.precision, 0)} />
              <Kv k="Validation recall" v={pct(e.val.recall, 0)} />
              <Kv k="Validation F1" v={pct(e.val.f1, 0)} />
              <Kv k="Validation support" v={e.val.support} />
              <Kv k="Test F1" v={e.test ? `${pct(e.test.f1, 0)} (n=${e.test.support})` : "not evaluated"} />
            </>
          ) : <p className="text-sm text-fg-3">No active model with per-class metrics.</p>}
        </div>
      </div>
      <div className="mt-5">
        <div className="mb-2 text-xs font-medium text-fg-2">Reference example</div>
        {e.local_example ? (
          <>
            <video controls muted loop playsInline className="w-full max-w-lg rounded-xl border border-ink-700 bg-black" src={`/api/dataset/videos/${e.local_example}/file`} />
            <p className="mt-1.5 text-[11px] text-fg-3">Your local copy of WLASL video {e.local_example} (C-UDA: for research use on this machine; not redistributed).</p>
          </>
        ) : (
          <p className="text-sm text-fg-3">No local video for this gloss. Reference clips are shown only from videos you have acquired under the WLASL licence.</p>
        )}
      </div>
    </Card>
  );
}

function AslVocabulary() {
  const { data, error } = useApi<{ model: { id: number; name: string; is_synthetic: boolean } | null; glosses: VocabEntry[] }>("/api/vocabulary");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<Sort>("class");
  const [sel, setSel] = useState<number | null>(null);

  const items = useMemo(() => {
    const g = (data?.glosses ?? []).filter((x) => x.gloss.toLowerCase().includes(q.trim().toLowerCase()));
    if (sort === "alpha") g.sort((a, b) => a.gloss.localeCompare(b.gloss));
    if (sort === "f1") g.sort((a, b) => (a.val?.f1 ?? -1) - (b.val?.f1 ?? -1));
    if (sort === "data") g.sort((a, b) => b.with_landmarks - a.with_landmarks || b.annotated - a.annotated);
    return g;
  }, [data, q, sort]);
  const selected = data?.glosses.find((g) => g.class_index === sel) ?? null;

  return (
    <div className="fade-in">
      <PageHeader eyebrow="Supported vocabulary" title="Sign Vocabulary Explorer"
        description="The 100 ASL signs the system can recognise (WLASL100, first 100 glosses of the official annotation file). Each label is a gloss — an English word used to name a sign — not a full natural-language translation." />

      <div className="mb-5"><Notice>Anything signed outside this list cannot be recognised. Glosses may have several regional or dialect variants; WLASL includes some of them, not all.</Notice></div>
      {data?.model?.is_synthetic && <div className="mb-5 flex items-center gap-2 text-sm text-fg-2"><SyntheticBadge />Per-class scores below come from a synthetic test model.</div>}

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-3" aria-hidden />
          <input className={cx(inputCls, "pl-9")} placeholder="Search glosses" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search glosses" />
        </div>
        <Segmented ariaLabel="Sort" value={sort} onChange={setSort} options={[
          { value: "class", label: "Class id" }, { value: "alpha", label: "A–Z" }, { value: "f1", label: "Weakest first" }, { value: "data", label: "Most data" }]} />
        <span className="text-xs text-fg-3">{items.length} of {data?.glosses.length ?? 0}</span>
        {data?.model && <span className="ml-auto flex items-center gap-2 text-xs text-fg-3"><span className="size-2 rounded-full bg-teal" />F1 ≥ 70% <span className="size-2 rounded-full bg-accent" />40–70% <span className="size-2 rounded-full bg-serious" />&lt; 40% (validation)</span>}
      </div>

      {error && <Notice tone="danger">{error}</Notice>}
      {data && data.glosses.length === 0 && <Card><Empty title="Vocabulary not loaded">Load the official annotations on the Dataset page.</Empty></Card>}

      {selected && <div className="mb-5"><DetailPanel e={selected} onClose={() => setSel(null)} modelName={data?.model?.name} /></div>}

      <ul className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
        {items.map((g) => (
          <li key={g.class_index}>
            <button onClick={() => setSel(g.class_index === sel ? null : g.class_index)} aria-pressed={sel === g.class_index}
              className={cx("group flex w-full flex-col gap-2 rounded-xl border bg-ink-850 p-3 text-left transition-colors hover:border-ink-600",
                sel === g.class_index ? "border-accent" : "border-ink-700")}>
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] text-fg-3">#{String(g.class_index).padStart(2, "0")}</span>
                {g.val && <span className={cx("size-2 rounded-full", f1Tone(g.val.f1))} title={`validation F1 ${pct(g.val.f1, 0)}`} />}
              </div>
              <span className="truncate font-mono text-[15px] font-medium text-fg">{g.gloss.toUpperCase()}</span>
              <div className="flex items-center justify-between text-[11px] text-fg-3">
                <span>{g.annotated} clips · {g.signers} signers</span>
                {g.val ? <span className="font-mono text-fg-2">{pct(g.val.f1, 0)}</span> : g.with_landmarks ? <Badge>{g.with_landmarks} seq</Badge> : null}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function VocabularyPage() {
  const { isAsl, loading, id } = useSignLanguage();
  if (loading) return <p className="text-sm text-fg-3">Loading…</p>;
  return isAsl ? <AslVocabulary /> : <LanguageVocabulary key={id} />;
}
