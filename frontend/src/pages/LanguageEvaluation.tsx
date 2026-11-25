// Evaluation results of a language pack (ISL results.json, docs/ISL_DESIGN.md 6.5 and 9.2). Every number is shown with
// its protocol label, n and 95% interval; accuracy always sits next to coverage and the silent error rate; a smoke run
// is labelled as such. No target numbers are shown, only measured ones.
import { useApi } from "../lib/hooks";
import { pct } from "../lib/format";
import { useSignLanguage } from "../state/language";
import { Badge, Card, Empty, Kv, Notice, PageHeader, ScrollRegion } from "../components/ui";

type Interval = [number, number];
type Protocol = {
  label?: string; n?: number; n_classes?: number; n_groups?: number; gallery_concepts?: number;
  metrics?: Record<string, number>; intervals?: Record<string, Interval>;
  policies?: Record<string, Record<string, number | string>>; aurc?: number; stages_p50_ms?: Record<string, number>;
};
type Results = {
  language: string; profile: string; created?: string; git_commit?: string; smoke?: boolean; label?: string;
  data?: Record<string, unknown> & { split?: string; split_label?: string };
  protocols?: Record<string, Protocol>;
};

function withCi(p: Protocol, key: string) {
  const v = p.metrics?.[key];
  const ci = p.intervals?.[key];
  return v === undefined ? "—" : ci ? `${pct(v)} [${pct(ci[0], 0)}–${pct(ci[1], 0)}]` : pct(v);
}

function ProtocolCard({ id, p, title }: { id: string; p: Protocol; title: string }) {
  return (
    <Card title={<span>{title} <Badge className="ml-1">{id}</Badge></span>} subtitle={p.label}>
      <Kv k="Top-1 (95% interval)" v={withCi(p, "top1")} />
      <Kv k="Top-5 (95% interval)" v={withCi(p, "top5")} />
      {p.metrics?.macro_f1 !== undefined && <Kv k="Macro F1" v={pct(p.metrics.macro_f1)} />}
      <Kv k="Clips" v={p.n ?? "—"} />
      {p.n_classes !== undefined && <Kv k="Classes" v={p.n_classes} />}
      {p.gallery_concepts !== undefined && <Kv k="Dictionary entries searched" v={p.gallery_concepts.toLocaleString()} />}
      {p.n_groups !== undefined && <Kv k="Signer / session groups" v={p.n_groups} />}
    </Card>
  );
}

export function LanguageEvaluation() {
  const { id, info } = useSignLanguage();
  const { data: r, error } = useApi<Results>(`/api/languages/${encodeURIComponent(id)}/results`);
  const P = r?.protocols ?? {};
  const e3 = P.E3;
  return (
    <div className="fade-in">
      <PageHeader eyebrow={`${info?.sign_language ?? id} · measured results`} title="Model Evaluation"
        description="Results of the language pack's models on held-out clips. Live webcam accuracy is expected to be lower (different camera, lighting and signers)." />
      {error ? (
        <Card><Empty title="No evaluation results for this language yet">{error.includes("404") || error.includes("no evaluation") ? "Build and evaluate the models (python -m app.isl.cli evaluate) to see results here." : error}</Empty></Card>
      ) : !r ? (
        <p className="text-sm text-fg-3">Loading…</p>
      ) : (
        <div className="flex flex-col gap-5">
          {r.smoke && <Notice tone="warn" title="Smoke run, not the app's accuracy">{r.label}</Notice>}
          <div className="flex flex-wrap gap-2 text-xs text-fg-3">
            <Badge tone={r.profile === "commercial-safe" ? "teal" : "warn"}>{r.profile}</Badge>
            {r.data?.split && <Badge>split {r.data.split}</Badge>}
            {r.data?.split_label && <Badge>{String(r.data.split_label)}</Badge>}
            {r.created && <span>{r.created}</span>}
            {r.git_commit && <span className="font-mono">commit {r.git_commit}</span>}
          </div>
          <div className="grid gap-5 lg:grid-cols-2">
            {P.E2 && <ProtocolCard id="E2" p={P.E2} title="Any dictionary word (open vocabulary)" />}
            {P.E1 && <ProtocolCard id="E1" p={P.E1} title="Core words (closed set)" />}
          </div>
          {e3?.policies && (
            <Card title={<span>Whole system: show / ask / not sure <Badge className="ml-1">E3</Badge></span>} subtitle={e3.label}>
              <ScrollRegion label="Show / ask / not sure table" className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-sm">
                  <caption className="sr-only">How often the whole system shows a word, asks "did you mean", or says "not sure", per policy</caption>
                  <thead className="text-xs text-fg-3">
                    <tr>
                      <th scope="col" className="py-2 pr-3 font-medium">Policy</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Shown without asking</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Correct when shown</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Wrong word shown silently</th>
                      <th scope="col" className="py-2 pr-3 font-medium">"Did you mean"</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Right word in the 5 choices</th>
                      <th scope="col" className="py-2 pr-3 font-medium">"Not sure"</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-[13px]">
                    {Object.entries(e3.policies).map(([k, v]) => (
                      <tr key={k} className="border-t border-ink-700/60">
                        <th scope="row" className="py-2 pr-3 font-sans font-normal">{k}</th>
                        <td className="py-2 pr-3">{pct(v.coverage as number)}{v.n_accepted !== undefined ? ` (${v.n_accepted})` : ""}</td>
                        <td className="py-2 pr-3">{pct(v.accepted_precision as number)}</td>
                        <td className="py-2 pr-3">{pct(v.silent_error_rate as number)}</td>
                        <td className="py-2 pr-3">{pct(v.suggest_rate as number)}</td>
                        <td className="py-2 pr-3">{pct(v.suggest_top5_hit as number)}</td>
                        <td className="py-2 pr-3">{pct(v.abstain_rate as number)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </ScrollRegion>
              <p className="mt-2 text-[11px] text-fg-3">n = {e3.n} test clips. Accuracy is only comparable together with how often a word is shown (coverage).</p>
            </Card>
          )}
          {P.E6 && (
            <Card title={<span>Speed <Badge className="ml-1">E6</Badge></span>} subtitle={P.E6.label}>
              {Object.entries(P.E6.metrics ?? {}).map(([k, v]) => <Kv key={k} k={k} v={typeof v === "number" ? v.toFixed(1) : String(v)} />)}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
