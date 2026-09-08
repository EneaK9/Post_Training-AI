"use client";

import { Suspense, useState } from "react";
import { EpisodeStrip } from "@/components/generate/episode-strip";
import { GenerateForm, type GenerateSpec } from "@/components/generate/form";
import { IdeaCard } from "@/components/generate/idea-card";
import { WhatModelSaw } from "@/components/generate/what-model-saw";
import { Shell } from "@/components/shell";
import { Empty, ErrorNote, Panel } from "@/components/ui";
import type { Schemas, TrajectoryOut } from "@/lib/api/client";
import { useBriefs, useGenerate, useReview } from "@/lib/api/hooks";

type Result = Schemas["GenerateOut"];

export default function GeneratePage() {
  return (
    <Shell>
      <Suspense fallback={null}>
        <GenerateScreen />
      </Suspense>
    </Shell>
  );
}

function GenerateScreen() {
  const generate = useGenerate();
  const review = useReview();
  const briefs = useBriefs();
  const [spec, setSpec] = useState<GenerateSpec | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const brand = briefs.data?.find((b) => b.id === spec?.brief_id)?.company ?? "Brand";

  const run = async (s: GenerateSpec) => {
    setSpec(s);
    const backends = [s.backend, ...(s.compare_backend ? [s.compare_backend] : [])];
    const out: Result[] = [];
    for (const backend of backends) {
      out.push(await generate.mutateAsync({ brief_id: s.brief_id, episode_id: s.episode_id, backend, k: s.k, renders_per_idea: s.renders_per_idea, no_llm: s.no_llm }));
    }
    setResults(out);
  };

  const regenerate = async (t: TrajectoryOut) => {
    if (!spec) return;
    await review.mutateAsync({ id: t.id, body: { label: "skip", note: "regenerate" } });
    const replacement = await generate.mutateAsync({ brief_id: spec.brief_id, episode_id: null, backend: spec.backend, k: 1, renders_per_idea: spec.renders_per_idea, no_llm: spec.no_llm });
    setResults((rs) => rs.map((r, i) => (i === 0 ? { ...r, ideas: [...r.ideas.filter((x) => x.trajectory.id !== t.id), ...replacement.ideas] } : r)));
  };

  return (
    <div className="grid gap-3 lg:grid-cols-[320px_1fr]">
      <GenerateForm onGenerate={run} busy={generate.isPending} />
      <div className="min-w-0 space-y-3">
        {spec?.episode_id && <EpisodeStrip episodeId={spec.episode_id} />}
        <ErrorNote error={generate.error} />
        {results.length === 0 && !generate.isPending && <Empty>Pick a brief and generate. Ideas land here as Meta feed mocks; approve with Run.</Empty>}
        {generate.isPending && results.length === 0 && <Empty>Generating… the prompt carries the playbook, the brief, and the archive sample.</Empty>}
        {results.length > 0 && (
          <div className={`grid gap-3 ${results.length > 1 ? "xl:grid-cols-2" : ""}`}>
            {results.map((r, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
                  <span className="font-semibold text-foreground">{r.backend}</span>
                  <span>{r.model}</span>
                  <span>{r.ideas.filter((i) => i.status === "queued").length} queued / {r.ideas.length}</span>
                  <span>{r.input_tokens} in / {r.output_tokens} out tokens</span>
                  <span>{r.cost_usd !== null && r.cost_usd !== undefined ? `$${r.cost_usd.toFixed(4)}` : "cost n/a"}</span>
                  {r.refused && <span className="text-red-700">backend refused</span>}
                </div>
                <div className={`grid gap-3 ${results.length > 1 ? "" : "md:grid-cols-2 2xl:grid-cols-3"}`}>
                  {r.ideas.map((idea) => (
                    <IdeaCard key={idea.trajectory.id} idea={idea} brand={brand} rmKind={idea.trajectory.rm_version ?? undefined} onRegenerate={idx === 0 ? regenerate : undefined} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        {spec && <WhatModelSaw briefId={spec.brief_id} trace={(results[0]?.trace as Record<string, unknown>) ?? null} />}
        {!spec && <Panel title="How this works"><p className="text-xs text-muted">Column 1 is the playbook of expert cards, column 2 the brief, column 3 the ranked archive of past attempts with what the audience said. The generator proposes K ideas as a set with common, uncommon, and rare labels; the verifier tags cards independently; novelty rejection drops duplicates; the reward model ranks the queue; your Run label is the approval. Nothing spends money until the budget governor allows it.</p></Panel>}
      </div>
    </div>
  );
}
