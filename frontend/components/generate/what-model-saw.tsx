"use client";

import Link from "next/link";
import { useArchiveSample } from "@/lib/api/hooks";
import { num } from "@/lib/format";
import { TierBadge } from "../chips";
import { Badge, Empty, Panel } from "../ui";

type Trace = Record<string, unknown> & {
  archive_refs?: { ref: string; reason: string; niche: string; tier: number | null }[];
  episode_refs?: { batch: number; ref: string; tier: number | null; screening_ratio: number | null }[];
  signal_refs?: string[];
  card_slugs?: string[];
  token_estimate?: number;
  trimmed?: string[];
  config_hash?: string;
};

/** The archive sample, episode history, and confirmed signals exactly as rendered into the prompt. */
export function WhatModelSaw({ briefId, trace }: { briefId: string | null; trace: Trace | null }) {
  const sample = useArchiveSample(briefId);
  const items = (sample.data?.items ?? []) as Array<Record<string, unknown> & { ref: string; reason: string; niche: string; tier: number | null; ratio: number | null; screening_ratio: number | null; angle: string; trajectory_id: string; brief_company: string; same_brief: boolean; signals: Array<{ ref: string; kind: string; text: string; count: number }> }>;
  const keptRefs = new Set((trace?.archive_refs ?? []).map((r) => r.ref));
  return (
    <Panel testId="what-model-saw" title="What the model saw" actions={trace && <span className="text-xs text-muted">~{trace.token_estimate} tokens · {trace.card_slugs?.length ?? 0} cards · config {trace.config_hash}{trace.trimmed?.length ? ` · trimmed ${trace.trimmed.length}` : ""}</span>}>
      {!briefId && <Empty>Pick a brief to see its archive sample.</Empty>}
      {briefId && items.length === 0 && !sample.isLoading && <Empty>No history for this brief or similar briefs yet.</Empty>}
      {items.length > 0 && (
        <div className="space-y-3 text-xs">
          <div className="flex flex-wrap gap-1">
            <Badge className="border border-line">{items.filter((i) => i.reason === "elite").length} best-per-niche</Badge>
            <Badge className="border border-line">{items.filter((i) => i.reason === "rare").length} rare niches</Badge>
            <Badge className="border border-line">{items.filter((i) => i.reason === "tier2_recent").length} recent tier 2+</Badge>
            <Badge className="border border-line">{sample.data?.similar_brief_ids.length ?? 0} similar briefs</Badge>
            {(sample.data?.excluded_holdout ?? 0) > 0 && <Badge className="bg-amber-100 text-amber-800">{sample.data?.excluded_holdout} held out</Badge>}
          </div>
          <table className="w-full text-left">
            <thead className="text-muted"><tr><th>ref</th><th>why</th><th>niche</th><th>result</th><th>angle</th><th>signals</th></tr></thead>
            <tbody>
              {items.map((i) => (
                <tr key={i.ref} className={`border-t border-line align-top ${trace && !keptRefs.has(i.ref) ? "opacity-50" : ""}`} title={trace && !keptRefs.has(i.ref) ? "trimmed from the prompt for token budget" : undefined}>
                  <td className="py-1 font-mono"><Link href={`?open=trajectory:${i.trajectory_id}`} scroll={false} className="hover:underline">{i.ref}</Link></td>
                  <td>{i.reason}{!i.same_brief && <span className="text-muted"> · {i.brief_company}</span>}</td>
                  <td className="font-mono text-muted">{i.niche}</td>
                  <td>{i.tier !== null ? <TierBadge tier={i.tier} ratio={i.ratio} /> : i.screening_ratio !== null ? <span className="text-muted">screen {num(i.screening_ratio)}x</span> : <span className="text-muted">pending</span>}</td>
                  <td className="max-w-xs truncate">{i.angle.split("\n")[0]}</td>
                  <td>{i.signals.map((s) => <div key={s.ref} className="truncate text-muted"><span className="font-mono">{s.kind}</span> {s.text} ×{s.count}</div>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {trace?.episode_refs && trace.episode_refs.length > 0 && (
            <div>
              <div className="mb-1 font-semibold uppercase text-muted">This search so far</div>
              <div className="flex flex-wrap gap-1">{trace.episode_refs.map((e) => <Badge key={e.ref} className="border border-line">batch {e.batch} · {e.ref} · {e.tier !== null ? `tier ${e.tier}` : e.screening_ratio !== null ? `screen ${num(e.screening_ratio)}x` : "pending"}</Badge>)}</div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
