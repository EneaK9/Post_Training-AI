"use client";

import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { fileUrl, type Schemas, type TrajectoryOut } from "@/lib/api/client";
import { useTrajectory } from "@/lib/api/hooks";
import { num } from "@/lib/format";
import { CardChips } from "../chips";
import { TrajectoryActions } from "../data/trajectory-card";
import { Badge, Button } from "../ui";

type Idea = Schemas["GeneratedIdeaOut"];

function linkify(text: string, citedIds: string[]) {
  return text.split(/(\[(?:card|history|signal):[^\]]+\])/g).map((p, i) => {
    const m = p.match(/^\[(card|history|signal):([^\]]+)\]$/);
    if (!m) return <span key={i}>{p}</span>;
    const href = m[1] === "card" ? `?open=card-slug:${m[2]}` : `?open=ref:${m[2]},${citedIds.join(",")}`;
    return <Link key={i} href={href} scroll={false} className="rounded bg-accent-soft px-1 font-mono text-[11px] text-accent hover:underline">{p}</Link>;
  });
}

/** One idea as a Meta feed ad mock with the model's reasoning and the review actions. */
export function IdeaCard({ idea, brand, rmKind, onRegenerate }: { idea: Idea; brand: string; rmKind?: string; onRegenerate?: (t: TrajectoryOut) => void }) {
  // the generate response is a snapshot; reviews and copy edits refetch the live row
  const live = useTrajectory(idea.trajectory.id);
  const t: TrajectoryOut = live.data ?? idea.trajectory;
  const [seed, setSeed] = useState(0);
  const [showReasoning, setShowReasoning] = useState(false);
  const render = t.renders.find((r) => r.seed === seed) ?? t.renders[0];
  const url = fileUrl(render?.image_uri);
  if (idea.status !== "queued") {
    return (
      <details data-testid="idea-rejected" className="rounded-lg border border-dashed border-line bg-panel px-3 py-2 text-xs text-muted">
        <summary className="cursor-pointer">
          {idea.status === "novelty_rejected" ? "Novelty rejected" : "Malformed output"}: {t.copy.headline || t.card_slugs.join(" + ")}
          {idea.novelty_reason && <span className="ml-2 font-mono">{idea.novelty_reason} d={num(idea.novelty_distance)}</span>}
        </summary>
        <div className="mt-2 space-y-1">
          <CardChips written={t.card_slugs} verified={t.verified_card_slugs} />
          {t.format_errors.length > 0 && <ul className="list-disc pl-4">{t.format_errors.map((e) => <li key={e}>{e}</li>)}</ul>}
          <p className="whitespace-pre-line">{t.angle}</p>
        </div>
      </details>
    );
  }
  return (
    <article data-testid="idea-card" className="flex flex-col rounded-lg border border-line bg-panel">
      <header className="flex items-center justify-between px-3 pt-3">
        <div>
          <div className="text-sm font-semibold">{brand}</div>
          <div className="text-[11px] text-muted">Sponsored · idea #{t.attempt_index}</div>
        </div>
        <div className="flex items-center gap-1">
          {t.typicality && <Badge className={clsx("border", t.typicality === "rare" ? "border-accent text-accent" : "border-line")}>{t.typicality}</Badge>}
          {t.rm_score !== null && t.rm_score !== undefined && <Badge className="bg-accent-soft text-accent" title={`${rmKind ?? t.rm_version}`}>rm {num(t.rm_score)}</Badge>}
          {t.review && <Badge className="border border-line">{t.review.label}</Badge>}
        </div>
      </header>
      <p className="px-3 py-2 text-sm">{t.copy.primary_text}</p>
      <div className="relative bg-background">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={t.copy.headline} className="aspect-square w-full object-cover" />
        ) : (
          <div className="flex aspect-square items-center justify-center text-xs text-muted">no render</div>
        )}
        {t.renders.length > 1 && (
          <div className="absolute bottom-2 right-2 flex gap-1">
            {t.renders.map((r) => (
              <button key={r.id} onClick={() => setSeed(r.seed)} className={clsx("h-6 w-6 rounded-full border text-[10px]", r.seed === seed ? "bg-accent text-white" : "bg-panel")} aria-label={`render ${r.seed}`}>{r.seed + 1}</button>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 border-t border-line bg-background px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{t.copy.headline}</div>
          <div className="truncate text-xs text-muted">{t.copy.description}</div>
        </div>
        <span className="shrink-0 rounded border border-line px-2 py-1 text-xs font-medium">{t.copy.cta || "Learn more"}</span>
      </div>
      <div className="space-y-2 px-3 py-2 text-xs">
        <CardChips written={t.card_slugs} verified={t.verified_card_slugs} />
        <p className="whitespace-pre-line text-muted">{t.angle}</p>
        <div className="flex flex-wrap gap-1 text-[11px] text-muted">
          <span>novelty {idea.novelty_reason ?? "–"} {idea.novelty_distance !== null && idea.novelty_distance !== undefined ? `d=${num(idea.novelty_distance)}` : ""}</span>
          <span>· verifier {idea.verifier_version}</span>
          {t.preship && !(t.preship.policy_ok && t.preship.brand_ok) && <Badge className="bg-red-100 text-red-800">pre-ship flagged</Badge>}
          {t.preship?.length_warnings.map((w) => <Badge key={w} className="border border-line">{w}</Badge>)}
        </div>
        <button onClick={() => setShowReasoning((s) => !s)} className="text-accent hover:underline">{showReasoning ? "hide reasoning" : "why this combination"}</button>
        {showReasoning && <p className="whitespace-pre-line rounded bg-background p-2">{linkify(t.reasoning, t.cited_ids)}</p>}
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-1 border-t border-line px-3 py-2">
        <TrajectoryActions t={t} />
        {onRegenerate && <Button size="sm" variant="ghost" onClick={() => onRegenerate(t)}>Regenerate</Button>}
      </footer>
    </article>
  );
}
