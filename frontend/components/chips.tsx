"use client";

import clsx from "clsx";
import Link from "next/link";
import { Badge } from "./ui";

export function TierBadge({ tier, ratio }: { tier: number | null | undefined; ratio?: number | null }) {
  if (tier === null || tier === undefined) return <Badge className="bg-line text-muted">pending</Badge>;
  return (
    <Badge className={`tier-${tier}`} title={ratio ? `${ratio.toFixed(2)}x median` : undefined}>
      tier {tier}
      {ratio ? ` · ${ratio.toFixed(1)}x` : ""}
    </Badge>
  );
}

export function CardChip({ slug, dim, flagged }: { slug: string; dim?: boolean; flagged?: boolean }) {
  return (
    <Link
      href={`?open=card-slug:${slug}`}
      scroll={false}
      className={clsx(
        "rounded border px-1.5 py-0.5 font-mono text-[11px] hover:bg-accent-soft",
        dim ? "border-dashed border-line text-muted" : "border-line",
        flagged && "border-amber-500 text-amber-700 dark:text-amber-300",
      )}
    >
      {slug}
    </Link>
  );
}

export function CardChips({ written, verified }: { written: string[]; verified: string[] }) {
  const w = new Set(written);
  const v = new Set(verified);
  const same = written.length === verified.length && written.every((s) => v.has(s));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[10px] uppercase text-muted">written</span>
        {written.map((s) => <CardChip key={s} slug={s} flagged={!same && !v.has(s)} />)}
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[10px] uppercase text-muted">verified</span>
        {verified.length === 0 && <span className="text-[11px] text-muted">none</span>}
        {verified.map((s) => <CardChip key={s} slug={s} dim flagged={!same && !w.has(s)} />)}
        {!same && <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">mismatch</Badge>}
      </div>
    </div>
  );
}

const SENTIMENT: Record<string, string> = {
  neg: "border-red-300 bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-200",
  pos: "border-green-300 bg-green-50 text-green-800 dark:bg-green-950/40 dark:text-green-200",
  neu: "border-line bg-panel",
};

export function SignalChip({
  signal,
  onConfirm,
  onReject,
}: {
  signal: { id: string; kind: string; text: string; sentiment: string; count: number; status: string };
  onConfirm?: () => void;
  onReject?: () => void;
}) {
  const proposed = signal.status === "proposed";
  return (
    <span
      data-testid={`signal-${signal.status}`}
      className={clsx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]", SENTIMENT[signal.sentiment] ?? SENTIMENT.neu, proposed && "opacity-70 border-dashed")}
      title={`${signal.kind} · ${signal.status}`}
    >
      <Link href={`?open=signal:${signal.id}`} scroll={false} className="hover:underline">
        <span className="text-muted">{signal.kind}</span> {signal.text} <span className="text-muted">×{signal.count}</span>
      </Link>
      {proposed && onConfirm && (
        <button onClick={onConfirm} aria-label="confirm signal" className="rounded px-1 hover:bg-green-200/60">✓</button>
      )}
      {proposed && onReject && (
        <button onClick={onReject} aria-label="reject signal" className="rounded px-1 hover:bg-red-200/60">✗</button>
      )}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const cls: Record<string, string> = {
    active: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200",
    draft: "bg-line text-muted",
    retired: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200",
    searching: "bg-accent-soft text-accent",
    outlier_found: "tier-2",
    budget_exhausted: "bg-line text-muted",
    stopped: "bg-line text-muted",
    run: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200",
    skip: "bg-line text-muted",
    wrong_cards: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
  };
  return <Badge className={cls[status] ?? "bg-line"}>{status.replace("_", " ")}</Badge>;
}
