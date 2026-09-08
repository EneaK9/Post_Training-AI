"use client";

import Link from "next/link";
import { useAccounts, useEpisode, useShipEpisode, useSimulateEpisode, useStepEpisode, useStopEpisode } from "@/lib/api/hooks";
import { num, shortId } from "@/lib/format";
import { StatusBadge, TierBadge } from "../chips";
import { Badge, Button, ErrorNote } from "../ui";

export function EpisodeStrip({ episodeId, onBatchGenerated }: { episodeId: string; onBatchGenerated?: () => void }) {
  const ep = useEpisode(episodeId);
  const accounts = useAccounts();
  const step = useStepEpisode();
  const simulate = useSimulateEpisode();
  const stop = useStopEpisode();
  const ship = useShipEpisode();
  const busy = step.isPending || simulate.isPending || stop.isPending || ship.isPending;
  if (!ep.data) return null;
  const e = ep.data;
  const account = accounts.data?.find((a) => a.id === e.ad_account_id);
  const blocked = account ? account.status !== "active" : false;
  const pct = Math.min(100, (e.spent / e.budget_cap) * 100);
  const found = e.status === "outlier_found";
  return (
    <div data-testid="episode-strip" className={`rounded-lg border p-3 text-xs ${found ? "border-green-500 bg-green-50 dark:bg-green-950/30" : "border-line bg-panel"}`}>
      <div className="flex flex-wrap items-center gap-3">
        <Link href={`?open=episode:${e.id}`} scroll={false} className="font-mono text-sm hover:underline">episode {shortId(e.id)}</Link>
        {found ? <Badge className="tier-2 text-sm">Outlier found</Badge> : <StatusBadge status={e.status} />}
        <span className="text-muted">{e.backend} · {e.batches.length} batches · {e.live_ads} live ads</span>
        <TierBadge tier={e.best_tier} />
        <Badge className="border border-line">{e.new_signals} new signals since last batch</Badge>
        {account && <Badge className={account.status === "active" ? "border border-line" : "bg-red-100 text-red-800"}>{account.name} · {account.status}</Badge>}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded bg-line"><div className="h-full bg-accent" style={{ width: `${pct}%` }} /></div>
        <span data-testid="episode-spent">${num(e.spent, 0)} / ${num(e.budget_cap, 0)}</span>
      </div>
      {blocked && <p className="mt-2 text-red-700">This account is {account?.status}. Reconnect it on the Model screen before running ads; generation still works.</p>}
      <div className="mt-2 flex flex-wrap items-center gap-1">
        <Button size="sm" variant="primary" disabled={e.status !== "searching" || busy} data-testid="next-batch" onClick={() => step.mutate({ id: e.id, approval: "manual", no_llm: true }, { onSuccess: onBatchGenerated })} title="generate the next batch into the queue; you approve">Next batch</Button>
        <Button size="sm" disabled={e.status !== "searching" || blocked || busy} data-testid="auto-batch" onClick={() => step.mutate({ id: e.id, approval: "top_rm", no_llm: true }, { onSuccess: onBatchGenerated })} title="let the reward model pick and ship the top ideas">Auto batch (rm)</Button>
        <Button size="sm" disabled={e.status !== "searching" || blocked || busy} data-testid="ship-approved" onClick={() => ship.mutate({ id: e.id })} title="ship every idea you labeled Run">Ship approved</Button>
        {account?.is_fake && (
          <>
            <Button size="sm" disabled={busy} data-testid="simulate-1" onClick={() => simulate.mutate({ id: e.id, days: 1 })}>Simulate 1 day</Button>
            <Button size="sm" disabled={busy} data-testid="simulate-7" onClick={() => simulate.mutate({ id: e.id, days: 7 })}>Simulate 7 days</Button>
          </>
        )}
        <Button size="sm" variant="danger" disabled={e.status !== "searching"} onClick={() => stop.mutate({ id: e.id })}>Stop</Button>
        {busy && <span className="text-muted">working…</span>}
      </div>
      <ErrorNote error={step.error ?? simulate.error ?? stop.error ?? ship.error} />
    </div>
  );
}
