"use client";

import Link from "next/link";
import { useState } from "react";
import { useBriefEpisodes, useBriefs } from "@/lib/api/hooks";
import { ago, clamp, num, shortId } from "@/lib/format";
import { StatusBadge, TierBadge } from "../chips";
import { Badge, Button, Empty, Panel } from "../ui";

export function BriefsColumn({ selected, onSelect }: { selected: string | null; onSelect: (id: string | null) => void }) {
  const briefs = useBriefs();
  return (
    <Panel testId="briefs-column" title={<>Briefs <span className="text-muted">{briefs.data?.length ?? 0}</span></>} actions={selected && <Button size="sm" onClick={() => onSelect(null)}>clear filter</Button>} className="flex max-h-[calc(100vh-140px)] flex-col overflow-hidden">
      <div className="-m-3 overflow-auto p-3">
        {briefs.data?.length === 0 && <Empty>No briefs yet.</Empty>}
        <ul className="space-y-2">
          {(briefs.data ?? []).map((b) => (
            <BriefItem key={b.id} b={b} selected={selected === b.id} onSelect={() => onSelect(selected === b.id ? null : b.id)} />
          ))}
        </ul>
      </div>
    </Panel>
  );
}

function BriefItem({ b, selected, onSelect }: { b: NonNullable<ReturnType<typeof useBriefs>["data"]>[number]; selected: boolean; onSelect: () => void }) {
  const [open, setOpen] = useState(false);
  const episodes = useBriefEpisodes(open ? b.id : null);
  return (
    <li data-testid={`brief-${b.id}`} className={`rounded border p-2 text-xs ${selected ? "border-accent bg-accent-soft/40" : "border-line"}`}>
      <div className="flex items-start justify-between gap-2">
        <button onClick={onSelect} className="text-left">
          <div className="font-semibold">{b.company}</div>
          <div className="text-muted">{clamp(b.product, 60)}</div>
        </button>
        <Link href={`?open=brief:${b.id}`} scroll={false} className="font-mono text-[10px] text-muted hover:underline">{shortId(b.id)}</Link>
      </div>
      <p className="mt-1"><span className="text-muted">World:</span> {clamp(b.world_state || "nothing notable", 90)}</p>
      <div className="mt-1 flex flex-wrap gap-1">
        <Badge className="border border-line">{b.category}</Badge>
        <Badge className="border border-line">{b.trajectory_count} attempts</Badge>
        <Badge className="border border-line">{b.episode_count} episodes</Badge>
        {b.constraints.map((c) => <Badge key={c} className="border border-dashed border-line text-muted">{c}</Badge>)}
      </div>
      <button onClick={() => setOpen((o) => !o)} className="mt-1 text-accent hover:underline">{open ? "hide episodes" : "show episodes"}</button>
      {open && (
        <ul className="mt-1 space-y-1">
          {episodes.data?.length === 0 && <li className="text-muted">No episodes yet.</li>}
          {(episodes.data ?? []).map((e) => (
            <li key={e.id} className="flex items-center justify-between rounded border border-line px-2 py-1">
              <Link href={`?open=episode:${e.id}`} scroll={false} className="font-mono hover:underline">{shortId(e.id)}</Link>
              <span>${num(e.spent, 0)}/${num(e.budget_cap, 0)}</span>
              <TierBadge tier={e.best_tier} />
              <StatusBadge status={e.status} />
              <span className="text-muted">{ago(e.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}
