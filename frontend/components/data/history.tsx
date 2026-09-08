"use client";

import { useState } from "react";
import { useCards, useTrajectories, type TrajectoryFilters } from "@/lib/api/hooks";
import { Button, Empty, Panel, Select } from "../ui";
import { TrajectoryCard } from "./trajectory-card";

const SIGNAL_KINDS = ["objection", "praise", "joke", "misreading", "question", "competitor_mention", "quote", "share_pattern", "comment_theme", "note"];

export function HistoryColumn({ briefId }: { briefId: string | null }) {
  const [filters, setFilters] = useState<TrajectoryFilters>({});
  const [offset, setOffset] = useState(0);
  const limit = 25;
  const cards = useCards();
  const q = useTrajectories({ ...filters, brief_id: briefId ?? undefined, limit, offset });
  const items = (q.data?.items ?? []) as Parameters<typeof TrajectoryCard>[0]["trajectory"][];
  const set = (patch: Partial<TrajectoryFilters>) => {
    setFilters((f) => ({ ...f, ...patch }));
    setOffset(0);
  };
  return (
    <Panel
      testId="history-column"
      title={<>History <span className="text-muted">{q.data?.total ?? 0}</span></>}
      className="flex max-h-[calc(100vh-140px)] flex-col overflow-hidden"
      actions={
        <div className="flex flex-wrap gap-1">
          <Select aria-label="tier" value={filters.min_tier ?? ""} onChange={(e) => set({ min_tier: e.target.value === "" ? undefined : Number(e.target.value) })}>
            <option value="">any tier</option><option value="0">measured</option><option value="1">tier 1+</option><option value="2">tier 2+</option><option value="3">tier 3</option>
          </Select>
          <Select aria-label="typicality" value={filters.typicality ?? ""} onChange={(e) => set({ typicality: e.target.value || undefined })}>
            <option value="">any typicality</option><option value="common">common</option><option value="uncommon">uncommon</option><option value="rare">rare</option>
          </Select>
          <Select aria-label="card" value={filters.card ?? ""} onChange={(e) => set({ card: e.target.value || undefined })}>
            <option value="">any card</option>
            {(cards.data ?? []).map((c) => <option key={c.id} value={c.slug}>{c.slug}</option>)}
          </Select>
          <Select aria-label="signal kind" value={filters.signal_kind ?? ""} onChange={(e) => set({ signal_kind: e.target.value || undefined })}>
            <option value="">any signal</option>
            {SIGNAL_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </Select>
          <Select aria-label="review" value={filters.review_label ?? (filters.unlabeled ? "unlabeled" : "")} onChange={(e) => { const v = e.target.value; set({ review_label: v && v !== "unlabeled" ? v : undefined, unlabeled: v === "unlabeled" ? true : undefined }); }}>
            <option value="">any review</option><option value="unlabeled">unlabeled</option><option value="run">run</option><option value="skip">skip</option><option value="wrong_cards">wrong cards</option>
          </Select>
          <label className="flex items-center gap-1 text-xs"><input type="checkbox" checked={!!filters.mismatch} onChange={(e) => set({ mismatch: e.target.checked ? true : undefined })} /> mismatch</label>
        </div>
      }
    >
      <div className="-m-3 space-y-2 overflow-auto p-3">
        {q.isLoading && <Empty>Loading…</Empty>}
        {!q.isLoading && items.length === 0 && <Empty>No trajectories match.</Empty>}
        {items.map((t) => t && <TrajectoryCard key={t.id} trajectory={t} />)}
        {(q.data?.total ?? 0) > limit && (
          <div className="flex items-center justify-between text-xs text-muted">
            <Button size="sm" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - limit))}>← newer</Button>
            <span>{offset + 1}–{Math.min(offset + limit, q.data?.total ?? 0)} of {q.data?.total}</span>
            <Button size="sm" disabled={offset + limit >= (q.data?.total ?? 0)} onClick={() => setOffset((o) => o + limit)}>older →</Button>
          </div>
        )}
      </div>
    </Panel>
  );
}
