"use client";

import Link from "next/link";
import { useReviewQueue, useSignalDecision, useSignalsQueue } from "@/lib/api/hooks";
import { shortId } from "@/lib/format";
import { SignalChip } from "../chips";
import { Empty, Panel } from "../ui";
import { TrajectoryCard } from "./trajectory-card";

export function ReviewQueueTab() {
  const q = useReviewQueue();
  return (
    <Panel testId="review-queue" title={<>Review queue <span className="text-muted">{q.data?.length ?? 0} unlabeled, sorted by rm_score</span></>}>
      {q.data?.length === 0 && <Empty>Nothing waiting. Generate ideas from the Generate screen.</Empty>}
      <div className="space-y-2">{(q.data ?? []).map((t) => <TrajectoryCard key={t.id} trajectory={t} />)}</div>
    </Panel>
  );
}

export function SignalsQueueTab() {
  const q = useSignalsQueue();
  const decide = useSignalDecision();
  const groups = new Map<string, NonNullable<typeof q.data>>();
  for (const s of q.data ?? []) (groups.get(s.trajectory_id) ?? groups.set(s.trajectory_id, []).get(s.trajectory_id)!).push(s);
  return (
    <Panel testId="signals-queue" title={<>Signals queue <span className="text-muted">{q.data?.length ?? 0} proposed</span></>}>
      {q.data?.length === 0 && <Empty>No proposed signals. Only confirmed signals reach prompts and the reward model.</Empty>}
      <div className="space-y-3">
        {[...groups.entries()].map(([tid, sigs]) => (
          <div key={tid} className="rounded border border-line p-2">
            <Link href={`?open=trajectory:${tid}`} scroll={false} className="font-mono text-xs text-accent hover:underline">trajectory {shortId(tid)}</Link>
            <div className="mt-1 flex flex-wrap gap-1">
              {sigs.map((s) => (
                <SignalChip key={s.id} signal={s} onConfirm={() => decide.mutate({ id: s.id, decision: "confirm" })} onReject={() => decide.mutate({ id: s.id, decision: "reject" })} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
