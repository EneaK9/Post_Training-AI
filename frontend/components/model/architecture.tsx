"use client";

import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { useArchitecture } from "@/lib/api/hooks";
import { Badge, Panel } from "../ui";

type ArchNode = { id: string; group: string; label: string; detail: string; config: string | null };
type Arch = { nodes: ArchNode[]; edges: { source: string; target: string }[]; config_hash: string; dry_run: boolean };

const GROUP_X: Record<string, number> = { loop_a: 0, evaluator: 420, loop_b: 840 };
const GROUP_COLOR: Record<string, string> = { loop_a: "#2f5bea", evaluator: "#16a34a", loop_b: "#7c3aed" };

/** Live architecture diagram from config. Clicking a node opens its config section. */
export function ArchitectureDiagram({ onOpenConfig }: { onOpenConfig: (section: string) => void }) {
  const arch = useArchitecture();
  const data = arch.data as Arch | undefined;
  const { nodes, edges } = useMemo(() => {
    if (!data) return { nodes: [] as Node[], edges: [] as Edge[] };
    const counters: Record<string, number> = {};
    const nodes: Node[] = data.nodes.map((n) => {
      const i = (counters[n.group] = (counters[n.group] ?? 0) + 1);
      return {
        id: n.id,
        position: { x: GROUP_X[n.group] ?? 0, y: i * 92 },
        data: { label: n.label },
        style: { width: 300, fontSize: 11, borderColor: GROUP_COLOR[n.group], borderWidth: 2, borderRadius: 8, padding: 6, textAlign: "left" as const },
      };
    });
    const edges: Edge[] = data.edges.map((e) => ({ id: `${e.source}-${e.target}`, source: e.source, target: e.target, animated: e.source === "policy" }));
    return { nodes, edges };
  }, [data]);
  const byId = useMemo(() => new Map((data?.nodes ?? []).map((n) => [n.id, n])), [data]);
  return (
    <Panel
      testId="architecture"
      title="Architecture (live from config)"
      actions={data && <><Badge className="border border-line">config {data.config_hash}</Badge><Badge className={data.dry_run ? "bg-amber-100 text-amber-800" : "bg-red-100 text-red-800"}>{data.dry_run ? "DRY RUN" : "LIVE SPEND"}</Badge></>}
    >
      <div className="grid grid-cols-3 gap-2 pb-2 text-xs font-semibold"><span style={{ color: GROUP_COLOR.loop_a }}>Loop A: search</span><span style={{ color: GROUP_COLOR.evaluator }}>Evaluator: Meta</span><span style={{ color: GROUP_COLOR.loop_b }}>Loop B: post-training</span></div>
      <div className="h-[520px] rounded border border-line">
        <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }} onNodeClick={(_, n) => { const a = byId.get(n.id); if (a?.config) onOpenConfig(a.config); }} nodesDraggable={false}>
          <Background />
          <Controls />
        </ReactFlow>
      </div>
      <ul className="mt-2 grid gap-1 text-xs md:grid-cols-2 xl:grid-cols-3">
        {(data?.nodes ?? []).map((n) => (
          <li key={n.id} className="rounded border border-line px-2 py-1">
            <button onClick={() => n.config && onOpenConfig(n.config)} className="font-medium hover:underline" style={{ color: GROUP_COLOR[n.group] }}>{n.label}</button>
            <div className="text-muted">{n.detail}</div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
