"use client";

import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { useCards, useCombinations, useRelations } from "@/lib/api/hooks";
import { Panel } from "../ui";

const KIND_COLOR: Record<string, string> = { strategy: "#2f5bea", style: "#0ea5e9", principle: "#16a34a", mechanic: "#f59e0b" };

/** Cards as nodes sized by tier 2+ count, relations as edges (dashed when suggested), the most
 * used combinations as hyperedge nodes connected to their cards. */
export function GraphTab() {
  const cards = useCards();
  const relations = useRelations();
  const combos = useCombinations("uses");
  const { nodes, edges } = useMemo(() => {
    const cs = cards.data ?? [];
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const n = Math.max(cs.length, 1);
    const R = 360;
    cs.forEach((c, i) => {
      const a = (2 * Math.PI * i) / n;
      const size = 28 + Math.min(c.stats.tier2_count, 10) * 6;
      nodes.push({
        id: c.id,
        position: { x: 500 + R * Math.cos(a), y: 400 + R * Math.sin(a) },
        data: { label: c.slug },
        style: { width: size + 60, fontSize: 10, borderColor: KIND_COLOR[c.kind] ?? "#999", borderWidth: 2, opacity: c.status === "active" ? 1 : 0.5, borderRadius: 8, padding: 4 },
      });
    });
    for (const r of relations.data ?? []) {
      if (r.status === "rejected") continue;
      edges.push({ id: r.id, source: r.from_id, target: r.to_id, label: r.kind, style: { strokeDasharray: r.source === "suggested" ? "4 4" : undefined }, animated: false });
    }
    (combos.data ?? []).slice(0, 25).forEach((c, i) => {
      const a = (2 * Math.PI * i) / 25;
      const id = `combo-${c.id}`;
      const tier2 = (c.tier_counts["2"] ?? 0) + (c.tier_counts["3"] ?? 0);
      nodes.push({ id, position: { x: 500 + 160 * Math.cos(a), y: 400 + 160 * Math.sin(a) }, data: { label: `${c.uses}× · t2 ${tier2}` }, style: { fontSize: 9, background: tier2 > 0 ? "#dcfce7" : "#f3f4f6", borderRadius: 999, padding: 2, width: 70 } });
      for (const cid of c.card_ids) edges.push({ id: `${id}-${cid}`, source: id, target: cid, style: { stroke: tier2 > 0 ? "#16a34a" : "#cbd5e1", strokeWidth: tier2 > 0 ? 2 : 1 } });
    });
    return { nodes, edges };
  }, [cards.data, relations.data, combos.data]);
  return (
    <Panel testId="graph-tab" title="Graph: cards, relations, top combinations">
      <div className="h-[70vh] rounded border border-line">
        <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
          <Background />
          <Controls />
        </ReactFlow>
      </div>
      <p className="mt-2 text-xs text-muted">Node size = tier 2+ count. Border color = kind (blue strategy, cyan style, green principle, amber mechanic). Dashed edges are suggested relations awaiting an expert.</p>
    </Panel>
  );
}
