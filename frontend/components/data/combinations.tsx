"use client";

import Link from "next/link";
import { useState } from "react";
import type { CombinationOut, Schemas } from "@/lib/api/client";
import { useCombinations, useNameAsCard } from "@/lib/api/hooks";
import { ago, pct } from "@/lib/format";
import { CardChip } from "../chips";
import { Badge, Button, Empty, ErrorNote, Field, Input, Modal, Panel, Select, Textarea } from "../ui";

export function CombinationsTab() {
  const [sort, setSort] = useState("tier2_rate");
  const combos = useCombinations(sort);
  const [naming, setNaming] = useState<CombinationOut | null>(null);
  return (
    <Panel
      testId="combinations-tab"
      title={<>Combinations <span className="text-muted">{combos.data?.length ?? 0} ever used</span></>}
      actions={
        <Select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="sort">
          <option value="tier2_rate">by tier 2 rate</option><option value="uses">by uses</option><option value="recent">by recency</option>
        </Select>
      }
    >
      {combos.data?.length === 0 && <Empty>No combinations yet.</Empty>}
      <table className="w-full text-left text-xs">
        <thead className="text-muted"><tr><th className="py-1">cards</th><th>niche</th><th>uses</th><th>measured</th><th>tier counts 0/1/2/3</th><th>tier 2 rate</th><th>last used</th><th></th></tr></thead>
        <tbody>
          {(combos.data ?? []).map((c) => {
            const measured = Object.values(c.tier_counts).reduce((a, b) => a + b, 0);
            return (
              <tr key={c.id} className="border-t border-line align-top" data-testid="combination-row">
                <td className="py-1"><div className="flex flex-wrap gap-1">{c.card_slugs.map((s) => <CardChip key={s} slug={s} />)}</div></td>
                <td className="font-mono text-muted">{c.niche_key}</td>
                <td>{c.uses}</td>
                <td>{measured}</td>
                <td>{[0, 1, 2, 3].map((t) => c.tier_counts[String(t)] ?? 0).join(" / ")}</td>
                <td>{c.tier2_rate !== null && c.tier2_rate !== undefined ? <Badge className={c.tier2_rate > 0 ? "tier-2" : "border border-line"}>{pct(c.tier2_rate)}</Badge> : "–"}</td>
                <td className="text-muted">{ago(c.last_used)}</td>
                <td>
                  {c.named_as_card_id ? (
                    <Link href={`?open=card:${c.named_as_card_id}`} scroll={false} className="text-accent hover:underline">named →</Link>
                  ) : (
                    <Button size="sm" onClick={() => setNaming(c)} data-testid="name-as-card">Name as card</Button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {naming && <NameAsCardModal combo={naming} onClose={() => setNaming(null)} />}
    </Panel>
  );
}

function NameAsCardModal({ combo, onClose }: { combo: CombinationOut; onClose: () => void }) {
  const name = useNameAsCard();
  const [form, setForm] = useState<Schemas["NameAsCardIn"]>({ slug: combo.card_slugs.join("-").slice(0, 60), name: combo.card_slugs.map((s) => s.replace(/-/g, " ")).join(" + "), kind: "strategy", definition: "", qualifying_condition: "" });
  return (
    <Modal title="Name this combination as a card" open onClose={onClose}>
      <p className="mb-2 text-xs text-muted">Creates a draft card. Cards: {combo.card_slugs.join(" + ")} · uses {combo.uses} · tiers {JSON.stringify(combo.tier_counts)}</p>
      <div className="space-y-2">
        <Field label="Slug"><Input data-testid="name-slug" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} /></Field>
        <Field label="Name"><Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
        <Field label="Kind"><Select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as Schemas["NameAsCardIn"]["kind"] })}><option value="strategy">strategy</option><option value="style">style</option><option value="principle">principle</option><option value="mechanic">mechanic</option></Select></Field>
        <Field label="Definition"><Textarea data-testid="name-definition" rows={3} value={form.definition} onChange={(e) => setForm({ ...form, definition: e.target.value })} /></Field>
        <ErrorNote error={name.error} />
        <div className="flex justify-end gap-2"><Button onClick={onClose}>Cancel</Button><Button variant="primary" data-testid="name-save" disabled={!form.definition || !form.slug} onClick={() => name.mutate({ id: combo.id, body: form }, { onSuccess: onClose })}>Create draft card</Button></div>
      </div>
    </Modal>
  );
}
