"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { CardOut, Schemas } from "@/lib/api/client";
import { useCards, useCreateCard, useRelations, useSetCardStatus, useUpdateCard } from "@/lib/api/hooks";
import { clamp, pct } from "@/lib/format";
import { StatusBadge } from "../chips";
import { Badge, Button, Empty, ErrorNote, Field, Input, Modal, Panel, Select, Textarea } from "../ui";

const KINDS = ["strategy", "style", "principle", "mechanic"] as const;

export function PlaybookColumn() {
  const cards = useCards();
  const relations = useRelations();
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<CardOut | null>(null);
  const [status, setStatus] = useState<string>("all");
  const grouped = useMemo(() => {
    const out: Record<string, CardOut[]> = {};
    for (const c of cards.data ?? []) {
      if (status !== "all" && c.status !== status) continue;
      (out[c.kind] ??= []).push(c);
    }
    return out;
  }, [cards.data, status]);
  const relBy = useMemo(() => {
    const m = new Map<string, Schemas["RelationOut"][]>();
    for (const r of relations.data ?? []) {
      (m.get(r.from_id) ?? m.set(r.from_id, []).get(r.from_id)!).push(r);
      (m.get(r.to_id) ?? m.set(r.to_id, []).get(r.to_id)!).push(r);
    }
    return m;
  }, [relations.data]);
  const byId = useMemo(() => new Map((cards.data ?? []).map((c) => [c.id, c])), [cards.data]);

  return (
    <Panel
      testId="playbook-column"
      title={<>Playbook <span className="text-muted">{cards.data?.length ?? 0} cards</span></>}
      actions={
        <>
          <Select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="status filter">
            <option value="all">all</option><option value="active">active</option><option value="draft">draft</option><option value="retired">retired</option>
          </Select>
          <Button size="sm" variant="primary" data-testid="add-card" onClick={() => setAdding(true)}>Add card</Button>
        </>
      }
      className="flex max-h-[calc(100vh-140px)] flex-col overflow-hidden"
    >
      <div className="-m-3 overflow-auto p-3">
        {KINDS.map((kind) => (
          <div key={kind} className="mb-4">
            <h3 className="mb-1 text-[11px] font-semibold uppercase text-muted">{kind} · {grouped[kind]?.length ?? 0}</h3>
            <ul className="space-y-2">
              {(grouped[kind] ?? []).map((c) => (
                <li key={c.id} data-testid={`card-${c.slug}`} className="rounded border border-line p-2 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <Link href={`?open=card:${c.id}`} scroll={false} className="font-semibold hover:underline">{c.name}</Link>
                      <div className="font-mono text-[10px] text-muted">{c.slug} · v{c.version} · {c.contributed_by}</div>
                    </div>
                    <StatusBadge status={c.status} />
                  </div>
                  <p className="mt-1">{clamp(c.definition, 160)}</p>
                  {c.qualifying_condition && <p className="mt-0.5 text-muted">When: {clamp(c.qualifying_condition, 100)}</p>}
                  {c.source && <p className="text-[10px] text-muted">Source: {c.source}</p>}
                  <div className="mt-1 flex flex-wrap items-center gap-1">
                    <Badge className="border border-line">uses {c.stats.uses}</Badge>
                    <Badge className={c.stats.tier2_count > 0 ? "tier-2" : "border border-line"}>tier 2+ {c.stats.tier2_count} ({pct(c.stats.tier2_rate)})</Badge>
                    <Badge className="border border-line">verifier {pct(c.stats.verifier_agreement)}</Badge>
                  </div>
                  {(relBy.get(c.id) ?? []).length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {(relBy.get(c.id) ?? []).map((r) => {
                        const other = byId.get(r.from_id === c.id ? r.to_id : r.from_id);
                        return (
                          <span key={r.id} className={`rounded border px-1 text-[10px] ${r.source === "suggested" ? "border-dashed text-muted" : "border-line"}`} title={`${r.source} · ${r.status}`}>
                            {r.kind} {other?.slug ?? "?"}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  <div className="mt-1 flex gap-1">
                    <Button size="sm" variant="ghost" onClick={() => setEditing(c)} data-testid={`edit-${c.slug}`}>Edit</Button>
                    <StatusButton card={c} />
                  </div>
                </li>
              ))}
              {(grouped[kind] ?? []).length === 0 && <Empty>No {kind} cards{kind !== "strategy" ? " yet: experts write these from their sources" : ""}.</Empty>}
            </ul>
          </div>
        ))}
      </div>
      <CardEditor open={adding} onClose={() => setAdding(false)} />
      <CardEditor open={!!editing} card={editing ?? undefined} onClose={() => setEditing(null)} />
    </Panel>
  );
}

function StatusButton({ card }: { card: CardOut }) {
  const m = useSetCardStatus();
  if (card.status === "active") return <Button size="sm" variant="ghost" onClick={() => m.mutate({ id: card.id, status: "retired" })}>Retire</Button>;
  return <Button size="sm" variant="ghost" data-testid={`activate-${card.slug}`} onClick={() => m.mutate({ id: card.id, status: "active" })}>Activate</Button>;
}

export function CardEditor({ open, onClose, card }: { open: boolean; onClose: () => void; card?: CardOut }) {
  const create = useCreateCard();
  const update = useUpdateCard();
  const [form, setForm] = useState<Schemas["CardCreate"]>({
    slug: card?.slug ?? "",
    name: card?.name ?? "",
    kind: (card?.kind as Schemas["CardCreate"]["kind"]) ?? "style",
    definition: card?.definition ?? "",
    qualifying_condition: card?.qualifying_condition ?? "",
    source: card?.source ?? "",
    status: "draft",
  });
  const pending = create.isPending || update.isPending;
  const submit = () => {
    const body = { ...form, source: form.source || null };
    if (card) update.mutate({ id: card.id, body: { name: body.name, kind: body.kind, definition: body.definition, qualifying_condition: body.qualifying_condition, source: body.source } }, { onSuccess: onClose });
    else create.mutate(body, { onSuccess: onClose });
  };
  return (
    <Modal title={card ? `Edit ${card.name} (creates v${card.version + 1})` : "New card (saved as draft)"} open={open} onClose={onClose}>
      <div className="space-y-2">
        {!card && <Field label="Slug (kebab-case)"><Input data-testid="card-slug" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} placeholder="ogilvy-long-copy" /></Field>}
        <Field label="Name"><Input data-testid="card-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
        <Field label="Kind">
          <Select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as Schemas["CardCreate"]["kind"] })}>
            {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </Select>
        </Field>
        <Field label="Definition (one paragraph)"><Textarea data-testid="card-definition" rows={4} value={form.definition} onChange={(e) => setForm({ ...form, definition: e.target.value })} /></Field>
        <Field label="Qualifying condition (when it applies)"><Textarea rows={2} value={form.qualifying_condition} onChange={(e) => setForm({ ...form, qualifying_condition: e.target.value })} /></Field>
        <Field label="Source (e.g. Ogilvy on Advertising, ch. 7)"><Input value={form.source ?? ""} onChange={(e) => setForm({ ...form, source: e.target.value })} /></Field>
        <ErrorNote error={create.error ?? update.error} />
        <div className="flex justify-end gap-2"><Button onClick={onClose}>Cancel</Button><Button variant="primary" data-testid="save-card" disabled={pending || !form.name || !form.definition || (!card && !form.slug)} onClick={submit}>{card ? "Save new version" : "Create draft"}</Button></div>
      </div>
    </Modal>
  );
}
