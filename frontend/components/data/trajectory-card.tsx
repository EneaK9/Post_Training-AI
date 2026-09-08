"use client";

import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { fileUrl, type TrajectoryDetail, type TrajectoryOut } from "@/lib/api/client";
import { useAddNote, useCards, useEditCopy, useReview, useSignalDecision, useTrajectory } from "@/lib/api/hooks";
import { ago, clamp, num, shortId } from "@/lib/format";
import { CardChips, SignalChip, StatusBadge, TierBadge } from "../chips";
import { Badge, Button, ErrorNote, Field, Input, Modal, Textarea } from "../ui";

type Props = { trajectory?: TrajectoryOut; id?: string; expanded?: boolean; showActions?: boolean };

export function TrajectoryCard({ trajectory, id, expanded: initialExpanded = false, showActions = true }: Props) {
  const fetched = useTrajectory(trajectory ? null : (id ?? null));
  const t: TrajectoryOut | TrajectoryDetail | undefined = trajectory ?? fetched.data;
  const [expanded, setExpanded] = useState(initialExpanded);
  const detail = useTrajectory(expanded && t ? t.id : null);
  if (!t) return <div className="rounded border border-line p-3 text-xs text-muted">Loading…</div>;
  const full = detail.data ?? (t as TrajectoryDetail);
  const labeled = t.review?.label;
  return (
    <article data-testid="trajectory-card" className="rounded-lg border border-line bg-panel p-3 text-sm">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs text-muted">
          <Link href={`?open=trajectory:${t.id}`} scroll={false} className="font-mono hover:underline">{shortId(t.id)}</Link>
          <span>#{t.attempt_index}</span>
          <span>{t.author_kind === "model" ? `model · ${t.backend ?? ""}` : t.author_id}</span>
          <span>{ago(t.created_at)}</span>
          {t.typicality && <Badge className="border border-line">{t.typicality}</Badge>}
          {!t.format_ok && <Badge className="bg-red-100 text-red-800">malformed</Badge>}
        </div>
        <div className="flex items-center gap-2">
          {t.screening_ratio !== null && t.screening_ratio !== undefined && (
            <Badge className={t.screening_ratio >= 1.5 ? "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200" : "bg-line"} title="CTR lower bound / account median CTR">screen {num(t.screening_ratio)}x</Badge>
          )}
          <TierBadge tier={t.outlier_tier} ratio={t.ratio} />
          {t.rm_score !== null && t.rm_score !== undefined && <Badge className="bg-accent-soft text-accent" title={t.rm_version ?? ""}>rm {num(t.rm_score)}</Badge>}
          {labeled && <StatusBadge status={labeled} />}
        </div>
      </header>
      <div className="mt-2 grid gap-3 md:grid-cols-[1fr_auto]">
        <div className="min-w-0 space-y-2">
          <CardChips written={t.card_slugs} verified={t.verified_card_slugs} />
          <p className="whitespace-pre-line text-xs text-muted">{t.angle}</p>
          <div>
            <div className="font-medium">{t.copy.headline || <span className="text-muted">no headline</span>}</div>
            <p className="text-xs">{expanded ? t.copy.primary_text : clamp(t.copy.primary_text, 180)}</p>
            {expanded && <p className="text-xs text-muted">{t.copy.description} · {t.copy.cta}</p>}
          </div>
          <div className="flex flex-wrap gap-1">
            {t.signals.map((s) => <SignalRow key={s.id} signal={s} />)}
          </div>
        </div>
        <div className="flex gap-1">
          {t.renders.map((r) => {
            const url = fileUrl(r.image_uri);
            return (
              <figure key={r.id} className="w-16 text-center">
                {url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={url} alt={`render ${r.seed}`} className="h-16 w-16 rounded border border-line object-cover" />
                ) : (
                  <div className="h-16 w-16 rounded border border-dashed border-line" />
                )}
                <figcaption className={clsx("mt-0.5 text-[10px]", r.status === "rejected" ? "text-red-600" : "text-muted")} title={r.rejection_reason ?? undefined}>{r.status}</figcaption>
              </figure>
            );
          })}
        </div>
      </div>
      <footer className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <button onClick={() => setExpanded((e) => !e)} className="text-xs text-accent hover:underline" data-testid="toggle-expand">{expanded ? "collapse" : "expand"}</button>
        {showActions && <TrajectoryActions t={t} />}
      </footer>
      {expanded && <Expanded t={full} />}
    </article>
  );
}

function SignalRow({ signal }: { signal: TrajectoryOut["signals"][number] }) {
  const decide = useSignalDecision();
  return (
    <SignalChip
      signal={signal}
      onConfirm={() => decide.mutate({ id: signal.id, decision: "confirm" })}
      onReject={() => decide.mutate({ id: signal.id, decision: "reject" })}
    />
  );
}

function linkify(text: string, citedIds: string[]) {
  const parts = text.split(/(\[(?:card|history|signal):[^\]]+\])/g);
  return parts.map((p, i) => {
    const m = p.match(/^\[(card|history|signal):([^\]]+)\]$/);
    if (!m) return <span key={i}>{p}</span>;
    const href = m[1] === "card" ? `?open=card-slug:${m[2]}` : `?open=ref:${m[2]},${citedIds.join(",")}`;
    return (
      <Link key={i} href={href} scroll={false} className="rounded bg-accent-soft px-1 font-mono text-[11px] text-accent hover:underline">{p}</Link>
    );
  });
}

function Expanded({ t }: { t: TrajectoryDetail | TrajectoryOut }) {
  const addNote = useAddNote();
  const [note, setNote] = useState("");
  const comments = "comments" in t ? t.comments : [];
  return (
    <div className="mt-3 space-y-3 border-t border-line pt-3 text-xs">
      <section>
        <h4 className="mb-1 font-semibold uppercase text-muted">Reasoning</h4>
        <p className="whitespace-pre-line">{linkify(t.reasoning || "—", t.cited_ids)}</p>
      </section>
      <section>
        <h4 className="mb-1 font-semibold uppercase text-muted">Visual brief</h4>
        <p>{t.visual_brief || "—"}</p>
      </section>
      {t.preship && (
        <section>
          <h4 className="mb-1 font-semibold uppercase text-muted">Pre-ship</h4>
          <div className="flex flex-wrap gap-1">
            <Badge className={t.preship.policy_ok ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}>policy {t.preship.policy_ok ? "ok" : "flagged"}</Badge>
            <Badge className={t.preship.brand_ok ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}>brand {t.preship.brand_ok ? "ok" : "flagged"}</Badge>
            {[...t.preship.policy_flags, ...t.preship.brand_flags, ...t.preship.length_warnings].map((f) => <Badge key={f} className="border border-line">{f}</Badge>)}
          </div>
        </section>
      )}
      {t.renders.some((r) => r.screening || r.outcome) && (
        <section>
          <h4 className="mb-1 font-semibold uppercase text-muted">Measurement</h4>
          <table className="w-full text-left">
            <thead className="text-muted"><tr><th>render</th><th>impr</th><th>ctr</th><th>screen</th><th>roas</th><th>ratio (lb)</th><th>tier</th><th>gates</th></tr></thead>
            <tbody>
              {t.renders.map((r) => (
                <tr key={r.id} className="border-t border-line">
                  <td>{r.seed} · {r.status}</td>
                  <td>{r.screening?.impressions ?? "–"}</td>
                  <td>{r.screening ? (r.screening.ctr * 100).toFixed(2) + "%" : "–"}</td>
                  <td>{r.screening ? `${num(r.screening.screening_ratio)}x ${r.screening.passed ? "✓" : "✗"}` : "–"}</td>
                  <td>{r.outcome ? num(r.outcome.value) : "–"}</td>
                  <td>{r.outcome ? `${num(r.outcome.ratio)} (${num(r.outcome.ratio_lower_bound)})` : "–"}</td>
                  <td>{r.outcome ? r.outcome.outlier_tier : "–"}</td>
                  <td>{r.outcome ? Object.entries(r.outcome.gates).map(([k, v]) => `${k.replace("_ok", "")}${v ? "✓" : "✗"}`).join(" ") : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <section>
        <h4 className="mb-1 font-semibold uppercase text-muted">What the audience said ({comments.length} comments)</h4>
        {t.signals.length === 0 && comments.length === 0 && <p className="text-muted">Nothing yet.</p>}
        {t.signals.map((s) => (
          <div key={s.id} className="mb-2 rounded border border-line p-2">
            <div className="flex items-center gap-2"><SignalChip signal={s} /></div>
            <ul className="mt-1 list-disc pl-4 text-muted">{s.evidence.slice(0, 3).map((e, i) => <li key={i}>“{String(e.excerpt)}”</li>)}</ul>
          </div>
        ))}
        {comments.length > 0 && (
          <details>
            <summary className="cursor-pointer text-muted">all comments</summary>
            <ul className="mt-1 max-h-48 space-y-0.5 overflow-auto">{comments.map((c) => <li key={c.id} className="rounded bg-background px-2 py-0.5">{c.text} <span className="text-muted">{ago(c.created_time)}</span></li>)}</ul>
          </details>
        )}
      </section>
      <section>
        <h4 className="mb-1 font-semibold uppercase text-muted">Notes</h4>
        <ul className="space-y-1">{t.notes.map((n) => <li key={n.id}><span className="text-muted">{n.author_id} · {ago(n.created_at)}:</span> {n.text}</li>)}</ul>
        <form
          className="mt-1 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (note.trim()) addNote.mutate({ id: t.id, text: note }, { onSuccess: () => setNote("") });
          }}
        >
          <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add a marketer note" />
          <Button type="submit" size="sm">Add</Button>
        </form>
      </section>
      <p className="text-muted">config {t.config_hash} · library v{t.library_version} · tag source {t.tag_source ?? "–"} · jaccard {num(t.tag_jaccard)}</p>
    </div>
  );
}

export function TrajectoryActions({ t }: { t: TrajectoryOut }) {
  const review = useReview();
  const editCopy = useEditCopy();
  const cards = useCards();
  const [wrongOpen, setWrongOpen] = useState(false);
  const [copyOpen, setCopyOpen] = useState(false);
  const [picked, setPicked] = useState<string[]>(t.verified_card_ids);
  const [copy, setCopy] = useState(t.copy);
  const shipped = t.renders.some((r) => r.shipped_at);
  const blocked = !t.format_ok || (t.preship ? !(t.preship.policy_ok && t.preship.brand_ok) : false);
  return (
    <div className="flex flex-wrap items-center gap-1">
      <ErrorNote error={review.error ?? editCopy.error} />
      <Button size="sm" variant="primary" disabled={blocked || review.isPending} title={blocked ? "malformed or flagged by pre-ship" : "approve for shipping"} data-testid="action-run" onClick={() => review.mutate({ id: t.id, body: { label: "run" } })}>Run</Button>
      <Button size="sm" disabled={review.isPending} data-testid="action-skip" onClick={() => review.mutate({ id: t.id, body: { label: "skip" } })}>Skip</Button>
      <Button size="sm" data-testid="action-wrong-cards" onClick={() => setWrongOpen(true)}>Wrong cards</Button>
      <Button size="sm" disabled={shipped} title={shipped ? "copy is frozen after shipping" : ""} onClick={() => setCopyOpen(true)}>Edit copy</Button>
      <Modal title="Which cards does this idea actually use?" open={wrongOpen} onClose={() => setWrongOpen(false)}>
        <div className="grid grid-cols-2 gap-1 text-xs">
          {(cards.data ?? []).filter((c) => c.status === "active").map((c) => (
            <label key={c.id} className="flex items-center gap-2 rounded border border-line px-2 py-1">
              <input type="checkbox" checked={picked.includes(c.id)} onChange={(e) => setPicked((p) => (e.target.checked ? [...p, c.id] : p.filter((x) => x !== c.id)))} />
              <span className="font-mono">{c.slug}</span> <span className="text-muted">{c.kind}</span>
            </label>
          ))}
        </div>
        <div className="mt-3 flex justify-end gap-2">
          <Button onClick={() => setWrongOpen(false)}>Cancel</Button>
          <Button variant="primary" data-testid="confirm-wrong-cards" disabled={picked.length === 0} onClick={() => review.mutate({ id: t.id, body: { label: "wrong_cards", corrected_card_ids: picked, note: "corrected on Data screen" } }, { onSuccess: () => setWrongOpen(false) })}>Save correction</Button>
        </div>
      </Modal>
      <Modal title="Edit copy" open={copyOpen} onClose={() => setCopyOpen(false)}>
        <div className="space-y-2">
          <Field label="Primary text"><Textarea rows={4} value={copy.primary_text} onChange={(e) => setCopy({ ...copy, primary_text: e.target.value })} /></Field>
          <Field label="Headline"><Input value={copy.headline} onChange={(e) => setCopy({ ...copy, headline: e.target.value })} /></Field>
          <Field label="Description"><Input value={copy.description} onChange={(e) => setCopy({ ...copy, description: e.target.value })} /></Field>
          <Field label="CTA"><Input value={copy.cta} onChange={(e) => setCopy({ ...copy, cta: e.target.value })} /></Field>
          <div className="flex justify-end gap-2"><Button onClick={() => setCopyOpen(false)}>Cancel</Button><Button variant="primary" onClick={() => editCopy.mutate({ id: t.id, body: copy }, { onSuccess: () => setCopyOpen(false) })}>Save</Button></div>
        </div>
      </Modal>
    </div>
  );
}
