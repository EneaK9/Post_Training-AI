"use client";

import { X } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "@/lib/api/client";
import { useBriefEpisodes, useCardVersions, useCards } from "@/lib/api/hooks";
import { ago, num, shortId } from "@/lib/format";
import { StatusBadge, TierBadge } from "./chips";
import { TrajectoryCard } from "./data/trajectory-card";
import { Badge, Empty } from "./ui";

/** Opens any object by id from any citation: ?open=<type>:<id>. */
export function ObjectDrawer() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const open = params.get("open");
  if (!open) return null;
  const idx = open.indexOf(":");
  const type = open.slice(0, idx);
  const id = open.slice(idx + 1);
  const close = () => {
    const next = new URLSearchParams(params.toString());
    next.delete("open");
    router.replace(`${pathname}${next.size ? `?${next}` : ""}`, { scroll: false });
  };
  return (
    <div className="fixed inset-y-0 right-0 z-40 w-[560px] max-w-full overflow-auto border-l border-line bg-panel shadow-2xl" data-testid="drawer">
      <header className="sticky top-0 flex items-center justify-between border-b border-line bg-panel px-4 py-2">
        <div className="text-xs uppercase text-muted">{type}</div>
        <button onClick={close} aria-label="close drawer" className="rounded p-1 hover:bg-accent-soft"><X size={16} /></button>
      </header>
      <div className="p-4">
        {type === "card" && <CardDetail id={id} />}
        {type === "card-slug" && <CardBySlug slug={id} />}
        {type === "brief" && <BriefDetail id={id} />}
        {type === "trajectory" && <TrajectoryCard id={id} expanded />}
        {type === "signal" && <SignalDetail id={id} />}
        {type === "episode" && <EpisodeDetail id={id} />}
        {type === "ref" && <RefResolver ref_={id} />}
      </div>
    </div>
  );
}

function CardBySlug({ slug }: { slug: string }) {
  const cards = useCards();
  const card = cards.data?.find((c) => c.slug === slug);
  if (cards.isLoading) return <Empty>Loading…</Empty>;
  if (!card) return <Empty>No card with slug {slug}</Empty>;
  return <CardDetail id={card.id} />;
}

export function CardDetail({ id }: { id: string }) {
  const card = useQuery({ queryKey: ["card", id], queryFn: () => unwrap(api.GET("/api/cards/{card_id}", { params: { path: { card_id: id } } })) });
  const versions = useCardVersions(id);
  if (!card.data) return <Empty>Loading…</Empty>;
  const c = card.data;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold">{c.name}</h3>
          <div className="font-mono text-xs text-muted">{c.slug} · {c.kind} · v{c.version}</div>
        </div>
        <StatusBadge status={c.status} />
      </div>
      <p>{c.definition}</p>
      <p className="text-muted"><span className="font-medium text-foreground">Applies when:</span> {c.qualifying_condition || "no stated condition"}</p>
      {c.source && <p className="text-xs text-muted">Source: {c.source}</p>}
      <p className="text-xs text-muted">Contributed by {c.contributed_by}</p>
      <div className="grid grid-cols-4 gap-2 text-center">
        <Stat label="uses" value={String(c.stats.uses)} />
        <Stat label="measured" value={String(c.stats.measured)} />
        <Stat label="tier 2+" value={`${c.stats.tier2_count} (${c.stats.tier2_rate === null || c.stats.tier2_rate === undefined ? "–" : (c.stats.tier2_rate * 100).toFixed(0) + "%"})`} />
        <Stat label="verifier agrees" value={c.stats.verifier_agreement === null || c.stats.verifier_agreement === undefined ? "–" : (c.stats.verifier_agreement * 100).toFixed(0) + "%"} />
      </div>
      <div>
        <h4 className="mb-1 text-xs font-semibold uppercase text-muted">Versions</h4>
        {versions.data?.length ? (
          <ol className="space-y-2">
            {versions.data.map((v, i) => {
              const prev = versions.data![i - 1];
              const changed = prev && (prev.definition !== v.definition || prev.qualifying_condition !== v.qualifying_condition || prev.status !== v.status || prev.name !== v.name);
              return (
                <li key={v.id} className="rounded border border-line p-2 text-xs">
                  <div className="flex justify-between text-muted"><span>v{v.version} · {v.edited_by}</span><span>{ago(v.created_at)} · {v.status}</span></div>
                  {i === 0 || changed ? (
                    <div className="mt-1 grid grid-cols-2 gap-2">
                      {prev && <pre className="whitespace-pre-wrap rounded bg-red-50 p-1 line-through dark:bg-red-950/30">{prev.definition}</pre>}
                      <pre className="whitespace-pre-wrap rounded bg-green-50 p-1 dark:bg-green-950/30">{v.definition}</pre>
                    </div>
                  ) : (
                    <div className="mt-1 text-muted">no text change</div>
                  )}
                </li>
              );
            })}
          </ol>
        ) : (
          <Empty>No versions</Empty>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-line p-2">
      <div className="text-[10px] uppercase text-muted">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}

function BriefDetail({ id }: { id: string }) {
  const brief = useQuery({ queryKey: ["brief", id], queryFn: () => unwrap(api.GET("/api/briefs/{brief_id}", { params: { path: { brief_id: id } } })) });
  const episodes = useBriefEpisodes(id);
  if (!brief.data) return <Empty>Loading…</Empty>;
  const b = brief.data;
  return (
    <div className="space-y-3 text-sm">
      <h3 className="text-base font-semibold">{b.company}: {b.product}</h3>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div><span className="text-muted">Offer</span><br />{b.offer}</div>
        <div><span className="text-muted">Audience</span><br />{b.audience}</div>
        <div><span className="text-muted">Category</span><br />{b.category}</div>
        <div><span className="text-muted">Goal</span><br />{b.goal_metric} · {b.channel}</div>
      </div>
      <p><span className="text-muted">World state{b.world_state_at ? ` (${ago(b.world_state_at)})` : ""}:</span> {b.world_state || "nothing notable"}</p>
      <div className="flex flex-wrap gap-1">{b.constraints.map((c) => <Badge key={c} className="border border-line">{c}</Badge>)}</div>
      <h4 className="text-xs font-semibold uppercase text-muted">Episodes ({b.episode_count})</h4>
      {episodes.data?.length ? (
        <ul className="space-y-1">
          {episodes.data.map((e) => (
            <li key={e.id} className="flex items-center justify-between rounded border border-line px-2 py-1 text-xs">
              <Link href={`?open=episode:${e.id}`} scroll={false} className="font-mono hover:underline">{shortId(e.id)}</Link>
              <span>${num(e.spent, 0)} / ${num(e.budget_cap, 0)}</span>
              <TierBadge tier={e.best_tier} />
              <StatusBadge status={e.status} />
            </li>
          ))}
        </ul>
      ) : (
        <Empty>No episodes yet</Empty>
      )}
      <Link href={`/data?brief=${b.id}`} className="text-xs text-accent hover:underline">Filter history by this brief →</Link>
    </div>
  );
}

function SignalDetail({ id }: { id: string }) {
  const s = useQuery({ queryKey: ["signal", id], queryFn: () => unwrap(api.GET("/api/signals/{signal_id}", { params: { path: { signal_id: id } } })) });
  if (!s.data) return <Empty>Loading…</Empty>;
  const sig = s.data;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center gap-2"><Badge className="border border-line">{sig.kind}</Badge><Badge className="border border-line">{sig.sentiment}</Badge><Badge className="border border-line">×{sig.count}</Badge><StatusBadge status={sig.status} /></div>
      <p className="text-base">{sig.text}</p>
      <h4 className="text-xs font-semibold uppercase text-muted">Evidence</h4>
      <ul className="space-y-1">{sig.evidence.map((e, i) => <li key={i} className="rounded border border-line px-2 py-1 text-xs"><span className="text-muted">{String(e.source)}</span> “{String(e.excerpt)}”</li>)}</ul>
      <p className="text-xs text-muted">Extracted by {sig.extracted_by}{sig.decided_by ? `, decided by ${sig.decided_by}` : ""} · {ago(sig.created_at)}</p>
      <Link href={`?open=trajectory:${sig.trajectory_id}`} scroll={false} className="text-xs text-accent hover:underline">Open trajectory →</Link>
    </div>
  );
}

function EpisodeDetail({ id }: { id: string }) {
  const ep = useQuery({ queryKey: ["episode", id], queryFn: () => unwrap(api.GET("/api/episodes/{episode_id}", { params: { path: { episode_id: id }, query: { include_trace: true } } })) });
  if (!ep.data) return <Empty>Loading…</Empty>;
  const e = ep.data;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center justify-between"><h3 className="font-mono text-base font-semibold">episode {shortId(e.id)}</h3><StatusBadge status={e.status} /></div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <Stat label="spent / cap" value={`$${num(e.spent, 0)} / $${num(e.budget_cap, 0)}`} />
        <Stat label="live ads" value={String(e.live_ads)} />
        <Stat label="best tier" value={e.best_tier === null || e.best_tier === undefined ? "–" : String(e.best_tier)} />
      </div>
      <p className="text-xs text-muted">backend {e.backend} · config {e.config_hash} · by {e.created_by} · {ago(e.created_at)}{e.stop_reason ? ` · ${e.stop_reason}` : ""}</p>
      <h4 className="text-xs font-semibold uppercase text-muted">Batches</h4>
      {e.batches.length === 0 && <Empty>No batches yet</Empty>}
      {e.batches.map((b) => (
        <div key={b.id} className="rounded border border-line p-2 text-xs">
          <div className="flex justify-between"><span>batch {b.index} · {b.state}</span><span className="text-muted">{b.trajectory_ids.length} ideas · {ago(b.created_at)}</span></div>
          <div className="mt-1 flex flex-wrap gap-1">{b.trajectory_ids.map((t) => <Link key={t} href={`?open=trajectory:${t}`} scroll={false} className="font-mono hover:underline">{shortId(t)}</Link>)}</div>
          {b.prompt_trace && <details className="mt-1"><summary className="cursor-pointer text-muted">what the model saw</summary><pre className="mt-1 max-h-60 overflow-auto rounded bg-background p-2 text-[10px]">{JSON.stringify(b.prompt_trace, null, 2)}</pre></details>}
        </div>
      ))}
    </div>
  );
}

/** Citations carry 8-hex refs; the drawer resolves them against the objects the trajectory cites. */
function RefResolver({ ref_ }: { ref_: string }) {
  const [ref, ...ids] = ref_.split(",");
  const match = ids.find((i) => i.replace(/-/g, "").startsWith(ref));
  if (!match) return <Empty>Could not resolve reference {ref}</Empty>;
  return <RefTry id={match} />;
}

function RefTry({ id }: { id: string }) {
  const t = useQuery({ queryKey: ["trajectory-try", id], queryFn: () => api.GET("/api/trajectories/{trajectory_id}", { params: { path: { trajectory_id: id } } }), retry: false });
  if (t.isLoading) return <Empty>Loading…</Empty>;
  if (t.data?.data) return <TrajectoryCard id={id} expanded />;
  return <SignalDetail id={id} />;
}
