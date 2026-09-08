"use client";

import { useState } from "react";
import type { Schemas } from "@/lib/api/client";
import { useAccounts, useBriefEpisodes, useBriefs, useConfig, useCreateBrief, useCreateEpisode } from "@/lib/api/hooks";
import { shortId } from "@/lib/format";
import { Button, ErrorNote, Field, Input, Panel, Select, Textarea } from "../ui";

export type GenerateSpec = { brief_id: string; episode_id: string | null; backend: Schemas["GenerateIn"]["backend"]; compare_backend: Schemas["GenerateIn"]["backend"] | null; k: number; renders_per_idea: number; no_llm: boolean };

export function GenerateForm({ onGenerate, busy }: { onGenerate: (spec: GenerateSpec) => void; busy: boolean }) {
  const briefs = useBriefs();
  const accounts = useAccounts();
  const config = useConfig();
  const createEpisode = useCreateEpisode();
  const createBrief = useCreateBrief();
  const [briefId, setBriefId] = useState<string>("");
  const [backend, setBackend] = useState<Schemas["GenerateIn"]["backend"]>("fake");
  const [compare, setCompare] = useState<Schemas["GenerateIn"]["backend"] | "">("");
  const [k, setK] = useState(8);
  const [renders, setRenders] = useState(3);
  const [noLlm, setNoLlm] = useState(true);
  const [episodeMode, setEpisodeMode] = useState<"none" | "new" | string>("none");
  const [cap, setCap] = useState(5000);
  const [newBrief, setNewBrief] = useState(false);
  const [nb, setNb] = useState<Schemas["BriefCreate"]>({ company: "", product: "", offer: "", audience: "", category: "skincare", world_state: "", constraints: [], brand_assets: [], raw_text: "", meta: {}, goal_metric: "purchases", channel: "meta_feed_image" });
  const episodes = useBriefEpisodes(briefId || null);
  const brief = briefs.data?.find((b) => b.id === briefId);
  const account = accounts.data?.find((a) => a.id === brief?.ad_account_id);
  const imageMode = (config.data?.config as { images?: { mode?: string } } | undefined)?.images?.mode ?? "brand_assets";
  const sizes = ((config.data?.config as { images?: { sizes?: number[][] } } | undefined)?.images?.sizes ?? [[1080, 1080]]).map((s) => s.join("×")).join(", ");

  const submit = async () => {
    if (!briefId) return;
    let episodeId: string | null = null;
    if (episodeMode === "new") {
      const ep = await createEpisode.mutateAsync({ brief_id: briefId, backend, budget_cap: cap, ad_account_id: brief?.ad_account_id ?? null, keep_running_after_outlier: false });
      episodeId = ep.id;
    } else if (episodeMode !== "none") {
      episodeId = episodeMode;
    }
    onGenerate({ brief_id: briefId, episode_id: episodeId, backend, compare_backend: compare || null, k, renders_per_idea: renders, no_llm: noLlm });
  };

  return (
    <Panel testId="generate-form" title="Generate" className="h-fit">
      <div className="space-y-2 text-xs">
        <Field label="Brief">
          <Select data-testid="brief-select" value={briefId} onChange={(e) => { setBriefId(e.target.value); setEpisodeMode("none"); }}>
            <option value="">choose a brief…</option>
            {(briefs.data ?? []).map((b) => <option key={b.id} value={b.id}>{b.company}: {b.product.slice(0, 40)}</option>)}
          </Select>
        </Field>
        <button className="text-accent hover:underline" onClick={() => setNewBrief((v) => !v)}>{newBrief ? "cancel new brief" : "+ new brief"}</button>
        {newBrief && (
          <div className="space-y-1 rounded border border-line p-2">
            <Input placeholder="Company" value={nb.company} onChange={(e) => setNb({ ...nb, company: e.target.value })} />
            <Input placeholder="Product" value={nb.product} onChange={(e) => setNb({ ...nb, product: e.target.value })} />
            <Input placeholder="Offer" value={nb.offer} onChange={(e) => setNb({ ...nb, offer: e.target.value })} />
            <Input placeholder="Audience" value={nb.audience} onChange={(e) => setNb({ ...nb, audience: e.target.value })} />
            <Input placeholder="Category (skincare, coffee, …)" value={nb.category} onChange={(e) => setNb({ ...nb, category: e.target.value })} />
            <Textarea rows={2} placeholder="World state right now" value={nb.world_state} onChange={(e) => setNb({ ...nb, world_state: e.target.value })} />
            <Input placeholder="Constraints, separated by ;" onChange={(e) => setNb({ ...nb, constraints: e.target.value.split(";").map((s) => s.trim()).filter(Boolean) })} />
            <Select value={nb.channel} onChange={(e) => setNb({ ...nb, channel: e.target.value })}><option value="meta_feed_image">meta_feed_image</option><option value="meta_video">meta_video (rejected in v1)</option></Select>
            <ErrorNote error={createBrief.error} />
            <Button size="sm" variant="primary" disabled={!nb.company || !nb.product || createBrief.isPending} onClick={() => createBrief.mutate({ ...nb, ad_account_id: accounts.data?.[0]?.id ?? null }, { onSuccess: (b) => { setBriefId(b.id); setNewBrief(false); } })}>Create brief</Button>
          </div>
        )}
        {brief && (
          <p className="rounded bg-background p-2 text-muted">
            <span className="text-foreground">{brief.company}</span> · {brief.category} · world: {brief.world_state || "nothing notable"}
            {account && <span className={account.status === "active" ? "" : " text-red-700"}> · account {account.name} ({account.status}{account.is_fake ? ", fake" : ""})</span>}
          </p>
        )}
        <div className="grid grid-cols-2 gap-2">
          <Field label="Backend">
            <Select data-testid="backend-select" value={backend} onChange={(e) => setBackend(e.target.value as Schemas["GenerateIn"]["backend"])}><option value="fake">fake</option><option value="anthropic">anthropic (Claude)</option><option value="local">local (vLLM)</option></Select>
          </Field>
          <Field label="Compare with">
            <Select value={compare} onChange={(e) => setCompare(e.target.value as Schemas["GenerateIn"]["backend"] | "")}><option value="">none</option><option value="fake">fake</option><option value="anthropic">anthropic</option><option value="local">local</option></Select>
          </Field>
          <Field label="K ideas"><Input data-testid="k-input" type="number" min={1} max={32} value={k} onChange={(e) => setK(Number(e.target.value))} /></Field>
          <Field label="Renders per idea"><Input type="number" min={1} max={6} value={renders} onChange={(e) => setRenders(Number(e.target.value))} /></Field>
        </div>
        <p className="text-muted">Image mode <span className="font-mono">{imageMode}</span> · sizes {sizes} (Model screen → images)</p>
        <label className="flex items-center gap-2"><input type="checkbox" checked={noLlm} onChange={(e) => setNoLlm(e.target.checked)} /> heuristic verifier and reward model (no API key needed)</label>
        <Field label="Episode">
          <Select data-testid="episode-select" value={episodeMode} onChange={(e) => setEpisodeMode(e.target.value)} disabled={!briefId}>
            <option value="none">no episode (queue only)</option>
            <option value="new">new episode</option>
            {(episodes.data ?? []).filter((e) => e.status === "searching").map((e) => <option key={e.id} value={e.id}>attach {shortId(e.id)} (${e.spent.toFixed(0)} / ${e.budget_cap.toFixed(0)})</option>)}
          </Select>
        </Field>
        {episodeMode === "new" && <Field label="Budget cap (USD)"><Input data-testid="cap-input" type="number" min={100} step={100} value={cap} onChange={(e) => setCap(Number(e.target.value))} /></Field>}
        <ErrorNote error={createEpisode.error} />
        <Button variant="primary" className="w-full justify-center" data-testid="generate-button" disabled={!briefId || busy || createEpisode.isPending} onClick={submit}>{busy ? "Generating…" : "Generate"}</Button>
      </div>
    </Panel>
  );
}
