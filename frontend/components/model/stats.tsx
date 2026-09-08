"use client";

import { useAccounts, useEvals, useGoldGap, useKillSwitch, useLaunchEval, useLaunchTraining, useLoopAStats, useMe, useReviewStats, useRewardModelStats, useSetKillSwitch, useTrainRewardModel, useTrainingRuns, useVerifierStats } from "@/lib/api/hooks";
import { useState } from "react";
import { ago, pct } from "@/lib/format";
import { CardChip, StatusBadge } from "../chips";
import { Badge, Button, Empty, Panel } from "../ui";

type Bucket = { generated: number; format_rejected: number; novelty_rejected: number; shipped: number; screened: number; screen_passed: number; measured: number; tier2: number; screen_pass_rate: number | null; tier2_rate: number | null };
type LoopA = { by_backend: Record<string, Bucket>; by_typicality: Record<string, Bucket>; by_combination: { cards: string[]; niche: string; uses: number; measured: number; tier2: number; tier2_rate: number | null }[]; totals: Bucket };

function BucketTable({ title, rows }: { title: string; rows: Record<string, Bucket> }) {
  const keys = Object.keys(rows);
  return (
    <div>
      <h4 className="mb-1 text-xs font-semibold uppercase text-muted">{title}</h4>
      {keys.length === 0 && <Empty>Nothing yet.</Empty>}
      {keys.length > 0 && (
        <table className="w-full text-left text-xs">
          <thead className="text-muted"><tr><th></th><th>generated</th><th>format ✗</th><th>novelty ✗</th><th>shipped</th><th>screen pass</th><th>measured</th><th>tier 2+</th><th>tier 2 rate</th></tr></thead>
          <tbody>
            {keys.map((k) => { const b = rows[k]; return (
              <tr key={k} className="border-t border-line"><td className="font-medium">{k}</td><td>{b.generated}</td><td>{b.format_rejected}</td><td>{b.novelty_rejected}</td><td>{b.shipped}</td><td>{pct(b.screen_pass_rate)} <span className="text-muted">({b.screen_passed}/{b.screened})</span></td><td>{b.measured}</td><td>{b.tier2}</td><td>{b.tier2_rate !== null ? <Badge className={b.tier2_rate > 0 ? "tier-2" : "border border-line"}>{pct(b.tier2_rate)}</Badge> : "–"}</td></tr>
            ); })}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function LoopAStatsPanel() {
  const q = useLoopAStats();
  const d = q.data as LoopA | undefined;
  return (
    <Panel testId="loop-a-stats" title="Loop A stats">
      {!d && <Empty>Loading…</Empty>}
      {d && (
        <div className="space-y-4">
          <BucketTable title="per backend" rows={d.by_backend} />
          <BucketTable title="per typicality label (do rare ideas hit more?)" rows={d.by_typicality} />
          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase text-muted">per combination (top 50 by tier 2 rate)</h4>
            <table className="w-full text-left text-xs">
              <thead className="text-muted"><tr><th>cards</th><th>niche</th><th>uses</th><th>measured</th><th>tier 2+</th><th>rate</th></tr></thead>
              <tbody>{d.by_combination.slice(0, 25).map((c, i) => (<tr key={i} className="border-t border-line"><td><div className="flex flex-wrap gap-1">{c.cards.map((s) => <CardChip key={s} slug={s} />)}</div></td><td className="font-mono text-muted">{c.niche}</td><td>{c.uses}</td><td>{c.measured}</td><td>{c.tier2}</td><td>{pct(c.tier2_rate)}</td></tr>))}</tbody>
            </table>
          </div>
        </div>
      )}
    </Panel>
  );
}

export function ReviewStatsPanel() {
  const q = useReviewStats();
  const rows = (q.data ?? {}) as Record<string, { run: number; skip: number; wrong_cards: number; novelty_rejects: number; format_rejects: number }>;
  return (
    <Panel testId="review-stats" title="Review stats per backend">
      {Object.keys(rows).length === 0 && <Empty>No reviews yet.</Empty>}
      {Object.keys(rows).length > 0 && (
        <table className="w-full text-left text-xs"><thead className="text-muted"><tr><th>backend</th><th>run</th><th>skip</th><th>wrong cards</th><th>novelty rejects</th><th>format rejects</th></tr></thead>
          <tbody>{Object.entries(rows).map(([k, v]) => <tr key={k} className="border-t border-line"><td className="font-medium">{k}</td><td>{v.run}</td><td>{v.skip}</td><td>{v.wrong_cards}</td><td>{v.novelty_rejects}</td><td>{v.format_rejects}</td></tr>)}</tbody></table>
      )}
    </Panel>
  );
}

export function VerifierPanel() {
  const q = useVerifierStats();
  const d = q.data as { tagged: number; agreement_rate: number | null; mean_jaccard: number | null; wrong_cards_labels: number; per_card: Record<string, { written: number; verified: number; both: number; corrected_in: number; corrected_out: number }>; versions: { version: string; kind: string; active: boolean; labels: number }[] } | undefined;
  return (
    <Panel testId="verifier-panel" title="Verifier">
      {d && (
        <div className="space-y-2 text-xs">
          <div className="flex flex-wrap gap-1"><Badge className="border border-line">{d.tagged} tagged</Badge><Badge className="border border-line">agreement {pct(d.agreement_rate)}</Badge><Badge className="border border-line">mean jaccard {d.mean_jaccard?.toFixed(2) ?? "–"}</Badge><Badge className="border border-line">{d.wrong_cards_labels} expert corrections</Badge>{d.versions.map((v) => <Badge key={v.version} className={v.active ? "bg-accent-soft text-accent" : "border border-line"}>{v.version} ({v.labels} labels)</Badge>)}</div>
          <table className="w-full text-left"><thead className="text-muted"><tr><th>card</th><th>written</th><th>verified</th><th>both</th><th>expert added</th><th>expert removed</th></tr></thead>
            <tbody>{Object.entries(d.per_card).sort((a, b) => b[1].written - a[1].written).slice(0, 30).map(([slug, v]) => <tr key={slug} className="border-t border-line"><td><CardChip slug={slug} /></td><td>{v.written}</td><td>{v.verified}</td><td>{v.both}</td><td>{v.corrected_in}</td><td>{v.corrected_out}</td></tr>)}</tbody></table>
          <p className="text-muted">Classifier v2 trains automatically once expert labels exceed the configured threshold.</p>
        </div>
      )}
    </Panel>
  );
}

export function RewardModelPanel() {
  const q = useRewardModelStats();
  const me = useMe();
  const gap = useGoldGap();
  const train = useTrainRewardModel();
  const canTrain = me.data?.role === "researcher";
  const g = gap.data;
  const d = q.data as { versions: { version: string; kind: string; active: boolean; metrics: Record<string, unknown>; created_at: string }[]; by_version: Record<string, { scored: number; measured: number; tier2: number; mean_score_tier2: number | null; mean_score_other: number | null }>; tier2_positives: number; cold_until_positives: number; lambda_pess: number; loop_b_ready: boolean } | undefined;
  return (
    <Panel testId="rm-panel" title="Reward model" actions={canTrain ? <Button size="sm" data-testid="train-rm" disabled={train.isPending} onClick={() => train.mutate({ kind: null, activate: true, seed: 0 })}>{train.isPending ? "Training…" : "Train reward model"}</Button> : undefined}>
      {train.data && <p className="mb-2 text-xs" data-testid="train-rm-result">{train.data.trained ? `trained ${train.data.version} on ${train.data.n_rows} rows (${train.data.n_positive} positives)` : train.data.reason}</p>}
      {train.error && <p className="mb-2 text-xs text-danger">{String(train.error)}</p>}
      {d && (
        <div className="space-y-2 text-xs">
          <div className="flex flex-wrap gap-1"><Badge className="border border-line">tier 2+ positives {d.tier2_positives} / {d.cold_until_positives} for rm_outcome</Badge><Badge className="border border-line">λ pess {d.lambda_pess}</Badge><Badge className={d.loop_b_ready ? "tier-2" : "border border-line"}>{d.loop_b_ready ? "Loop B ready" : "Loop B waits for data"}</Badge>{g && <Badge data-testid="gold-gap" className={g.tripped ? "bg-danger/15 text-danger" : "border border-line"}>gold gap {g.gap !== null ? g.gap.toFixed(3) : "n/a"} / {g.threshold} · {g.tripped ? "tripped: Loop B paused" : g.detail}</Badge>}</div>
          <table className="w-full text-left"><thead className="text-muted"><tr><th>version</th><th>scored</th><th>measured</th><th>tier 2+</th><th>mean score · tier 2+</th><th>mean score · rest</th><th>separation</th></tr></thead>
            <tbody>{Object.entries(d.by_version).map(([v, b]) => <tr key={v} className="border-t border-line"><td className="font-mono">{v}</td><td>{b.scored}</td><td>{b.measured}</td><td>{b.tier2}</td><td>{b.mean_score_tier2?.toFixed(2) ?? "–"}</td><td>{b.mean_score_other?.toFixed(2) ?? "–"}</td><td>{b.mean_score_tier2 !== null && b.mean_score_other !== null ? (b.mean_score_tier2 - b.mean_score_other).toFixed(2) : "–"}</td></tr>)}</tbody></table>
          <p className="text-muted">Ensemble of {3} heads on frozen embeddings; rm_score = mean − λ·std. Separation: a useful model scores eventual tier 2+ ideas higher than the rest. The gold gap compares proxy score with realized tier 2 rate across windows; a trip pauses Loop B, never Loop A.</p>
        </div>
      )}
    </Panel>
  );
}

export function RunsPanel() {
  const me = useMe();
  const runs = useTrainingRuns();
  const evals = useEvals();
  const rm = useRewardModelStats();
  const launch = useLaunchTraining();
  const launchEval = useLaunchEval();
  const [stage, setStage] = useState<"rft" | "dpo" | "grpo_offpolicy" | "grpo_onpolicy">("rft");
  const [mode, setMode] = useState<"dry" | "smoke" | "full">("dry");
  const [nBriefs, setNBriefs] = useState(2);
  const isResearcher = me.data?.role === "researcher";
  const ready = (rm.data as { loop_b_ready?: boolean } | undefined)?.loop_b_ready ?? false;
  const canLaunch = isResearcher && (ready || mode !== "full") && !launch.isPending;
  const trainingRuns = runs.data ?? [];
  const evalRuns = evals.data ?? [];
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel
        testId="loop-b-runs"
        title="Loop B runs"
        actions={
          isResearcher ? (
            <div className="flex items-center gap-1">
              <select data-testid="train-stage" className="rounded border border-line bg-panel px-1 py-0.5 text-xs" value={stage} onChange={(e) => setStage(e.target.value as typeof stage)}>
                <option value="rft">rft</option><option value="dpo">dpo</option><option value="grpo_offpolicy">grpo_offpolicy</option><option value="grpo_onpolicy">grpo_onpolicy (simulator)</option>
              </select>
              <select data-testid="train-mode" className="rounded border border-line bg-panel px-1 py-0.5 text-xs" value={mode} onChange={(e) => setMode(e.target.value as typeof mode)} title="dry: load the snapshot and report counts; smoke: a few steps; full: the real run (needs Loop B ready)">
                <option value="dry">dry</option><option value="smoke">smoke</option><option value="full">full</option>
              </select>
              <Button size="sm" variant="primary" data-testid="launch-training" disabled={!canLaunch} title={ready ? "snapshot the archive and queue a run pinning current versions" : "full runs need 50 tier 2+ trajectories with verified cards; dry and smoke runs are allowed"} onClick={() => launch.mutate({ stage, base_model: "Qwen/Qwen3-8B", adapter_from: null, smoke: mode === "smoke", dry: mode === "dry", simulator: stage === "grpo_onpolicy", allow_rm_reward: false })}>{launch.isPending ? "Queuing…" : "Launch run"}</Button>
            </div>
          ) : undefined
        }
      >
        {launch.error && <p className="mb-2 text-xs text-danger" data-testid="launch-training-error">{String(launch.error)}</p>}
        {trainingRuns.length === 0 && <Empty>No runs yet. Loop B starts when the archive holds enough real outliers; dry and smoke runs exercise the pipeline before that.</Empty>}
        {trainingRuns.length > 0 && (
          <table className="w-full text-left text-xs" data-testid="training-runs-table"><thead className="text-muted"><tr><th>run</th><th>stage</th><th>status</th><th>snapshot</th><th>rows · tier 2+</th><th>stop reason</th><th>when</th></tr></thead>
            <tbody>{trainingRuns.map((r) => { const snap = (r.metrics as { snapshot?: { n_trajectories?: number; n_tier2?: number } }).snapshot; return (
              <tr key={r.id} className="border-t border-line"><td className="font-mono">{r.id.slice(0, 8)}</td><td>{r.stage}</td><td><StatusBadge status={r.status} /></td><td className="font-mono">{r.snapshot_hash?.slice(0, 8) ?? "–"}</td><td>{snap ? `${snap.n_trajectories ?? "?"} · ${snap.n_tier2 ?? "?"}` : "–"}</td><td className="truncate text-muted">{r.stop_reason ?? "–"}</td><td className="text-muted">{ago(r.created_at)}</td></tr>
            ); })}</tbody></table>
        )}
      </Panel>
      <Panel
        testId="eval-runs"
        title="Online eval"
        actions={
          isResearcher ? (
            <div className="flex items-center gap-1">
              <input data-testid="eval-briefs" type="number" min={1} max={200} className="w-14 rounded border border-line bg-panel px-1 py-0.5 text-xs" value={nBriefs} onChange={(e) => setNBriefs(Number(e.target.value))} title="held-out briefs" />
              <Button size="sm" data-testid="launch-eval" disabled={launchEval.isPending} title="fake arms run to completion on the simulator; live arms start episodes and fill in with the daily cadence" onClick={() => launchEval.mutate({ kind: "online", systems: ["loop_a_fake", "random_fake"], n_briefs: nBriefs, budget_cap: 1200, seed: 0, max_days: 60, inline: true })}>{launchEval.isPending ? "Running…" : "Run eval (fake arms)"}</Button>
            </div>
          ) : undefined
        }
      >
        {launchEval.error && <p className="mb-2 text-xs text-danger">{String(launchEval.error)}</p>}
        {evalRuns.length === 0 && <Empty>No evals yet. Systems compared blind on held-out briefs: loop_a_api, loop_a_local, loop_b, mean-objective RL; tier 2 rate with bootstrap intervals and loop_b_vs_loop_a. Success needs non-overlapping intervals on 20+ briefs.</Empty>}
        {evalRuns.length > 0 && (
          <div className="space-y-2">
            {evalRuns.map((e) => (
              <div key={e.id} className="rounded border border-line p-2 text-xs" data-testid="eval-run">
                <div className="flex flex-wrap items-center gap-2"><span className="font-mono">{e.id.slice(0, 8)}</span><span>{e.kind}</span><StatusBadge status={e.status} /><span className="text-muted">{e.holdout_brief_ids.length} briefs · {ago(e.created_at)}</span>{e.summary.success === true && <Badge className="tier-2">success</Badge>}{e.summary.success === false && <Badge className="border border-line">no separation</Badge>}</div>
                {e.arms.length > 0 && (
                  <table className="mt-1 w-full text-left"><thead className="text-muted"><tr><th>arm</th><th>n</th><th>tier 2 rate</th><th>95% CI</th></tr></thead>
                    <tbody>{e.arms.map((a) => <tr key={a.id} className="border-t border-line"><td>{a.blind_label} <span className="text-muted">{e.status === "completed" ? `(${a.system})` : ""}</span></td><td>{a.n_briefs}</td><td>{pct(a.tier2_rate)}</td><td>{a.ci_low !== null && a.ci_high !== null ? `${pct(a.ci_low)} – ${pct(a.ci_high)}` : "–"}</td></tr>)}</tbody></table>
                )}
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

export function AccountsPanel() {
  const accounts = useAccounts();
  const ks = useKillSwitch();
  const setKs = useSetKillSwitch();
  const me = useMe();
  const canToggle = me.data?.role === "operator" || me.data?.role === "researcher";
  return (
    <Panel testId="accounts-panel" title="Evaluator accounts" actions={ks.data && <Button size="sm" variant={ks.data.shipping_enabled ? "danger" : "primary"} disabled={!canToggle || setKs.isPending} data-testid="kill-switch" onClick={() => setKs.mutate({ shipping_enabled: !ks.data!.shipping_enabled, reason: ks.data!.shipping_enabled ? "manual stop from Model screen" : "re-enabled from Model screen" })}>{ks.data.shipping_enabled ? "Disable all shipping" : "Enable shipping"}</Button>}>
      {ks.data && <p className={`mb-2 text-xs ${ks.data.shipping_enabled ? "text-muted" : "text-red-700"}`}>Kill switch: shipping {ks.data.shipping_enabled ? "enabled" : "DISABLED"} · {ks.data.changed_by} · {ago(ks.data.changed_at)}{ks.data.reason ? ` · ${ks.data.reason}` : ""}</p>}
      <table className="w-full text-left text-xs"><thead className="text-muted"><tr><th>account</th><th>meta id</th><th>status</th><th>attribution</th><th>api</th><th>daily cap</th><th>tokens</th><th>last sync</th></tr></thead>
        <tbody>{(accounts.data ?? []).map((a) => <tr key={a.id} className="border-t border-line"><td className="font-medium">{a.name}{a.is_fake && <Badge className="ml-1 border border-line">fake</Badge>}</td><td className="font-mono">{a.meta_account_id}</td><td><StatusBadge status={a.status} />{a.status === "needs_reauth" && <span className="ml-1 text-red-700">Run is blocked for this account</span>}</td><td>{a.attribution_setting}</td><td>{a.api_version}</td><td>${a.daily_cap_usd ?? "–"}</td><td>{a.has_access_token ? "ads ✓" : "ads ✗"} {a.has_page_token ? "page ✓" : "page ✗"}</td><td className="text-muted">{a.last_sync_at ? ago(a.last_sync_at) : "never"}{a.last_error ? ` · ${a.last_error.slice(0, 60)}` : ""}</td></tr>)}</tbody></table>
    </Panel>
  );
}
