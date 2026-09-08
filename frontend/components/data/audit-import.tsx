"use client";

import { useState } from "react";
import { uploadCsv, type Schemas } from "@/lib/api/client";
import { useAudit } from "@/lib/api/hooks";
import { ago } from "@/lib/format";
import { Button, Empty, ErrorNote, Panel } from "../ui";

export function AuditTab() {
  const audit = useAudit();
  return (
    <Panel testId="audit-tab" title={<>Audit <span className="text-muted">last {audit.data?.length ?? 0}</span></>}>
      {audit.data?.length === 0 && <Empty>No audit entries yet.</Empty>}
      <table className="w-full text-left text-xs">
        <thead className="text-muted"><tr><th className="py-1">when</th><th>actor</th><th>action</th><th>object</th><th>change</th></tr></thead>
        <tbody>
          {(audit.data ?? []).map((a) => (
            <tr key={a.id} className="border-t border-line align-top">
              <td className="py-1 text-muted">{ago(a.at)}</td>
              <td>{a.actor_id}</td>
              <td className="font-mono">{a.action}</td>
              <td className="font-mono text-muted">{a.object_type} {a.object_id.slice(0, 8)}</td>
              <td className="max-w-md truncate text-muted" title={JSON.stringify({ before: a.before, after: a.after })}>{JSON.stringify(a.after ?? a.before)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

const IMPORTS = [
  { id: "cards", path: "/api/import/cards", columns: "slug,name,kind,definition,qualifying_condition,source,status" },
  { id: "trajectories", path: "/api/import/trajectories", columns: "brief_id,angle,primary_text,headline,description,cta,visual_brief,cards (slug|slug),author_id,created_at,campaign_id" },
  { id: "outcomes", path: "/api/import/outcomes", columns: "trajectory_id or render_id,day,phase,impressions,link_clicks,spend,purchases,revenue" },
  { id: "comments", path: "/api/import/comments", columns: "render_id,external_id,commenter_id,text,created_time,like_count" },
];

export function ImportTab() {
  return (
    <Panel testId="import-tab" title="CSV import">
      <p className="mb-3 text-xs text-muted">Practitioner history arrives here. Trajectories without card tags get verifier-proposed tags and stay out of training until an expert confirms them. Comments are PII-stripped before storage.</p>
      <div className="grid gap-3 md:grid-cols-2">{IMPORTS.map((i) => <ImportForm key={i.id} {...i} />)}</div>
    </Panel>
  );
}

function ImportForm({ id: k, path, columns }: (typeof IMPORTS)[number]) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Schemas["ImportResult"] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="rounded border border-line p-3 text-xs">
      <div className="font-semibold capitalize">{k}</div>
      <div className="mb-2 font-mono text-[10px] text-muted">{columns}</div>
      <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div className="mt-2 flex items-center gap-2">
        <Button size="sm" variant="primary" disabled={!file || busy} onClick={async () => { if (!file) return; setBusy(true); setError(null); try { setResult(await uploadCsv(path, file)); } catch (e) { setError(e); } finally { setBusy(false); } }}>Upload</Button>
        {result && <span>created {result.created} · updated {result.updated} · skipped {result.skipped}{result.errors.length ? ` · ${result.errors.length} errors` : ""}</span>}
      </div>
      <ErrorNote error={error} />
      {result?.errors.length ? <ul className="mt-1 list-disc pl-4 text-red-700">{result.errors.slice(0, 10).map((e, i) => <li key={i}>{e}</li>)}</ul> : null}
    </div>
  );
}
