"use client";

import { useEffect, useRef, useState } from "react";
import type { Schemas } from "@/lib/api/client";
import { useApplyConfig, useConfig, useConfigHistory, useValidateConfig } from "@/lib/api/hooks";
import { ago } from "@/lib/format";
import { Badge, Button, ErrorNote, Field, Input, Panel, Textarea } from "../ui";

export function ConfigEditor({ focusSection }: { focusSection: string | null }) {
  const config = useConfig();
  const history = useConfigHistory();
  const validate = useValidateConfig();
  const apply = useApplyConfig();
  // null = show the applied config; a string = the researcher's edits
  const [edited, setEdited] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [validation, setValidation] = useState<Schemas["ConfigValidateOut"] | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const yaml = edited ?? config.data?.yaml ?? "";
  const setYaml = (v: string | null) => setEdited(v);
  useEffect(() => {
    if (!focusSection || !area.current) return;
    const idx = yaml.indexOf(`\n${focusSection}:`);
    if (idx >= 0) {
      area.current.focus();
      area.current.setSelectionRange(idx + 1, idx + 1 + focusSection.length + 1);
      const line = yaml.slice(0, idx).split("\n").length;
      area.current.scrollTop = Math.max(0, (line - 2) * 18);
    }
  }, [focusSection, yaml]);
  const c = config.data;
  return (
    <Panel
      testId="config-editor"
      title="Config"
      actions={c && <><Badge className="border border-line" data-testid="config-hash">applied {c.hash}</Badge>{c.file_hash !== c.hash && <Badge className="bg-amber-100 text-amber-800" title="config/config.yaml differs from the applied version">file {c.file_hash}</Badge>}<span className="text-xs text-muted">{c.applied_by ? `${c.applied_by} · ${ago(c.applied_at)}` : "file default"}</span></>}
    >
      <div className="grid gap-3 lg:grid-cols-[1fr_360px]">
        <div className="space-y-2">
          <Textarea ref={area} data-testid="config-yaml" rows={28} className="font-mono text-[11px]" value={yaml} onChange={(e) => { setYaml(e.target.value); setValidation(null); }} spellCheck={false} />
          <div className="flex flex-wrap items-center gap-2">
            <Button data-testid="config-validate" disabled={validate.isPending} onClick={() => validate.mutate({ yaml }, { onSuccess: setValidation })}>Validate</Button>
            <Field label=""><Input placeholder="note for the history" value={note} onChange={(e) => setNote(e.target.value)} /></Field>
            <Button variant="primary" data-testid="config-apply" disabled={!validation?.ok || apply.isPending || (validation && Object.keys(validation.diff).length === 0) || false} onClick={() => apply.mutate({ yaml, note }, { onSuccess: () => { setValidation(null); setYaml(null); } })}>Apply</Button>
            {c && <Button variant="ghost" onClick={() => { setYaml(null); setValidation(null); }}>Reset</Button>}
          </div>
          <ErrorNote error={validate.error ?? apply.error} />
          {validation && (
            <div data-testid="config-diff" className="rounded border border-line p-2 text-xs">
              {!validation.ok && <ul className="list-disc pl-4 text-red-700">{validation.errors.map((e) => <li key={e}>{e}</li>)}</ul>}
              {validation.ok && Object.keys(validation.diff).length === 0 && <p className="text-muted">No changes against the applied config.</p>}
              {validation.ok && Object.keys(validation.diff).length > 0 && (
                <table className="w-full text-left">
                  <thead className="text-muted"><tr><th>path</th><th>current</th><th>proposed</th></tr></thead>
                  <tbody>
                    {Object.entries(validation.diff as Record<string, { current: unknown; proposed: unknown }>).map(([k, v]) => (
                      <tr key={k} className="border-t border-line"><td className="font-mono">{k}</td><td className="text-muted">{JSON.stringify(v.current)}</td><td>{JSON.stringify(v.proposed)}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
              {validation.ok && validation.recompute_required && <p className="mt-1 text-amber-700">Applying recomputes every tier from raw daily rows.</p>}
              {validation.ok && <p className="mt-1 text-muted">proposed hash {validation.hash}</p>}
            </div>
          )}
        </div>
        <div className="text-xs">
          <h4 className="mb-1 font-semibold uppercase text-muted">History</h4>
          <ul className="space-y-1">
            {(history.data ?? []).map((h) => (
              <li key={h.hash} className="rounded border border-line px-2 py-1">
                <div className="flex justify-between"><span className="font-mono">{h.hash}</span><span className="text-muted">{ago(h.applied_at)}</span></div>
                <div className="text-muted">{h.applied_by}{h.note ? ` · ${h.note}` : ""}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  );
}
