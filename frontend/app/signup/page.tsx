"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Button, ErrorNote, Field, Input } from "@/components/ui";
import { api, unwrap } from "@/lib/api/client";

type Role = "operator" | "researcher" | "expert";

const ROLE_HELP: Record<Role, string> = {
  expert: "authors cards, labels ideas, confirms signals",
  operator: "runs episodes, ships approved ideas, manages accounts",
  researcher: "edits config, trains models, launches evals",
};

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState<Role>("expert");
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    unwrap(api.GET("/api/auth/options"))
      .then((o) => setEnabled(o.signup_enabled))
      .catch(() => setEnabled(true));
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form
        className="w-full max-w-sm space-y-3 rounded-lg border border-line bg-panel p-6"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError(null);
          try {
            await unwrap(api.POST("/api/auth/signup", { body: { email, password, display_name: displayName, role } }));
            router.replace("/data");
          } catch (err) {
            setError(err);
          } finally {
            setBusy(false);
          }
        }}
      >
        <h1 className="text-lg font-semibold">Create an account</h1>
        <p className="text-xs text-muted">Pick the role you work in. Roles gate what you can change, not what you can see.</p>
        {enabled === false && <p className="rounded border border-line bg-panel p-2 text-xs text-muted" data-testid="signup-disabled">Sign-up is disabled on this deployment. Ask an operator to create your account.</p>}
        <Field label="Email"><Input name="email" type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required /></Field>
        <Field label="Display name"><Input name="display_name" autoComplete="name" value={displayName} onChange={(e) => setDisplayName(e.target.value)} /></Field>
        <Field label="Password (8+ characters)"><Input name="password" type="password" autoComplete="new-password" minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} required /></Field>
        <Field label="Role">
          <select
            name="role"
            data-testid="signup-role"
            className="w-full rounded border border-line bg-panel px-2 py-1.5 text-sm"
            value={role}
            onChange={(e) => setRole(e.target.value as Role)}
          >
            {(Object.keys(ROLE_HELP) as Role[]).map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <p className="mt-1 text-xs text-muted">{ROLE_HELP[role]}</p>
        </Field>
        <ErrorNote error={error} />
        <Button type="submit" variant="primary" disabled={busy || enabled === false} className="w-full justify-center">Create account</Button>
        <p className="text-center text-xs text-muted">Already have one? <a className="underline" href="/login">Sign in</a></p>
      </form>
    </main>
  );
}
