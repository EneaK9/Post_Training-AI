"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Button, ErrorNote, Field, Input } from "@/components/ui";
import { api, unwrap } from "@/lib/api/client";

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form
        className="w-full max-w-sm space-y-3 rounded-lg border border-line bg-panel p-6"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError(null);
          try {
            await unwrap(api.POST("/api/auth/login", { body: { email, password } }));
            router.replace(params.get("next") || "/data");
          } catch (err) {
            setError(err);
          } finally {
            setBusy(false);
          }
        }}
      >
        <h1 className="text-lg font-semibold">Outlier AI</h1>
        <p className="text-xs text-muted">Sign in as an operator, researcher, or expert.</p>
        <Field label="Email"><Input name="email" type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required /></Field>
        <Field label="Password"><Input name="password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required /></Field>
        <ErrorNote error={error} />
        <Button type="submit" variant="primary" disabled={busy} className="w-full justify-center">Sign in</Button>
      </form>
    </main>
  );
}
