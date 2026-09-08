"use client";

import clsx from "clsx";
import { Database, LogOut, Search, Sparkles, SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ApiError } from "@/lib/api/client";
import { useLogout, useMe, useSearch } from "@/lib/api/hooks";
import { ObjectDrawer } from "./drawer";
import { Badge } from "./ui";

const NAV = [
  { href: "/generate", label: "Generate", icon: Sparkles },
  { href: "/model", label: "Model", icon: SlidersHorizontal },
  { href: "/data", label: "Data", icon: Database },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const me = useMe();
  const router = useRouter();
  const pathname = usePathname();
  const logout = useLogout();

  useEffect(() => {
    if (me.error instanceof ApiError && me.error.status === 401) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [me.error, router, pathname]);

  if (me.isLoading) return <div className="p-6 text-sm text-muted">Loading…</div>;
  if (!me.data) return null;

  return (
    <div className="flex min-h-screen">
      <aside className="flex w-52 shrink-0 flex-col border-r border-line bg-panel">
        <div className="px-4 py-4">
          <div className="text-base font-semibold">Outlier AI</div>
          <div className="text-[11px] text-muted">search the tail, not the mean</div>
        </div>
        <nav className="flex flex-col gap-0.5 px-2">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={clsx("flex items-center gap-2 rounded-md px-2 py-1.5 text-sm", pathname.startsWith(n.href) ? "bg-accent-soft font-medium" : "hover:bg-accent-soft/60")}
            >
              <n.icon size={16} /> {n.label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto border-t border-line p-3 text-xs">
          <div className="truncate" title={me.data.email}>{me.data.display_name || me.data.email}</div>
          <div className="mt-1 flex items-center justify-between">
            <Badge className="bg-accent-soft text-accent">{me.data.role}</Badge>
            <button
              onClick={() => logout.mutate(undefined, { onSuccess: () => router.replace("/login") })}
              className="flex items-center gap-1 text-muted hover:text-foreground"
            >
              <LogOut size={14} /> log out
            </button>
          </div>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-2">
          <Suspense fallback={null}>
            <GlobalSearch />
          </Suspense>
        </header>
        <main className="min-w-0 flex-1 p-4">{children}</main>
      </div>
      <Suspense fallback={null}>
        <ObjectDrawer />
      </Suspense>
    </div>
  );
}

function GlobalSearch() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const results = useSearch(q);
  return (
    <div className="relative w-full max-w-xl">
      <div className="flex items-center gap-2 rounded-md border border-line bg-background px-2 py-1">
        <Search size={14} className="text-muted" />
        <input
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onFocus={() => setOpen(true)}
          placeholder="Search cards, briefs, trajectories, signals, episodes"
          className="w-full bg-transparent text-sm outline-none"
          data-testid="global-search"
        />
      </div>
      {open && results.data && results.data.hits.length > 0 && (
        <ul className="absolute z-40 mt-1 w-full rounded-md border border-line bg-panel shadow-lg">
          {results.data.hits.slice(0, 12).map((h) => (
            <li key={`${h.type}-${h.id}`}>
              <Link href={`?open=${h.type}:${h.id}`} scroll={false} className="flex items-center justify-between gap-2 px-3 py-1.5 text-sm hover:bg-accent-soft" onClick={() => setOpen(false)}>
                <span className="truncate">{h.label}</span>
                <span className="shrink-0 text-xs text-muted">{h.type} · {h.sub}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
