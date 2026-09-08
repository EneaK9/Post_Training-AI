export function shortId(id: string | null | undefined): string {
  return id ? id.replace(/-/g, "").slice(0, 8) : "";
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return "";
  const ms = Date.now() - new Date(iso).getTime();
  const d = Math.floor(ms / 86400000);
  if (d >= 1) return `${d}d ago`;
  const h = Math.floor(ms / 3600000);
  if (h >= 1) return `${h}h ago`;
  return `${Math.max(1, Math.floor(ms / 60000))}m ago`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  return x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 2): string {
  return x === null || x === undefined ? "–" : x.toFixed(digits);
}

export function clamp(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
