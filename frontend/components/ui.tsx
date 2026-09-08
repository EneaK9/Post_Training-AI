"use client";

import clsx from "clsx";
import { X } from "lucide-react";
import type { ButtonHTMLAttributes, ComponentProps, ReactNode, SelectHTMLAttributes } from "react";

export function Button({
  variant = "default",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "danger" | "ghost"; size?: "sm" | "md" }) {
  return (
    <button
      {...props}
      className={clsx(
        "inline-flex items-center gap-1 rounded-md border font-medium transition disabled:opacity-50",
        size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
        variant === "default" && "border-line bg-panel hover:bg-accent-soft",
        variant === "primary" && "border-accent bg-accent text-white hover:opacity-90",
        variant === "danger" && "border-red-600 bg-red-600 text-white hover:opacity-90",
        variant === "ghost" && "border-transparent hover:bg-accent-soft",
        className,
      )}
    />
  );
}

export function Badge({ children, className, ...rest }: ComponentProps<"span">) {
  return (
    <span {...rest} className={clsx("inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium leading-4", className)}>
      {children}
    </span>
  );
}

export function Panel({ title, actions, children, className, testId }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; testId?: string }) {
  return (
    <section data-testid={testId} className={clsx("rounded-lg border border-line bg-panel", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
          <h2 className="text-sm font-semibold">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Input(props: ComponentProps<"input">) {
  return <input {...props} className={clsx("w-full rounded-md border border-line bg-background px-2 py-1.5 text-sm", props.className)} />;
}

export function Textarea(props: ComponentProps<"textarea">) {
  return <textarea {...props} className={clsx("w-full rounded-md border border-line bg-background px-2 py-1.5 text-sm", props.className)} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={clsx("rounded-md border border-line bg-background px-2 py-1.5 text-sm", props.className)} />;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-xs">
      <span className="mb-1 block text-muted">{label}</span>
      {children}
    </label>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string; count?: number }[]; value: T; onChange: (t: T) => void }) {
  return (
    <nav className="flex flex-wrap gap-1 border-b border-line" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === value}
          data-testid={`tab-${t.id}`}
          onClick={() => onChange(t.id)}
          className={clsx(
            "-mb-px border-b-2 px-3 py-2 text-sm",
            t.id === value ? "border-accent font-semibold" : "border-transparent text-muted hover:text-foreground",
          )}
        >
          {t.label}
          {t.count !== undefined && <span className="ml-1 rounded bg-accent-soft px-1 text-xs">{t.count}</span>}
        </button>
      ))}
    </nav>
  );
}

export function Modal({ title, open, onClose, children, wide }: { title: string; open: boolean; onClose: () => void; children: ReactNode; wide?: boolean }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 p-6" onClick={onClose} role="dialog" aria-modal="true">
      <div className={clsx("mt-8 w-full rounded-lg border border-line bg-panel shadow-xl", wide ? "max-w-4xl" : "max-w-xl")} onClick={(e) => e.stopPropagation()}>
        <header className="flex items-center justify-between border-b border-line px-4 py-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <button onClick={onClose} aria-label="close" className="rounded p-1 hover:bg-accent-soft">
            <X size={16} />
          </button>
        </header>
        <div className="max-h-[75vh] overflow-auto p-4">{children}</div>
      </div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-muted">{children}</p>;
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <p className="rounded border border-red-300 bg-red-50 px-2 py-1 text-xs text-red-700 dark:bg-red-950/40">{msg}</p>;
}
