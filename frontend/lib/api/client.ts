import createClient from "openapi-fetch";
import type { components, paths } from "./types";

export type Schemas = components["schemas"];
export type TrajectoryOut = Schemas["TrajectoryOut"];
export type TrajectoryDetail = Schemas["TrajectoryDetail"];
export type CardOut = Schemas["CardOut"];
export type BriefOut = Schemas["BriefOut"];
export type SignalOut = Schemas["SignalOut"];
export type CombinationOut = Schemas["CombinationOut"];
export type EpisodeOut = Schemas["EpisodeOut"];
export type RelationOut = Schemas["RelationOut"];
export type AuditOut = Schemas["AuditOut"];
export type UserOut = Schemas["UserOut"];

export const api = createClient<paths>({ baseUrl: "", credentials: "include" });

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `request failed (${status})`);
    this.status = status;
    this.detail = detail;
  }
}

type Result<T> = { data?: T; error?: unknown; response: Response };

export async function unwrap<T>(p: Promise<Result<T>>): Promise<T> {
  const { data, error, response } = await p;
  if (error !== undefined || !response.ok) {
    const detail = (error as { detail?: unknown } | undefined)?.detail ?? error;
    throw new ApiError(response.status, detail);
  }
  return data as T;
}

/** storage:// URIs become /api/files/<key>; anything else passes through. */
export function fileUrl(uri: string | null | undefined): string | null {
  if (!uri) return null;
  return uri.startsWith("storage://") ? `/api/files/${uri.slice("storage://".length)}` : uri;
}

export async function uploadCsv(path: string, file: File): Promise<Schemas["ImportResult"]> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(path, { method: "POST", body, credentials: "include" });
  if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({})))?.detail);
  return (await res.json()) as Schemas["ImportResult"];
}
