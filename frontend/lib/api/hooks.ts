"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";

export const keys = {
  me: ["me"] as const,
  cards: ["cards"] as const,
  relations: ["relations"] as const,
  combinations: (sort: string) => ["combinations", sort] as const,
  briefs: ["briefs"] as const,
  episodes: (briefId: string) => ["episodes", briefId] as const,
  trajectories: (params: Record<string, unknown>) => ["trajectories", params] as const,
  trajectory: (id: string) => ["trajectory", id] as const,
  reviewQueue: ["reviewQueue"] as const,
  signalsQueue: ["signalsQueue"] as const,
  audit: ["audit"] as const,
  config: ["config"] as const,
  search: (q: string) => ["search", q] as const,
};

export function useMe() {
  return useQuery({ queryKey: keys.me, queryFn: () => unwrap(api.GET("/api/auth/me")), retry: false });
}

export function useCards() {
  return useQuery({ queryKey: keys.cards, queryFn: () => unwrap(api.GET("/api/cards")) });
}

export function useRelations() {
  return useQuery({ queryKey: keys.relations, queryFn: () => unwrap(api.GET("/api/cards/relations")) });
}

export function useCombinations(sort: string = "tier2_rate") {
  return useQuery({
    queryKey: keys.combinations(sort),
    queryFn: () => unwrap(api.GET("/api/combinations", { params: { query: { sort, limit: 300 } } })),
  });
}

export function useBriefs() {
  return useQuery({ queryKey: keys.briefs, queryFn: () => unwrap(api.GET("/api/briefs")) });
}

export function useBriefEpisodes(briefId: string | null) {
  return useQuery({
    queryKey: keys.episodes(briefId ?? ""),
    queryFn: () => unwrap(api.GET("/api/briefs/{brief_id}/episodes", { params: { path: { brief_id: briefId! } } })),
    enabled: !!briefId,
  });
}

export type TrajectoryFilters = {
  brief_id?: string;
  tier?: number;
  min_tier?: number;
  card?: string;
  mismatch?: boolean;
  signal_kind?: string;
  typicality?: string;
  review_label?: string;
  unlabeled?: boolean;
  episode_id?: string;
  limit?: number;
  offset?: number;
};

export function useTrajectories(filters: TrajectoryFilters) {
  return useQuery({
    queryKey: keys.trajectories(filters),
    queryFn: () => unwrap(api.GET("/api/trajectories", { params: { query: filters } })),
  });
}

export function useTrajectory(id: string | null) {
  return useQuery({
    queryKey: keys.trajectory(id ?? ""),
    queryFn: () => unwrap(api.GET("/api/trajectories/{trajectory_id}", { params: { path: { trajectory_id: id! } } })),
    enabled: !!id,
  });
}

export function useReviewQueue() {
  return useQuery({ queryKey: keys.reviewQueue, queryFn: () => unwrap(api.GET("/api/reviews/queue", { params: { query: { limit: 100 } } })) });
}

export function useSignalsQueue() {
  return useQuery({ queryKey: keys.signalsQueue, queryFn: () => unwrap(api.GET("/api/signals/queue")) });
}

export function useAudit() {
  return useQuery({ queryKey: keys.audit, queryFn: () => unwrap(api.GET("/api/audit", { params: { query: { limit: 200 } } })) });
}

export function useSearch(q: string) {
  return useQuery({
    queryKey: keys.search(q),
    queryFn: () => unwrap(api.GET("/api/search", { params: { query: { q } } })),
    enabled: q.trim().length > 1,
  });
}

function useInvalidate() {
  const qc = useQueryClient();
  return (...groups: readonly (readonly unknown[])[]) => Promise.all(groups.map((k) => qc.invalidateQueries({ queryKey: k })));
}

export function useCreateCard() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["CardCreate"]) => unwrap(api.POST("/api/cards", { body })),
    onSuccess: () => inv(keys.cards, keys.audit),
  });
}

export function useUpdateCard() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Schemas["CardUpdate"] }) =>
      unwrap(api.PUT("/api/cards/{card_id}", { params: { path: { card_id: id } }, body })),
    onSuccess: () => inv(keys.cards, keys.audit),
  });
}

export function useSetCardStatus() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: "active" | "retired" }) =>
      status === "active"
        ? unwrap(api.POST("/api/cards/{card_id}/activate", { params: { path: { card_id: id } } }))
        : unwrap(api.POST("/api/cards/{card_id}/retire", { params: { path: { card_id: id } } })),
    onSuccess: () => inv(keys.cards, keys.audit),
  });
}

export function useCardVersions(id: string | null) {
  return useQuery({
    queryKey: ["cardVersions", id],
    queryFn: () => unwrap(api.GET("/api/cards/{card_id}/versions", { params: { path: { card_id: id! } } })),
    enabled: !!id,
  });
}

export function useReview() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Schemas["ReviewIn"] }) =>
      unwrap(api.POST("/api/trajectories/{trajectory_id}/review", { params: { path: { trajectory_id: id } }, body })),
    onSuccess: (_d, v) => inv(keys.reviewQueue, ["trajectories"], keys.trajectory(v.id), keys.audit),
  });
}

export function useEditCopy() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Schemas["CopyEdit"] }) =>
      unwrap(api.PUT("/api/trajectories/{trajectory_id}/copy", { params: { path: { trajectory_id: id } }, body })),
    onSuccess: (_d, v) => inv(keys.reviewQueue, ["trajectories"], keys.trajectory(v.id)),
  });
}

export function useAddNote() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      unwrap(api.POST("/api/trajectories/{trajectory_id}/notes", { params: { path: { trajectory_id: id } }, body: { text } })),
    onSuccess: (_d, v) => inv(["trajectories"], keys.trajectory(v.id)),
  });
}

export function useSignalDecision() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: "confirm" | "reject" }) =>
      decision === "confirm"
        ? unwrap(api.PUT("/api/signals/{signal_id}/confirm", { params: { path: { signal_id: id } } }))
        : unwrap(api.PUT("/api/signals/{signal_id}/reject", { params: { path: { signal_id: id } } })),
    onSuccess: () => inv(keys.signalsQueue, ["trajectories"], ["trajectory"], keys.audit),
  });
}

export function useNameAsCard() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Schemas["NameAsCardIn"] }) =>
      unwrap(api.POST("/api/combinations/{combination_id}/name_as_card", { params: { path: { combination_id: id } }, body })),
    onSuccess: () => inv(keys.cards, ["combinations"], keys.audit),
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/auth/logout")),
    onSuccess: () => qc.clear(),
  });
}
