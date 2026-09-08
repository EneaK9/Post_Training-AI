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

// ---- Generate + Model screens --------------------------------------------------------------

export function useEpisode(id: string | null) {
  return useQuery({
    queryKey: ["episode", id],
    queryFn: () => unwrap(api.GET("/api/episodes/{episode_id}", { params: { path: { episode_id: id! }, query: { include_trace: true } } })),
    enabled: !!id,
    refetchInterval: 15_000,
  });
}

export function useCreateEpisode() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["EpisodeCreate"]) => unwrap(api.POST("/api/episodes", { body })),
    onSuccess: () => inv(["episodes"], keys.briefs),
  });
}

export function useGenerate() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["GenerateIn"]) => unwrap(api.POST("/api/generate", { body })),
    onSuccess: (_d, v) => inv(["trajectories"], keys.reviewQueue, ["episode", v.episode_id ?? ""], ["combinations"], keys.audit),
  });
}

export function useStepEpisode() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, approval, k, no_llm }: { id: string; approval: "manual" | "top_rm" | "random"; k?: number; no_llm?: boolean }) =>
      unwrap(api.POST("/api/episodes/{episode_id}/step", { params: { path: { episode_id: id }, query: { approval, k, no_llm } } })),
    onSuccess: (_d, v) => inv(["episode", v.id], ["trajectories"], keys.reviewQueue),
  });
}

export function useSimulateEpisode() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, days }: { id: string; days: number }) =>
      unwrap(api.POST("/api/episodes/{episode_id}/simulate", { params: { path: { episode_id: id }, query: { days } } })),
    onSuccess: (_d, v) => inv(["episode", v.id], ["trajectories"], keys.reviewQueue, keys.signalsQueue, ["stats"]),
  });
}

export function useStopEpisode() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id }: { id: string }) => unwrap(api.POST("/api/episodes/{episode_id}/stop", { params: { path: { episode_id: id } } })),
    onSuccess: (_d, v) => inv(["episode", v.id], ["episodes"]),
  });
}

export function useCreateBrief() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["BriefCreate"]) => unwrap(api.POST("/api/briefs", { body })),
    onSuccess: () => inv(keys.briefs),
  });
}

export function useArchiveSample(briefId: string | null) {
  return useQuery({
    queryKey: ["archiveSample", briefId],
    queryFn: () => unwrap(api.GET("/api/archive/sample", { params: { query: { brief_id: briefId! } } })),
    enabled: !!briefId,
  });
}

export function useLoopAStats() {
  return useQuery({ queryKey: ["stats", "loop_a"], queryFn: () => unwrap(api.GET("/api/stats/loop_a")) });
}
export function useReviewStats() {
  return useQuery({ queryKey: ["stats", "reviews"], queryFn: () => unwrap(api.GET("/api/stats/reviews")) });
}
export function useVerifierStats() {
  return useQuery({ queryKey: ["stats", "verifier"], queryFn: () => unwrap(api.GET("/api/stats/verifier")) });
}
export function useRewardModelStats() {
  return useQuery({ queryKey: ["stats", "reward_model"], queryFn: () => unwrap(api.GET("/api/stats/reward_model")) });
}
export function useArchitecture() {
  return useQuery({ queryKey: ["stats", "architecture"], queryFn: () => unwrap(api.GET("/api/stats/architecture")) });
}

export function useConfig() {
  return useQuery({ queryKey: keys.config, queryFn: () => unwrap(api.GET("/api/config")) });
}
export function useConfigHistory() {
  return useQuery({ queryKey: ["config", "history"], queryFn: () => unwrap(api.GET("/api/config/history")) });
}
export function useValidateConfig() {
  return useMutation({ mutationFn: (body: Schemas["ConfigIn"]) => unwrap(api.POST("/api/config/validate", { body })) });
}
export function useApplyConfig() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["ConfigIn"]) => unwrap(api.POST("/api/config/apply", { body })),
    onSuccess: () => inv(keys.config, ["config"], ["stats"], ["trajectories"], keys.audit),
  });
}

export function useAccounts() {
  return useQuery({ queryKey: ["accounts"], queryFn: () => unwrap(api.GET("/api/meta/accounts")) });
}
export function useKillSwitch() {
  return useQuery({ queryKey: ["killSwitch"], queryFn: () => unwrap(api.GET("/api/meta/kill_switch")) });
}
export function useSetKillSwitch() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["KillSwitchIn"]) => unwrap(api.PUT("/api/meta/kill_switch", { body })),
    onSuccess: () => inv(["killSwitch"], keys.audit),
  });
}

export function useShipEpisode() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id }: { id: string }) => unwrap(api.POST("/api/episodes/{episode_id}/ship", { params: { path: { episode_id: id } } })),
    onSuccess: (_d, v) => inv(["episode", v.id], ["trajectories"], keys.reviewQueue, keys.audit),
  });
}

// ---- Phase 6: training runs, eval, reward model, feedback ---------------------------------
export function useTrainingRuns() {
  return useQuery({ queryKey: ["training", "runs"], queryFn: () => unwrap(api.GET("/api/training/runs")) });
}
export function useLaunchTraining() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["TrainingRunIn"]) => unwrap(api.POST("/api/training/runs", { body })),
    onSuccess: () => inv(["training"], ["stats"], keys.audit),
  });
}
export function useEvals() {
  return useQuery({ queryKey: ["evals"], queryFn: () => unwrap(api.GET("/api/eval")) });
}
export function useLaunchEval() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["EvalLaunchIn"]) => unwrap(api.POST("/api/eval/launch", { body })),
    onSuccess: () => inv(["evals"], ["stats"], ["trajectories"], keys.audit),
  });
}
export function useGoldGap() {
  return useQuery({ queryKey: ["stats", "gold_gap"], queryFn: () => unwrap(api.GET("/api/stats/gold_gap")) });
}
export function useTrainRewardModel() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: Schemas["RMTrainIn"]) => unwrap(api.POST("/api/feedback/retrain_rm", { body })),
    onSuccess: () => inv(["stats"], keys.audit),
  });
}
export function useSuggestRelations() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/feedback/suggest_relations")),
    onSuccess: () => inv(["relations"], ["cards"], keys.audit),
  });
}
