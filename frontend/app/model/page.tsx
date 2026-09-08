"use client";

import { Suspense, useState } from "react";
import { ArchitectureDiagram } from "@/components/model/architecture";
import { ConfigEditor } from "@/components/model/config-editor";
import { AccountsPanel, LoopAStatsPanel, ReviewStatsPanel, RewardModelPanel, RunsPanel, VerifierPanel } from "@/components/model/stats";
import { Shell } from "@/components/shell";

export default function ModelPage() {
  return (
    <Shell>
      <Suspense fallback={null}>
        <ModelScreen />
      </Suspense>
    </Shell>
  );
}

function ModelScreen() {
  const [section, setSection] = useState<string | null>(null);
  return (
    <div className="space-y-3">
      <ArchitectureDiagram onOpenConfig={setSection} />
      <ConfigEditor focusSection={section} />
      <LoopAStatsPanel />
      <RunsPanel />
      <div className="grid gap-3 lg:grid-cols-2">
        <RewardModelPanel />
        <VerifierPanel />
      </div>
      <ReviewStatsPanel />
      <AccountsPanel />
    </div>
  );
}
