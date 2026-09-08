"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { AuditTab, ImportTab } from "@/components/data/audit-import";
import { BriefsColumn } from "@/components/data/briefs";
import { CombinationsTab } from "@/components/data/combinations";
import { GraphTab } from "@/components/data/graph";
import { HistoryColumn } from "@/components/data/history";
import { PlaybookColumn } from "@/components/data/playbook";
import { ReviewQueueTab, SignalsQueueTab } from "@/components/data/queues";
import { Shell } from "@/components/shell";
import { Tabs } from "@/components/ui";
import { useReviewQueue, useSignalsQueue } from "@/lib/api/hooks";

type Tab = "board" | "combinations" | "review" | "signals" | "graph" | "audit" | "import";

export default function DataPage() {
  return (
    <Shell>
      <Suspense fallback={null}>
        <DataScreen />
      </Suspense>
    </Shell>
  );
}

function DataScreen() {
  const params = useSearchParams();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>((params.get("tab") as Tab) || "board");
  const briefId = params.get("brief");
  const review = useReviewQueue();
  const signals = useSignalsQueue();
  const setBrief = (id: string | null) => {
    const next = new URLSearchParams(params.toString());
    if (id) next.set("brief", id);
    else next.delete("brief");
    router.replace(`/data?${next}`, { scroll: false });
  };
  return (
    <div className="space-y-3">
      <Tabs<Tab>
        value={tab}
        onChange={setTab}
        tabs={[
          { id: "board", label: "Board" },
          { id: "combinations", label: "Combinations" },
          { id: "review", label: "Review queue", count: review.data?.length },
          { id: "signals", label: "Signals queue", count: signals.data?.length },
          { id: "graph", label: "Graph" },
          { id: "audit", label: "Audit" },
          { id: "import", label: "Import" },
        ]}
      />
      {tab === "board" && (
        <div className="grid gap-3 lg:grid-cols-[minmax(280px,1fr)_minmax(260px,1fr)_minmax(420px,2fr)]">
          <PlaybookColumn />
          <BriefsColumn selected={briefId} onSelect={setBrief} />
          <HistoryColumn briefId={briefId} />
        </div>
      )}
      {tab === "combinations" && <CombinationsTab />}
      {tab === "review" && <ReviewQueueTab />}
      {tab === "signals" && <SignalsQueueTab />}
      {tab === "graph" && <GraphTab />}
      {tab === "audit" && <AuditTab />}
      {tab === "import" && <ImportTab />}
    </div>
  );
}
