"use client";

import { WorkspaceProvider, useWorkspace } from "@/state/workspace";
import LibrarySidebar from "@/components/LibrarySidebar";
import InspectorPanel from "@/components/InspectorPanel";
import TabBar from "@/components/TabBar";
import DocumentTab from "@/components/tabs/DocumentTab";
import StructureTab from "@/components/tabs/StructureTab";
import ChunksTab from "@/components/tabs/ChunksTab";
import RetrievalTab from "@/components/tabs/RetrievalTab";
import EvaluationTab from "@/components/tabs/EvaluationTab";
import UnprocessedNotice from "@/components/UnprocessedNotice";
import { isProcessedDoc } from "@/lib/types";
import dynamic from "next/dynamic";
import styles from "@/app/workspace.module.css";

// The graph tab pulls the 3D stack; load its bundle only when opened.
const GraphTab = dynamic(() => import("@/components/tabs/GraphTab"), {
  loading: () => (
    <div role="status" className={styles.graphTabLoading}>
      loading knowledge graph …
    </div>
  ),
});

function TopBar() {
  const { state } = useWorkspace();
  const doc = state.documents?.find((d) => d.document_id === state.documentId);
  return (
    <header className={styles.topbar}>
      <h1 className={styles.brand}>
        JUNG&nbsp;ARCHIVE
        <span className={styles.brandSub}>document intelligence inspector</span>
      </h1>
      <div className={styles.status} aria-live="polite">
        {doc ? (
          <>
            <strong>{doc.title}</strong>
            <span>{doc.author}</span>
            <em data-testid="topbar-status">
              {doc.source_type} · {doc.status} · {doc.page_count} pp
            </em>
          </>
        ) : (
          "no document"
        )}
      </div>
    </header>
  );
}

function MainPanel() {
  const { state } = useWorkspace();
  const doc =
    state.documents?.find((d) => d.document_id === state.documentId) ?? null;

  // Honest status screen instead of a crash for undiscovered/review/
  // excluded/error documents.
  if (!isProcessedDoc(doc)) {
    return (
      <main
        className={styles.main}
        id={`panel-${state.tab}`}
        role="tabpanel"
        aria-labelledby={`tab-${state.tab}`}
      >
        {doc ? (
          <UnprocessedNotice doc={doc} />
        ) : (
          <div className={styles.noDoc}>no document selected</div>
        )}
      </main>
    );
  }

  return (
    <main
      className={styles.main}
      id={`panel-${state.tab}`}
      role="tabpanel"
      aria-labelledby={`tab-${state.tab}`}
    >
      {state.tab === "document" && <DocumentTab />}
      {state.tab === "structure" && <StructureTab />}
      {state.tab === "chunks" && <ChunksTab />}
      {state.tab === "retrieval" && <RetrievalTab />}
      {state.tab === "evaluation" && <EvaluationTab />}
      {state.tab === "graph" && <GraphTab />}
    </main>
  );
}

export default function Workspace() {
  return (
    <WorkspaceProvider>
      <div className={styles.shell}>
        <TopBar />
        <div className={styles.body}>
          <LibrarySidebar />
          <div className={styles.center}>
            <MainPanel />
          </div>
          <InspectorPanel />
        </div>
        <TabBar />
      </div>
    </WorkspaceProvider>
  );
}
