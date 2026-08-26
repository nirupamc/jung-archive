"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type {
  EvalFailureCase,
  EvalLatestSummary,
  EvalLevelMetrics,
  EvalRunDetail,
  EvalRunListItem,
} from "@/lib/evaluations";
import { useWorkspace } from "@/state/workspace";
import styles from "./EvaluationTab.module.css";

const MODE_LABELS: Record<string, string> = {
  dense: "Dense",
  bm25: "BM25",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + Reranker",
};

type SubView = "overview" | "experiments" | "failures";

export default function EvaluationTab() {
  const [subView, setSubView] = useState<SubView>("overview");
  const [latest, setLatest] = useState<EvalLatestSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .evaluationLatest()
      .then((s) => !cancelled && setLatest(s))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={styles.wrap}>
      <div
        className={styles.subNav}
        role="radiogroup"
        aria-label="evaluation sub-views"
      >
        {(["overview", "experiments", "failures"] as SubView[]).map((v) => (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={subView === v}
            data-testid={`eval-sub-${v}`}
            className={[
              styles.subBtn,
              subView === v ? styles.subActive : "",
            ].join(" ")}
            onClick={() => setSubView(v)}
          >
            {v.toUpperCase()}
          </button>
        ))}
      </div>

      {error && (
        <div role="alert" className={styles.error}>
          evaluation data unavailable: {error}
        </div>
      )}

      {!error && !latest && (
        <div role="status" className={styles.loading}>
          loading evaluation results …
        </div>
      )}

      {latest && subView === "overview" && <Overview summary={latest} />}
      {latest && subView === "experiments" && (
        <Experiments latest={latest} />
      )}
      {subView === "failures" && <Failures onReady={setError} />}
    </div>
  );
}

// ----------------------------------------------------------------------
// OVERVIEW

function Overview({ summary }: { summary: EvalLatestSummary }) {
  const byMode = useMemo(
    () =>
      [...summary.aggregates].sort((a) =>
        a.mode === "hybrid_rerank" ? 1 : 0,
      ),
    [summary],
  );
  return (
    <section aria-label="benchmark overview">
      <p className={styles.metaLine}>
        run <code>{summary.run_id}</code> · dataset{" "}
        {summary.dataset_version} · {summary.aggregates[0]?.n_questions ?? "?"}{" "}
        questions · avg {summary.avg_query_time_ms ?? "?"} ms/query
      </p>
      <table className={styles.table} data-testid="overview-table">
        <caption className="srOnly">retrieval benchmark results</caption>
        <thead>
          <tr>
            <th scope="col">Mode</th>
            <th scope="col">Hit@1</th>
            <th scope="col">Recall@5</th>
            <th scope="col">MRR</th>
            <th scope="col">NDCG@5</th>
            <th scope="col">Recall@10</th>
          </tr>
        </thead>
        <tbody>
          {byMode.map((agg) => {
            const cm = agg.chunk_metrics;
            return (
              <tr key={agg.mode} data-testid={`row-${agg.mode}`}>
                <th scope="row">{MODE_LABELS[agg.mode] ?? agg.mode}</th>
                <td>{cm.hit_at_k["1"]?.toFixed(3) ?? "—"}</td>
                <td>{cm.recall_at_k["5"]?.toFixed(3) ?? "—"}</td>
                <td>{cm.mrr.toFixed(3)}</td>
                <td>{cm.ndcg_at_k["5"]?.toFixed(3) ?? "—"}</td>
                <td>{cm.recall_at_k["10"]?.toFixed(3) ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {Object.keys(summary.failure_counts).length > 0 && (
        <section aria-label="failure counts" className={styles.failCounts}>
          <h3>failure categories</h3>
          <ul>
            {Object.entries(summary.failure_counts).map(([cat, n]) => (
              <li key={cat}>
                <code>{cat}</code>: {n}
              </li>
            ))}
          </ul>
        </section>
      )}

      {summary.evidence_summary &&
        Object.keys(summary.evidence_summary).length > 0 && (
          <section aria-label="evidence quality" className={styles.evidenceBox}>
            <h3>evidence quality (reranked path)</h3>
            <dl className={styles.kv}>
              {Object.entries(summary.evidence_summary).map(([k, v]) => (
                <div key={k}>
                  <dt>{k.replace(/_/g, " ")}</dt>
                  <dd>
                    {typeof v === "number" ? v.toFixed(3) : v}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        )}

      <section aria-label="generation metrics" className={styles.genNote}>
        generation evaluation:{" "}
        <strong data-testid="generation-status">
          {summary.generation_eval.status}
        </strong>{" "}
        — {summary.generation_eval.note}
      </section>
    </section>
  );
}

// ----------------------------------------------------------------------
// EXPERIMENTS (run comparison)

function Experiments({ latest }: { latest: EvalLatestSummary }) {
  const [runs, setRuns] = useState<EvalRunListItem[] | null>(null);
  const [aId, setAId] = useState<string>("");
  const [bId, setBId] = useState<string>(latest.run_id);
  const [detailB, setDetailB] = useState<EvalRunDetail | null>(null);
  const [detailA, setDetailA] = useState<EvalRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .evaluationRuns()
      .then((rs) => {
        setRuns(rs);
        if (!aId && rs.length > 0) setAId(rs[0].run_id);
      })
      .catch((e: Error) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!aId || !bId) return;
    Promise.all([api.evaluationRun(aId), api.evaluationRun(bId)])
      .then(([da, db]) => {
        setError(null);
        setDetailA(da);
        setDetailB(db);
      })
      .catch((e: Error) => setError(e.message));
  }, [aId, bId]);

  const deltas = useMemo(() => {
    if (!detailA || !detailB) return [] as DeltaRow[];
    return computeDeltas(detailA, detailB);
  }, [detailA, detailB]);

  if (error)
    return <div role="alert" className={styles.error}>{error}</div>;

  return (
    <section aria-label="experiment comparison">
      <p className={styles.metaLine}>compare two runs</p>
      <div className={styles.selectorRow}>
        <label>
          RUN A{" "}
          <select
            value={aId}
            data-testid="select-run-a"
            onChange={(e) => setAId(e.target.value)}
          >
            {(runs ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id.slice(0, 16)} ({r.modes.join(",")})
              </option>
            ))}
          </select>
        </label>
        <span className={styles.vs}>vs.</span>
        <label>
          RUN B{" "}
          <select
            value={bId}
            data-testid="select-run-b"
            onChange={(e) => setBId(e.target.value)}
          >
            {(runs ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id.slice(0, 16)} ({r.modes.join(",")})
              </option>
            ))}
          </select>
        </label>
      </div>

      {deltas.length > 0 ? (
        <table className={styles.table} data-testid="delta-table">
          <caption className="srOnly">
            metric deltas A to B; negative values are regressions
          </caption>
          <thead>
            <tr>
              <th scope="col">Mode</th>
              <th scope="col">Metric</th>
              <th scope="col">Δ A → B</th>
            </tr>
          </thead>
          <tbody>
            {deltas.map((d) => (
              <tr key={`${d.mode}-${d.metric}`} data-testid={`delta-${d.mode}-${d.metric}`}>
                <th scope="row">{MODE_LABELS[d.mode] ?? d.mode}</th>
                <td>{d.metric}</td>
                <td className={d.delta >= 0 ? styles.pos : styles.neg}>
                  {d.delta >= 0 ? "+" : ""}
                  {d.delta.toFixed(3)}
                  {d.delta < 0 ? " (regression)" : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        detailA && detailB && (
          <p className={styles.loading}>no common modes between runs</p>
        )
      )}
      <p className={styles.caveat}>
        Deltas are raw differences on {`${30}`} benchmark questions; they are
        not significance-tested.
      </p>
    </section>
  );
}

interface DeltaRow {
  mode: string;
  metric: string;
  delta: number;
}

function computeDeltas(a: unknown, b: unknown): DeltaRow[] {
  const aggA = aggregatesOf(a);
  const aggB = aggregatesOf(b);
  const rows: DeltaRow[] = [];
  for (const mode of Object.keys(aggB)) {
    if (!(mode in aggA)) continue;
    const ma = aggA[mode];
    const mb = aggB[mode];
    for (const metric of ["hit@1", "recall@5", "mrr", "ndcg@5"]) {
      rows.push({
        mode,
        metric,
        delta: round3(metricValue(mb, metric) - metricValue(ma, metric)),
      });
    }
  }
  return rows;
}

type AggMap = Record<
  string,
  { chunk_metrics: EvalLevelMetrics } & Record<string, unknown>
>;

function aggregatesOf(run: unknown): AggMap {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const aggs = (run as any).aggregates as Array<Record<string, any>>;
  return Object.fromEntries(aggs.map((x) => [x.mode, x]));
}

function metricValue(
  m: { chunk_metrics: EvalLevelMetrics },
  metric: string,
): number {
  const cm = m.chunk_metrics;
  switch (metric) {
    case "hit@1":
      return cm.hit_at_k["1"] ?? 0;
    case "recall@5":
      return cm.recall_at_k["5"] ?? 0;
    case "ndcg@5":
      return cm.ndcg_at_k["5"] ?? 0;
    default:
      return cm.mrr;
  }
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

// ----------------------------------------------------------------------
// FAILURES

function Failures({ onReady }: { onReady: (e: string | null) => void }) {
  const { traceToSource, state } = useWorkspace();
  const [runs, setRuns] = useState<EvalRunListItem[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [detail, setDetail] = useState<EvalRunDetail | null>(null);
  const [category, setCategory] = useState<string>("ALL");
  const [error, setError] = useState<string | null>(null);
  const docId = state.documentId;
  const [chunkIndex, setChunkIndex] = useState<
    Record<string, { page_numbers: number[]; source_block_ids: string[]; chunk_id: string }>
  >({});

  useEffect(() => {
    api
      .evaluationRuns()
      .then((rs) => {
        setRuns(rs);
        if (rs.length > 0) setRunId(rs[rs.length - 1].run_id);
      })
      .catch((e: Error) => onReady(e.message));
  }, [onReady]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    api
      .evaluationRun(runId)
      .then((d) => !cancelled && setDetail(d))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [runId]);

  // Load the selected document's chunks once so failure cases can trace
  // ground-truth chunks to their source pages/blocks.
  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    api
      .chunks(docId)
      .then((chunks) => {
        if (cancelled) return;
        const idx: typeof chunkIndex = {};
        for (const c of chunks) {
          idx[c.chunk_id] = {
            page_numbers: c.page_numbers,
            source_block_ids: c.source_block_ids,
            chunk_id: c.chunk_id,
          };
        }
        setChunkIndex(idx);
      })
      .catch(() => {
        /* tracing stays disabled; failures still viewable */
      });
    return () => {
      cancelled = true;
    };
  }, [docId]);

  if (error)
    return <div role="alert" className={styles.error}>{error}</div>;
  if (!detail)
    return <div role="status" className={styles.loading}>loading failures …</div>;

  const cats = ["ALL", ...new Set(detail.failures.map((f) => f.category))];
  const shown: EvalFailureCase[] =
    category === "ALL"
      ? detail.failures
      : detail.failures.filter((f) => f.category === category);

  return (
    <section aria-label="failure inspector">
      <div className={styles.selectorRow}>
        <label>
          run{" "}
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id.slice(0, 20)}
              </option>
            ))}
          </select>
        </label>
        <label>
          category{" "}
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            {cats.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <span>{shown.length} cases</span>
      </div>

      <ul className={styles.failureList} data-testid="failure-list">
        {shown.map((f) => (
          <li key={`${f.category}-${f.question_id}`} className={styles.failureCard}>
            <p className={styles.failCat}>{f.category}</p>
            <h4 className={styles.failQ}>{f.question}</h4>
            <p className={styles.failMeta}>
              ground truth:{" "}
              {(f.detail.ground_truth_chunk_ids ?? []).map((cid) => (
                <code key={cid}>{short(cid)} </code>
              ))}
            </p>
            <ExtraDetail detail={f.detail} />
            <PerModeRanks rows={perMode(detail, f.question_id)} />
            <div className={styles.traceRow}>
              {(f.detail.ground_truth_chunk_ids ?? []).map((cid) => {
                const chunk = chunkIndex[cid] ?? null;
                return (
                  <button
                    key={cid}
                    type="button"
                    data-testid={`trace-fail-${f.question_id}-${short(cid)}`}
                    disabled={!chunk}
                    onClick={() =>
                      chunk &&
                      traceToSource({
                        pageNumbers: chunk.page_numbers,
                        blockIds: chunk.source_block_ids,
                        chunkId: chunk.chunk_id,
                      })
                    }
                  >
                    trace {short(cid)} → source
                  </button>
                );
              })}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function short(chunkId: string): string {
  const i = chunkId.lastIndexOf("-c");
  return i >= 0 ? chunkId.slice(i + 1) : chunkId;
}

function ExtraDetail({ detail }: { detail: Record<string, unknown> }) {
  const extra = Object.entries(detail).filter(
    ([k]) => k !== "ground_truth_chunk_ids" && k !== "relevant_pages",
  );
  if (extra.length === 0) return null;
  return (
    <p className={styles.failMeta}>
      {extra.map(([k, v]) => (
        <span key={k}>
          {k}: <code>{String(v)}</code>{" "}
        </span>
      ))}
    </p>
  );
}

interface PerQueryLite {
  question_id: string;
  mode: string;
  retrieved_chunk_ids: string[];
  first_relevant_rank_chunk: number | null;
}

function perMode(detail: EvalRunDetail, questionId: string): PerQueryLite[] {
  return Object.entries(detail.per_query).map(([mode, results]) => {
    const r = results.find((x) => x.question_id === questionId);
    return {
      question_id: questionId,
      mode,
      retrieved_chunk_ids: r?.retrieved_chunk_ids ?? [],
      first_relevant_rank_chunk: r?.first_relevant_rank_chunk ?? null,
    };
  });
}

function PerModeRanks({ rows }: { rows: PerQueryLite[] }) {
  return (
    <table className={styles.rankTable}>
      <tbody>
        {rows.map((r) => (
          <tr key={r.mode} data-testid={`rank-${r.question_id}-${r.mode}`}>
            <th scope="row">{MODE_LABELS[r.mode] ?? r.mode}</th>
            <td>
              {r.first_relevant_rank_chunk
                ? `hit at #${r.first_relevant_rank_chunk}`
                : "not retrieved"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
