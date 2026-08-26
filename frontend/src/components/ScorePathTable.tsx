"use client";

import type { RetrievalResult } from "@/lib/types";
import styles from "./ScorePathTable.module.css";

function fmt(v: number | null): string {
  return v === null ? "—" : String(v);
}

export default function ScorePathTable({ result: r }: { result: RetrievalResult }) {
  const rows: Array<[string, number | null, number | null]> = [
    ["dense", r.dense_rank, r.dense_score],
    ["bm25", r.bm25_rank, r.bm25_score],
    ["fusion (RRF)", r.fusion_rank, r.fusion_score],
    ["reranker (cross-encoder logit)", r.reranker_rank, r.reranker_score],
  ];
  return (
    <table className={styles.table} data-testid={`scores-${r.chunk_id}`}>
      <caption className="srOnly">score path for {r.chunk_id}</caption>
      <thead>
        <tr>
          <th scope="col">stage</th>
          <th scope="col">rank</th>
          <th scope="col">score</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([stage, rank, score]) => (
          <tr key={stage} data-testid={`score-row-${stage.split(" ")[0]}`}>
            <th scope="row">{stage}</th>
            <td>{rank === null ? "—" : `#${rank}`}</td>
            <td className={styles.num}>{fmt(score)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
