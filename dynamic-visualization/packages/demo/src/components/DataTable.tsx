import React, { useState, useMemo, useEffect } from "react";
import { useSelection, useChartData } from "chart-lib";
import type { ChartDataRow } from "chart-lib";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Domain-specific primary columns shown first in the table
const PRIORITY_COLS = [
  "prot_id",
  "central_check_id",
  "src_sys_nm",
  "orientation",
  "status",
  "priority",
  "type",
  "check_id",
  "issue_action",
  "upd_sys_nm",
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface DataTableProps {
  dataUrl: string;
  pageSize?: number;
}

// ---------------------------------------------------------------------------
// DataTable
// Renders tabular data from the shared chart-lib cache.
// - Priority columns (filter fields + common chart fields) appear first.
// - Clicking any cell sets a cross-filter selection via SelectionContext.
// - Rows matching the active selection are highlighted.
// - When a chart selection fires, the table scrolls to the first match.
// ---------------------------------------------------------------------------

export const DataTable: React.FC<DataTableProps> = ({
  dataUrl,
  pageSize = 50,
}) => {
  const { selection, clearSelection, setSelectionByValues } = useSelection();
  const { data, loading, error } = useChartData(dataUrl);
  const [page, setPage] = useState(0);

  // When an external selection changes (from a chart click), jump to the page
  // that contains the first matching row so the user can see highlighted rows.
  useEffect(() => {
    if (!selection || data.length === 0) return;
    const idx = data.findIndex((row) =>
      selection.selectedValues.some(
        (sv) => String(sv) === String(row[selection.field]),
      ),
    );
    if (idx >= 0) {
      setPage(Math.floor(idx / pageSize));
    }
  }, [selection, data, pageSize]);

  // Derive an ordered column list: priority cols first, then remaining cols
  const columns = useMemo(() => {
    if (data.length === 0) return PRIORITY_COLS;
    const allCols = Object.keys(data[0]!).filter(
      (k) => !k.startsWith("_") && k !== "count",
    );
    const priority = PRIORITY_COLS.filter((c) => allCols.includes(c));
    const rest = allCols.filter((c) => !PRIORITY_COLS.includes(c));
    return [...priority, ...rest];
  }, [data]);

  // Slice current page
  const visibleRows = useMemo(() => {
    const start = page * pageSize;
    return data.slice(start, start + pageSize);
  }, [data, page, pageSize]);

  const totalPages = Math.max(1, Math.ceil(data.length / pageSize));
  const hasFilter = selection !== null;

  // Count rows in the full dataset that match the current selection
  const matchingCount = useMemo(() => {
    if (!selection) return 0;
    return data.filter((row) =>
      selection.selectedValues.some(
        (sv) => String(sv) === String(row[selection.field]),
      ),
    ).length;
  }, [selection, data]);

  const isRowHighlighted = (row: ChartDataRow): boolean => {
    if (!selection) return false;
    return selection.selectedValues.some(
      (sv) => String(sv) === String(row[selection.field]),
    );
  };

  // Clicking a cell cross-filters on that column + value
  const handleCellClick = (row: ChartDataRow, field: string) => {
    const val = row[field];
    if (val === null || val === undefined || val === "") return;
    setSelectionByValues("data-table", field, [val as string | number]);
  };

  return (
    <div style={tbl.root}>
      {/* ── Header ── */}
      <div style={tbl.header}>
        <div style={tbl.headerLeft}>
          <h2 style={tbl.title}>Dataset</h2>
          <span style={tbl.count}>{data.length.toLocaleString()} rows</span>
          {hasFilter && (
            <span style={tbl.filterBadge}>
              ●&nbsp;{matchingCount.toLocaleString()} match&nbsp;
              <code style={tbl.filterCode}>
                {selection!.field} ∈ [{selection!.selectedValues.join(", ")}]
              </code>
            </span>
          )}
        </div>

        <div style={tbl.headerRight}>
          <button
            onClick={clearSelection}
            disabled={!hasFilter}
            style={{
              ...tbl.resetBtn,
              ...(hasFilter ? tbl.resetBtnActive : {}),
            }}
            title="Clear cross-filter selection"
          >
            <span style={{ fontSize: 14 }}>↺</span> Reset
            {hasFilter && <span style={tbl.dot} />}
          </button>
        </div>
      </div>

      {/* ── Loading / error ── */}
      {loading && <div style={tbl.loadingBar}>Loading data…</div>}
      {error && !loading && (
        <div style={tbl.errorMsg}>⚠ {error.message}</div>
      )}

      {/* ── Table ── */}
      {!loading && data.length > 0 && (
        <>
          <div style={tbl.hint}>
            Click any cell to cross-filter all charts by that column&nbsp;+&nbsp;value.
          </div>

          <div style={tbl.tableWrapper}>
            <table style={tbl.table}>
              <thead>
                <tr>
                  {columns.map((col) => (
                    <th
                      key={col}
                      style={{
                        ...tbl.th,
                        ...(selection?.field === col ? tbl.thActive : {}),
                      }}
                      title={`Click a cell to cross-filter by ${col}`}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row, i) => {
                  const hl = isRowHighlighted(row);
                  return (
                    <tr
                      key={i}
                      style={{
                        ...tbl.tr,
                        ...(hl
                          ? tbl.trHighlighted
                          : i % 2 === 0
                            ? {}
                            : tbl.trOdd),
                      }}
                    >
                      {columns.map((col) => (
                        <td
                          key={col}
                          style={{
                            ...tbl.td,
                            ...(selection?.field === col
                              ? tbl.tdActiveField
                              : {}),
                          }}
                          onClick={() => handleCellClick(row, col)}
                          title={`Filter by ${col} = ${row[col] ?? "—"}`}
                        >
                          {row[col] !== null && row[col] !== undefined
                            ? String(row[col])
                            : "—"}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ── Pagination ── */}
          <div style={tbl.pagination}>
            <button
              onClick={() => setPage(0)}
              disabled={page === 0}
              style={tbl.pageBtn}
              title="First page"
            >
              ⟪
            </button>
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              style={tbl.pageBtn}
              title="Previous page"
            >
              ‹
            </button>

            <span style={tbl.pageInfo}>
              Page&nbsp;{page + 1}&nbsp;/&nbsp;{totalPages} &nbsp;·&nbsp;
              rows&nbsp;{(page * pageSize + 1).toLocaleString()}–
              {Math.min((page + 1) * pageSize, data.length).toLocaleString()}
              &nbsp;of&nbsp;{data.length.toLocaleString()}
            </span>

            <button
              onClick={() =>
                setPage((p) => Math.min(totalPages - 1, p + 1))
              }
              disabled={page >= totalPages - 1}
              style={tbl.pageBtn}
              title="Next page"
            >
              ›
            </button>
            <button
              onClick={() => setPage(totalPages - 1)}
              disabled={page >= totalPages - 1}
              style={tbl.pageBtn}
              title="Last page"
            >
              ⟫
            </button>
          </div>
        </>
      )}

      {/* Empty state */}
      {!loading && !error && data.length === 0 && (
        <div style={tbl.emptyState}>No data available.</div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const tbl = {
  root: {
    background: "#fff",
    borderRadius: 10,
    boxShadow: "0 1px 6px rgba(0,0,0,.08)",
    marginTop: 20,
    display: "flex",
    flexDirection: "column" as const,
  } satisfies React.CSSProperties,

  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 18px 10px",
    borderBottom: "1px solid #f0f2f5",
    gap: 12,
    flexWrap: "wrap" as const,
  } satisfies React.CSSProperties,

  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    flexWrap: "wrap" as const,
  } satisfies React.CSSProperties,

  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  } satisfies React.CSSProperties,

  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 700,
    color: "#1a1a2e",
  } satisfies React.CSSProperties,

  count: {
    fontSize: 12,
    color: "#888",
    background: "#f5f5f5",
    borderRadius: 4,
    padding: "2px 7px",
  } satisfies React.CSSProperties,

  filterBadge: {
    fontSize: 12,
    fontWeight: 600,
    color: "#e67e22",
    background: "#fef9f2",
    border: "1px solid #f5cba7",
    borderRadius: 4,
    padding: "2px 8px",
    display: "flex",
    alignItems: "center",
    gap: 4,
  } satisfies React.CSSProperties,

  filterCode: {
    fontFamily: "monospace",
    fontSize: 11,
    color: "#c0392b",
  } satisfies React.CSSProperties,

  resetBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    padding: "6px 14px",
    fontSize: 12,
    fontWeight: 700,
    color: "#ccc",
    background: "#fff",
    border: "1.5px solid #eee",
    borderRadius: 6,
    cursor: "default",
    outline: "none",
    letterSpacing: ".02em",
  } satisfies React.CSSProperties,

  resetBtnActive: {
    color: "#e67e22",
    borderColor: "#e67e22",
    background: "#fef9f2",
    cursor: "pointer",
  } satisfies React.CSSProperties,

  dot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#e67e22",
    display: "inline-block",
  } satisfies React.CSSProperties,

  hint: {
    fontSize: 11,
    color: "#bbb",
    padding: "5px 18px",
    background: "#fafafa",
    borderBottom: "1px solid #f0f2f5",
    flexShrink: 0,
  } satisfies React.CSSProperties,

  loadingBar: {
    padding: "24px",
    textAlign: "center" as const,
    fontSize: 13,
    color: "#888",
  } satisfies React.CSSProperties,

  errorMsg: {
    padding: "16px 18px",
    color: "#e74c3c",
    fontSize: 13,
    background: "#fdf2f2",
    borderTop: "1px solid #f5c6c6",
  } satisfies React.CSSProperties,

  emptyState: {
    padding: "32px",
    textAlign: "center" as const,
    fontSize: 13,
    color: "#bbb",
  } satisfies React.CSSProperties,

  tableWrapper: {
    overflowX: "auto" as const,
    overflowY: "auto" as const,
    maxHeight: 420,
    flex: 1,
    borderBottom: "1px solid #f0f2f5",
  } satisfies React.CSSProperties,

  table: {
    borderCollapse: "collapse" as const,
    width: "100%",
    fontSize: 13,
    tableLayout: "auto" as const,
  } satisfies React.CSSProperties,

  th: {
    position: "sticky" as const,
    top: 0,
    zIndex: 2,
    background: "#f0f2f5",
    padding: "9px 14px",
    textAlign: "left" as const,
    fontWeight: 700,
    fontSize: 11,
    color: "#444",
    borderBottom: "2px solid #d8dbe4",
    whiteSpace: "nowrap" as const,
    letterSpacing: ".03em",
    textTransform: "uppercase" as const,
    userSelect: "none" as const,
    boxShadow: "0 1px 0 #d8dbe4",
  } satisfies React.CSSProperties,

  thActive: {
    background: "#fef3e2",
    color: "#e67e22",
    borderBottomColor: "#e67e22",
    boxShadow: "0 1px 0 #e67e22",
  } satisfies React.CSSProperties,

  tr: {
    transition: "background 0.1s",
  } satisfies React.CSSProperties,

  trOdd: {
    background: "#fafafa",
  } satisfies React.CSSProperties,

  trHighlighted: {
    background: "#fff8ee",
    boxShadow: "inset 3px 0 0 #e67e22",
  } satisfies React.CSSProperties,

  td: {
    padding: "8px 14px",
    borderBottom: "1px solid #f0f2f5",
    color: "#2c3e50",
    whiteSpace: "nowrap" as const,
    cursor: "pointer",
    transition: "background 0.08s",
    fontSize: 13,
    lineHeight: 1.4,
  } satisfies React.CSSProperties,

  tdActiveField: {
    background: "rgba(230, 126, 34, 0.06)",
    fontWeight: 600,
    color: "#c0392b",
  } satisfies React.CSSProperties,

  pagination: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    padding: "10px 18px",
    background: "#fafafa",
    borderRadius: "0 0 10px 10px",
    flexShrink: 0,
  } satisfies React.CSSProperties,

  pageBtn: {
    padding: "4px 10px",
    fontSize: 13,
    fontWeight: 700,
    background: "#fff",
    border: "1px solid #ddd",
    borderRadius: 5,
    cursor: "pointer",
    color: "#444",
    outline: "none",
    transition: "all 0.12s",
  } satisfies React.CSSProperties,

  pageInfo: {
    fontSize: 12,
    color: "#666",
    minWidth: 220,
    textAlign: "center" as const,
  } satisfies React.CSSProperties,
} as const;
