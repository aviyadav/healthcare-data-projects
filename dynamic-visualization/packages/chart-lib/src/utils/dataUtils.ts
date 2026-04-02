import type { ChartConfig } from "../types/ChartConfig";
import type { ChartDataRow } from "../types/ChartData";

// ---------------------------------------------------------------------------
// Shared colour palette — used by all renderers for consistent colours
// ---------------------------------------------------------------------------

export const PALETTE = [
  "#636EFA",
  "#EF553B",
  "#00CC96",
  "#AB63FA",
  "#FFA15A",
  "#19D3F3",
  "#FF6692",
  "#B6E880",
  "#FF97FF",
  "#FECB52",
];

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

type AggFn = (values: number[]) => number;

const AGG_FNS: Record<NonNullable<ChartConfig["aggregation"]>, AggFn> = {
  sum: (vals) => vals.reduce((a, b) => a + b, 0),
  count: (vals) => vals.length,
  mean: (vals) => vals.reduce((a, b) => a + b, 0) / vals.length,
  median: (vals) => {
    const sorted = [...vals].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
      ? (sorted[mid] ?? 0)
      : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
  },
};

/**
 * Pre-aggregate rows by grouping on `groupField` and reducing `valueField`
 * via the named aggregation function.
 *
 * The returned rows each have exactly two fields:
 *   { [groupField]: string, [valueField]: number }
 */
export function aggregate(
  rows: ChartDataRow[],
  groupField: string,
  valueField: string,
  aggName: NonNullable<ChartConfig["aggregation"]>,
): ChartDataRow[] {
  const groups = new Map<string, { vals: number[]; rowCount: number }>();

  for (const row of rows) {
    const key = String(row[groupField] ?? "");
    const val = Number(row[valueField]);
    const entry = groups.get(key);
    if (entry) {
      entry.rowCount++;
      if (!isNaN(val)) entry.vals.push(val);
    } else {
      groups.set(key, { vals: isNaN(val) ? [] : [val], rowCount: 1 });
    }
  }

  return Array.from(groups.entries()).map(([key, { vals, rowCount }]) => ({
    [groupField]: key,
    [valueField]:
      aggName === "count"
        ? rowCount
        : AGG_FNS[aggName](vals.length > 0 ? vals : [0]),
  }));
}

// ---------------------------------------------------------------------------
// Split rows by a categorical field into keyed groups
// ---------------------------------------------------------------------------

export function groupByField(
  rows: ChartDataRow[],
  field: string,
): Map<string, ChartDataRow[]> {
  const map = new Map<string, ChartDataRow[]>();
  for (const row of rows) {
    const key = String(row[field] ?? "Other");
    const existing = map.get(key);
    if (existing) {
      existing.push(row);
    } else {
      map.set(key, [row]);
    }
  }
  return map;
}

// ---------------------------------------------------------------------------
// Extract a single column from a row array as a typed value array
// ---------------------------------------------------------------------------

export function col(
  rows: ChartDataRow[],
  field: string,
): (string | number | null)[] {
  return rows.map((r) => r[field] ?? null);
}

// ---------------------------------------------------------------------------
// Build a colour-per-row array from a categorical field
// ---------------------------------------------------------------------------

export function buildColorArray(
  rows: ChartDataRow[],
  colorField: string,
  palette: string[],
): string[] {
  const categories = Array.from(
    new Set(rows.map((r) => String(r[colorField] ?? ""))),
  );
  const colorMap = new Map(
    categories.map((c, i) => [c, palette[i % palette.length] ?? "#636EFA"]),
  );
  return rows.map(
    (r) => colorMap.get(String(r[colorField] ?? "")) ?? "#636EFA",
  );
}

// ---------------------------------------------------------------------------
// Apply pre-aggregation from a ChartConfig if `aggregation` and `groupBy`
// are both set — convenience wrapper used by every renderer.
// ---------------------------------------------------------------------------

export function applyAggregation(
  rows: ChartDataRow[],
  config: ChartConfig,
): ChartDataRow[] {
  const { aggregation, groupBy, yAxis } = config;
  if (aggregation && groupBy && yAxis) {
    return aggregate(rows, groupBy, yAxis.field, aggregation);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Sankey data helpers
// ---------------------------------------------------------------------------

/** A single node in a Sankey diagram. */
export interface SankeyNodeData {
  name: string;
}

/** A single directed link (flow) in a Sankey diagram. */
export interface SankeyLinkData {
  /** Zero-based index into the nodes array for the link source. */
  source: number;
  /** Zero-based index into the nodes array for the link target. */
  target: number;
  /** Aggregated flow weight. */
  value: number;
}

/** Fully-prepared node + link graph ready to hand to a Sankey renderer. */
export interface SankeyData {
  nodes: SankeyNodeData[];
  links: SankeyLinkData[];
}

/**
 * Build a Sankey-ready `{nodes, links}` graph from flat data rows.
 *
 * Rows are grouped by the composite key `(sourceField, targetField)` and the
 * `valueField` is reduced by `aggName` within each group.  Nodes are deduplicated
 * across both source and target values; the final node order is
 * [unique-sources…, target-only-nodes…] so sources always appear on the left.
 *
 * @param rows        Flat data rows (pre-filtered but NOT pre-aggregated).
 * @param sourceField Column whose values become source-node labels.
 * @param targetField Column whose values become target-node labels.
 * @param valueField  Column used as the numeric flow weight.
 * @param aggName     How to reduce multiple `valueField` values within a group.
 */
export function buildSankeyData(
  rows: ChartDataRow[],
  sourceField: string,
  targetField: string,
  valueField: string,
  aggName: NonNullable<ChartConfig["aggregation"]> = "sum",
): SankeyData {
  // ── 1. Group rows by (source, target) ──────────────────────────────────
  const groupMap = new Map<string, { vals: number[]; count: number }>();

  for (const row of rows) {
    const src = String(row[sourceField] ?? "");
    const tgt = String(row[targetField] ?? "");
    if (!src || !tgt) continue;

    const val = Number(row[valueField] ?? 0);
    const key = `${src}\0${tgt}`;
    const entry = groupMap.get(key);
    if (entry) {
      entry.count++;
      if (!isNaN(val)) entry.vals.push(val);
    } else {
      groupMap.set(key, { vals: isNaN(val) ? [] : [val], count: 1 });
    }
  }

  // ── 2. Collect unique node names (sources first, then target-only) ──────
  const sourceNames = new Set<string>();
  const targetNames = new Set<string>();

  for (const key of groupMap.keys()) {
    const nul = key.indexOf("\0");
    sourceNames.add(key.slice(0, nul));
    targetNames.add(key.slice(nul + 1));
  }

  // Sources come first so they naturally sit on the left column.
  const allNames = [
    ...Array.from(sourceNames),
    ...Array.from(targetNames).filter((n) => !sourceNames.has(n)),
  ];
  const nodeIndex = new Map(allNames.map((n, i) => [n, i]));

  // ── 3. Build link objects ───────────────────────────────────────────────
  const aggFn = AGG_FNS[aggName];
  const links: SankeyLinkData[] = [];

  for (const [key, { vals, count }] of groupMap.entries()) {
    const nul = key.indexOf("\0");
    const src = key.slice(0, nul);
    const tgt = key.slice(nul + 1);
    const value =
      aggName === "count" ? count : aggFn(vals.length > 0 ? vals : [0]);

    links.push({
      source: nodeIndex.get(src) ?? 0,
      target: nodeIndex.get(tgt) ?? 0,
      value,
    });
  }

  return {
    nodes: allNames.map((name) => ({ name })),
    links,
  };
}
