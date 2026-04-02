import type { Data, Layout } from "plotly.js";
import type { ChartConfig } from "../types/ChartConfig";
import type { ChartDataRow } from "../types/ChartData";
import {
  PALETTE,
  aggregate,
  groupByField,
  col,
  buildColorArray,
  applyAggregation,
  buildSankeyData,
} from "../utils/dataUtils";

// ---------------------------------------------------------------------------
// buildTraces — converts ChartConfig + data rows into Plotly Data[]
// ---------------------------------------------------------------------------

export function buildTraces(
  config: ChartConfig,
  rawData: ChartDataRow[],
): Data[] {
  const { type, yAxis, zAxis, marker, groupBy, colorBy, stackGroup } = config;

  // Sankey does its own two-field aggregation — skip the standard pre-aggregation
  if (type === "sankey") return buildSankeyTraces(rawData, config);

  // Pre-aggregate when both groupBy and yAxis are set
  const rows = applyAggregation(rawData, config);

  switch (type) {
    case "pie":
      return buildPieTraces(rows, config);

    case "heatmap":
      return buildHeatmapTraces(rows, config, zAxis?.field);

    case "histogram":
      return buildHistogramTraces(rows, config);

    case "box":
      return buildBoxTraces(rows, config);

    default:
      // line, scatter, bar, area
      return buildXYTraces(rows, config, {
        ...(stackGroup !== undefined ? { stackGroup } : {}),
        ...(colorBy !== undefined ? { colorBy } : {}),
        ...(marker !== undefined ? { marker } : {}),
      });
  }
}

// ---------------------------------------------------------------------------
// Sankey trace
// ---------------------------------------------------------------------------

function buildSankeyTraces(rows: ChartDataRow[], config: ChartConfig): Data[] {
  const { xAxis, yAxis, sankeyTarget, aggregation } = config;
  if (!sankeyTarget || !yAxis) return [];

  const { nodes, links } = buildSankeyData(
    rows,
    xAxis.field,
    sankeyTarget,
    yAxis.field,
    aggregation ?? "sum",
  );

  const nodeColors = nodes.map(
    (_, i) => PALETTE[i % PALETTE.length] ?? "#636EFA",
  );

  // Build per-link colours derived from the source node colour with reduced opacity
  const linkColors = links.map((l) => {
    const hex = nodeColors[l.source] ?? "#636EFA";
    return `${hex}55`; // ~33% opacity suffix
  });

  return [
    {
      type: "sankey",
      orientation: "h",
      arrangement: "snap",
      node: {
        pad: 16,
        thickness: 22,
        line: { color: "#ccc", width: 0.5 },
        label: nodes.map((n) => n.name),
        color: nodeColors,
        hovertemplate: "%{label}<br>Total: %{value}<extra></extra>",
      },
      link: {
        source: links.map((l) => l.source),
        target: links.map((l) => l.target),
        value: links.map((l) => l.value),
        color: linkColors,
        hovertemplate:
          "%{source.label} → %{target.label}<br>Value: %{value}<extra></extra>",
      },
    } as unknown as Data,
  ];
}

// ---------------------------------------------------------------------------
// XY-type traces: line, scatter, bar, area
// ---------------------------------------------------------------------------

function buildXYTraces(
  rows: ChartDataRow[],
  config: ChartConfig,
  opts: {
    stackGroup?: string;
    colorBy?: string;
    marker?: ChartConfig["marker"];
  },
): Data[] {
  const { type, xAxis, yAxis, marker } = config;

  const plotlyType = type === "bar" ? "bar" : "scatter";
  const mode =
    type === "line" || type === "area"
      ? "lines"
      : type === "scatter"
        ? "markers"
        : undefined;

  const markerColor = marker?.colorField
    ? buildColorArray(rows, marker.colorField, PALETTE)
    : undefined;

  const markerSize = marker?.sizeField
    ? (col(rows, marker.sizeField) as number[])
    : marker?.size;

  // Split into categorical traces if colorBy or groupBy is set
  const splitField = opts.colorBy ?? config.groupBy;
  if (splitField) {
    const groups = groupByField(rows, splitField);
    return Array.from(groups.entries()).map(([name, groupRows], idx) => {
      const traceColor = PALETTE[idx % PALETTE.length] ?? "#636EFA";
      const base: Record<string, unknown> = {
        type: plotlyType,
        name,
        x: col(groupRows, xAxis.field),
        y: yAxis ? col(groupRows, yAxis.field) : undefined,
        marker: {
          color: traceColor,
          ...(marker?.size !== undefined ? { size: marker.size } : {}),
          ...(marker?.opacity !== undefined ? { opacity: marker.opacity } : {}),
        },
      };
      if (mode) base["mode"] = mode;
      if (type === "area") {
        base["fill"] = "tonexty";
        base["stackgroup"] = opts.stackGroup ?? "one";
      }
      return base as unknown as Data;
    });
  }

  // Single trace
  const trace: Record<string, unknown> = {
    type: plotlyType,
    x: col(rows, xAxis.field),
    y: yAxis ? col(rows, yAxis.field) : undefined,
    marker: {
      ...(markerColor !== undefined ? { color: markerColor } : {}),
      ...(markerSize !== undefined ? { size: markerSize } : {}),
      ...(marker?.opacity !== undefined ? { opacity: marker.opacity } : {}),
    },
  };
  if (mode) trace["mode"] = mode;
  if (type === "area") {
    trace["fill"] = "tozeroy";
    if (opts.stackGroup) trace["stackgroup"] = opts.stackGroup;
  }
  return [trace as unknown as Data];
}

// ---------------------------------------------------------------------------
// Pie trace
// ---------------------------------------------------------------------------

function buildPieTraces(rows: ChartDataRow[], config: ChartConfig): Data[] {
  const { xAxis, yAxis } = config;
  const labels = col(rows, xAxis.field) as string[];
  const values = yAxis ? (col(rows, yAxis.field) as number[]) : [];
  return [
    {
      type: "pie",
      labels,
      values,
      hole: 0.4,
      marker: { colors: PALETTE },
    } as unknown as Data,
  ];
}

// ---------------------------------------------------------------------------
// Heatmap trace
// ---------------------------------------------------------------------------

function buildHeatmapTraces(
  rows: ChartDataRow[],
  config: ChartConfig,
  zField: string | undefined,
): Data[] {
  const { xAxis, yAxis, marker } = config;
  if (!yAxis || !zField) return [];

  const xVals = Array.from(new Set(col(rows, xAxis.field) as string[]));
  const yVals = Array.from(new Set(col(rows, yAxis.field) as string[]));

  const z: number[][] = yVals.map(() => xVals.map(() => 0));
  for (const row of rows) {
    const xi = xVals.indexOf(String(row[xAxis.field] ?? ""));
    const yi = yVals.indexOf(String(row[yAxis.field] ?? ""));
    if (xi !== -1 && yi !== -1) {
      const zRow = z[yi];
      if (zRow) {
        const incoming = Number(row[zField] ?? 0);
        zRow[xi] = (zRow[xi] ?? 0) + (isNaN(incoming) ? 0 : incoming);
      }
    }
  }

  return [
    {
      type: "heatmap",
      x: xVals,
      y: yVals,
      z,
      colorscale: marker?.colorScale ?? "Viridis",
    } as unknown as Data,
  ];
}

// ---------------------------------------------------------------------------
// Histogram trace
// ---------------------------------------------------------------------------

function buildHistogramTraces(
  rows: ChartDataRow[],
  config: ChartConfig,
): Data[] {
  const { xAxis, colorBy } = config;

  if (colorBy) {
    const groups = groupByField(rows, colorBy);
    return Array.from(groups.entries()).map(
      ([name, groupRows], idx) =>
        ({
          type: "histogram",
          name,
          x: col(groupRows, xAxis.field) as (string | number)[],
          marker: {
            color: PALETTE[idx % PALETTE.length] ?? "#636EFA",
            opacity: config.marker?.opacity ?? 0.75,
          },
        }) as unknown as Data,
    );
  }

  return [
    {
      type: "histogram",
      x: col(rows, xAxis.field) as (string | number)[],
      marker: { color: PALETTE[0], opacity: config.marker?.opacity ?? 0.75 },
    } as unknown as Data,
  ];
}

// ---------------------------------------------------------------------------
// Box plot traces
// ---------------------------------------------------------------------------

function buildBoxTraces(rows: ChartDataRow[], config: ChartConfig): Data[] {
  const { xAxis, yAxis, colorBy } = config;

  const splitField = colorBy ?? (yAxis ? undefined : xAxis.field);

  if (splitField) {
    const groups = groupByField(rows, splitField);
    return Array.from(groups.entries()).map(
      ([name, groupRows], idx) =>
        ({
          type: "box",
          name,
          y: col(groupRows, yAxis?.field ?? xAxis.field) as number[],
          boxmean: true,
          marker: { color: PALETTE[idx % PALETTE.length] ?? "#636EFA" },
        }) as unknown as Data,
    );
  }

  return [
    {
      type: "box",
      x: xAxis ? (col(rows, xAxis.field) as string[]) : undefined,
      y: yAxis ? (col(rows, yAxis.field) as number[]) : undefined,
      boxmean: true,
      boxpoints: "outliers" as const,
    } as unknown as Data,
  ];
}

// ---------------------------------------------------------------------------
// buildLayout — merges axis config, selection settings, and layout overrides
// ---------------------------------------------------------------------------

export function buildLayout(config: ChartConfig): Partial<Layout> {
  const {
    title,
    xAxis,
    yAxis,
    selectionMode,
    enableSelection,
    type,
    layout: overrides,
  } = config;

  // Sankey charts don't use traditional Cartesian axes or drag-select
  if (type === "sankey") {
    return {
      ...(title !== undefined ? { title: { text: title } } : {}),
      autosize: true,
      hovermode: "closest",
      font: { family: "'Inter', system-ui, sans-serif" },
      ...(overrides as Partial<Layout> | undefined),
    } as Partial<Layout>;
  }

  const xaxisType =
    xAxis.axisType ?? (type === "histogram" ? "linear" : undefined);

  const base = {
    ...(title !== undefined ? { title: { text: title } } : {}),
    autosize: true,
    hovermode: "closest",
    dragmode: enableSelection
      ? selectionMode === "lasso"
        ? "lasso"
        : "select"
      : "zoom",
    clickmode: "event+select",
    xaxis: {
      ...(xAxis.title !== undefined ? { title: { text: xAxis.title } } : {}),
      ...(xaxisType !== undefined ? { type: xaxisType } : {}),
      automargin: true,
    },
    ...(yAxis !== undefined
      ? {
          yaxis: {
            ...(yAxis.title !== undefined
              ? { title: { text: yAxis.title } }
              : {}),
            ...(yAxis.axisType !== undefined ? { type: yAxis.axisType } : {}),
            automargin: true,
          },
        }
      : {}),
    ...(type === "bar" && config.groupBy ? { barmode: "group" } : {}),
  };

  return {
    ...base,
    ...(overrides as Partial<Layout> | undefined),
  } as Partial<Layout>;
}
