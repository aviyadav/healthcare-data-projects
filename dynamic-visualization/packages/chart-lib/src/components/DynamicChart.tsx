import React, { useMemo } from "react";
import type { PlotMouseEvent, PlotSelectionEvent } from "plotly.js";

import type { ChartConfig } from "../types/ChartConfig";
import type { ChartDataRow } from "../types/ChartData";
import { useChartConfig } from "../hooks/useChartConfig";
import { useChartData } from "../hooks/useChartData";
import { useSelection } from "../context/SelectionContext";
import { useRenderer } from "../context/RendererContext";
import { PlotlyRenderer } from "../renderers/PlotlyRenderer";

import { D3Renderer } from "../renderers/D3Renderer";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface DynamicChartProps {
  /**
   * Chart configuration — either a full `ChartConfig` object or a URL / path
   * to a JSON file that will be fetched and validated at runtime.
   */
  config: ChartConfig | string;

  /**
   * Chart data — either an inline `ChartDataRow[]` or a URL / path to a JSON
   * file. Cross-filtering is applied automatically before handing data to the
   * active renderer.
   */
  data: ChartDataRow[] | string;

  /**
   * Plotly-specific: called when the user clicks a data point.
   * Only fires when the active renderer is 'plotly'.
   */
  onChartClick?: (event: PlotMouseEvent, config: ChartConfig) => void;

  /**
   * Plotly-specific: called when the user completes a box/lasso selection.
   * Only fires when the active renderer is 'plotly'.
   */
  onChartSelect?: (event: PlotSelectionEvent, config: ChartConfig) => void;

  style?: React.CSSProperties;
  className?: string;
}

// ---------------------------------------------------------------------------
// Loading / error states
// ---------------------------------------------------------------------------

const LoadingState: React.FC<{
  style?: React.CSSProperties;
  className?: string;
}> = ({ style, className }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: 200,
      ...style,
    }}
    className={className}
    role="status"
    aria-label="Loading chart"
  >
    <span style={{ color: "#888", fontSize: 14 }}>Loading…</span>
  </div>
);

const ErrorState: React.FC<{
  message: string;
  style?: React.CSSProperties;
  className?: string;
}> = ({ message, style, className }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: 200,
      color: "#c0392b",
      fontSize: 13,
      padding: 16,
      border: "1px solid #e74c3c",
      borderRadius: 4,
      ...style,
    }}
    className={className}
    role="alert"
  >
    <strong>Chart error:&nbsp;</strong>
    {message}
  </div>
);

// ---------------------------------------------------------------------------
// DynamicChart — routing shell
//
// Responsibilities:
//   1. Resolve ChartConfig (inline object OR remote JSON URL)
//   2. Resolve ChartDataRow[] (inline array OR remote JSON URL, cached)
//   3. Apply cross-filter from SelectionContext (other charts' selections)
//   4. Delegate to the renderer chosen by RendererContext
//
// The three renderers all receive the same (config, filteredData) pair so the
// ChartConfig schema remains the single source of truth regardless of which
// rendering library is active.
// ---------------------------------------------------------------------------

export const DynamicChart: React.FC<DynamicChartProps> = ({
  config: configSource,
  data: dataSource,
  onChartClick,
  onChartSelect,
  style,
  className,
}) => {
  // ── 1. Resolve config & data ─────────────────────────────────────────────
  const {
    config,
    loading: configLoading,
    error: configError,
  } = useChartConfig(configSource);
  const {
    data: rawData,
    loading: dataLoading,
    error: dataError,
    streaming,
  } = useChartData(dataSource);

  // ── 2. Cross-filter ──────────────────────────────────────────────────────
  const { selection, filterData } = useSelection();

  const displayData = useMemo<ChartDataRow[]>(() => {
    if (!config || !selection || selection.sourceChartId === config.id)
      return rawData;
    return filterData(rawData, selection.field);
  }, [rawData, config, selection, filterData]);

  // ── 3. Pick renderer ─────────────────────────────────────────────────────
  const { renderer } = useRenderer();

  // ── Loading / error guards ───────────────────────────────────────────────
  // For a streaming source we allow rendering as soon as at least one row has
  // arrived — the chart updates live as more rows land.  Only block on a full
  // loading spinner when there is truly nothing to show yet.
  const showLoading =
    configLoading ||
    (dataLoading && !streaming) ||
    (streaming && rawData.length === 0);

  if (showLoading) {
    return (
      <LoadingState
        {...(style ? { style } : {})}
        {...(className ? { className } : {})}
      />
    );
  }

  const error = configError ?? dataError;
  if (error) {
    return (
      <ErrorState
        message={error.message}
        {...(style ? { style } : {})}
        {...(className ? { className } : {})}
      />
    );
  }

  if (!config) return null;

  // ── 4. Delegate to the active renderer ───────────────────────────────────
  // Capture the renderer element so we can optionally wrap it with a "LIVE"
  // streaming badge when the stream is still open.
  let chartEl: React.ReactElement;
  switch (renderer) {
    case "d3":
      chartEl = (
        <D3Renderer
          config={config}
          data={displayData}
          {...(style ? { style } : {})}
          {...(className ? { className } : {})}
        />
      );
      break;

    case "plotly":
    default:
      chartEl = (
        <PlotlyRenderer
          config={config}
          data={displayData}
          {...(onChartClick ? { onChartClick } : {})}
          {...(onChartSelect ? { onChartSelect } : {})}
          {...(style ? { style } : {})}
          {...(className ? { className } : {})}
        />
      );
  }

  // While the NDJSON stream is still open, wrap the chart in a relative
  // container and overlay a pulsing "LIVE" badge in the top-right corner.
  if (!streaming) return chartEl;

  return (
    <div
      style={{ position: "relative", width: "100%", height: "100%" }}
      className={className}
    >
      <style>{`@keyframes dv-stream-pulse { 0%,100%{opacity:1} 50%{opacity:.25} }`}</style>
      {chartEl}
      <div style={streamBadgeStyle}>
        <span
          style={{
            ...streamDotStyle,
            animation: "dv-stream-pulse 1.4s ease-in-out infinite",
          }}
        />
        LIVE
      </div>
    </div>
  );
};

DynamicChart.displayName = "DynamicChart";

// ---------------------------------------------------------------------------
// Streaming badge styles
// ---------------------------------------------------------------------------

const streamBadgeStyle: React.CSSProperties = {
  position: "absolute",
  top: 6,
  right: 6,
  display: "flex",
  alignItems: "center",
  gap: 4,
  background: "rgba(255,255,255,0.88)",
  border: "1px solid #b7e4c7",
  borderRadius: 4,
  padding: "2px 7px",
  fontSize: 10,
  fontWeight: 700,
  color: "#27ae60",
  letterSpacing: ".06em",
  pointerEvents: "none",
  userSelect: "none",
  zIndex: 10,
};

const streamDotStyle: React.CSSProperties = {
  width: 6,
  height: 6,
  borderRadius: "50%",
  background: "#27ae60",
  display: "inline-block",
  flexShrink: 0,
};
