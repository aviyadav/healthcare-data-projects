import React, { useState, useCallback } from "react";
import { DynamicChart, useSelection } from "chart-lib";
import type { ChartConfig } from "chart-lib";
import type { PlotMouseEvent, PlotSelectionEvent } from "plotly.js";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ChartCardProps {
  label: string;
  configUrl: string;
  dataUrl: string;
  onChartClick?: (event: PlotMouseEvent, config: ChartConfig) => void;
  onChartSelect?: (event: PlotSelectionEvent, config: ChartConfig) => void;
}

// ---------------------------------------------------------------------------
// ChartCard
// Renders a labelled card containing a DynamicChart and a reset button.
// The reset button:
//   1. Clears the global cross-filter selection (via SelectionContext)
//   2. Remounts the chart (via chartKey) to restore default zoom / pan
//
// Must be rendered inside a <ChartProvider> so that useSelection works.
// ---------------------------------------------------------------------------

export const ChartCard: React.FC<ChartCardProps> = ({
  label,
  configUrl,
  dataUrl,
  onChartClick,
  onChartSelect,
}) => {
  const { clearSelection, selection } = useSelection();
  const [chartKey, setChartKey] = useState(0);
  const [isResetting, setIsResetting] = useState(false);

  const hasActiveFilter = selection !== null;

  const handleReset = useCallback(() => {
    setIsResetting(true);
    clearSelection();
    setChartKey((k) => k + 1);
    // Brief visual feedback then restore button state
    setTimeout(() => setIsResetting(false), 600);
  }, [clearSelection]);

  return (
    <div style={cardStyles.card}>
      {/* ── Card header ── */}
      <div style={cardStyles.header}>
        <span style={cardStyles.label}>{label}</span>

        <div style={cardStyles.headerRight}>
          {hasActiveFilter && (
            <span style={cardStyles.filterBadge} title="Cross-filter active">
              ● filtered
            </span>
          )}
          <button
            onClick={handleReset}
            style={{
              ...cardStyles.resetBtn,
              ...(isResetting ? cardStyles.resetBtnActive : {}),
            }}
            title="Reset zoom and clear cross-filter selection"
            aria-label={`Reset ${label} chart`}
          >
            <span
              style={{
                ...cardStyles.resetIcon,
                ...(isResetting ? cardStyles.resetIconSpin : {}),
              }}
            >
              ↺
            </span>
            Reset
          </button>
        </div>
      </div>

      {/* ── Chart area ── */}
      <div style={cardStyles.chartWrapper}>
        <DynamicChart
          key={chartKey}
          config={configUrl}
          data={dataUrl}
          {...(onChartClick ? { onChartClick } : {})}
          {...(onChartSelect ? { onChartSelect } : {})}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </div>
  );
};

ChartCard.displayName = "ChartCard";

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const cardStyles = {
  card: {
    background: "#ffffff",
    borderRadius: 10,
    boxShadow: "0 1px 6px rgba(0,0,0,.09)",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    transition: "box-shadow 0.2s",
    height: "100%",
  } satisfies React.CSSProperties,

  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 12px",
    borderBottom: "1px solid #f0f0f0",
    minHeight: 36,
    gap: 8,
  } satisfies React.CSSProperties,

  label: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase" as const,
    letterSpacing: ".07em",
    color: "#777",
    flexShrink: 0,
  } satisfies React.CSSProperties,

  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  filterBadge: {
    fontSize: 10,
    fontWeight: 600,
    color: "#e67e22",
    background: "#fef3e2",
    border: "1px solid #f0c080",
    borderRadius: 4,
    padding: "1px 6px",
    letterSpacing: ".03em",
    whiteSpace: "nowrap" as const,
  } satisfies React.CSSProperties,

  resetBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "3px 10px",
    fontSize: 11,
    fontWeight: 600,
    color: "#555",
    background: "#f5f5f5",
    border: "1px solid #ddd",
    borderRadius: 5,
    cursor: "pointer",
    outline: "none",
    transition: "background 0.15s, color 0.15s, border-color 0.15s",
    userSelect: "none" as const,
    letterSpacing: ".03em",
  } satisfies React.CSSProperties,

  resetBtnActive: {
    background: "#e8f4fd",
    color: "#2980b9",
    borderColor: "#aad4f0",
  } satisfies React.CSSProperties,

  resetIcon: {
    fontSize: 13,
    display: "inline-block",
    lineHeight: 1,
  } satisfies React.CSSProperties,

  resetIconSpin: {
    // CSS animation via inline style isn't ideal but works for a demo
    // A keyframe would be cleaner; here we use a subtle transform instead
    transform: "rotate(-45deg)",
    transition: "transform 0.3s ease",
  } satisfies React.CSSProperties,

  chartWrapper: {
    height: 300,
    padding: 4,
    flex: 1,
  } satisfies React.CSSProperties,
} as const;
