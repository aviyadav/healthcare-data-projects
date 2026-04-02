import React, { useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { PlotMouseEvent, PlotSelectionEvent } from 'plotly.js';

import type { ChartConfig } from '../types/ChartConfig';
import type { ChartDataRow } from '../types/ChartData';
import { useSelection } from '../context/SelectionContext';
import { buildTraces, buildLayout } from '../adapters/plotlyAdapter';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PlotlyRendererProps {
  config: ChartConfig;
  data: ChartDataRow[];
  onChartClick?:  (event: PlotMouseEvent,    config: ChartConfig) => void;
  onChartSelect?: (event: PlotSelectionEvent, config: ChartConfig) => void;
  style?:    React.CSSProperties;
  className?: string;
}

// ---------------------------------------------------------------------------
// PlotlyRenderer
// ---------------------------------------------------------------------------

export const PlotlyRenderer: React.FC<PlotlyRendererProps> = ({
  config,
  data,
  onChartClick,
  onChartSelect,
  style,
  className,
}) => {
  const { setSelectionFromEvent } = useSelection();

  // Memoised trace and layout computation
  const traces = useMemo(() => buildTraces(config, data), [config, data]);
  const layout = useMemo(() => buildLayout(config),       [config]);

  const handleClick = useCallback(
    (event: Readonly<PlotMouseEvent>) => {
      setSelectionFromEvent(event as PlotMouseEvent, config);
      onChartClick?.(event as PlotMouseEvent, config);
    },
    [config, setSelectionFromEvent, onChartClick],
  );

  const handleSelect = useCallback(
    (event: Readonly<PlotSelectionEvent>) => {
      setSelectionFromEvent(event as PlotSelectionEvent, config);
      onChartSelect?.(event as PlotSelectionEvent, config);
    },
    [config, setSelectionFromEvent, onChartSelect],
  );

  return (
    <Plot
      data={traces}
      layout={{ ...layout, autosize: true }}
      config={{
        responsive:     true,
        displaylogo:    false,
        scrollZoom:     true,
        modeBarButtonsToRemove: ['toImage'],
      }}
      style={{ width: '100%', height: '100%', ...style }}
      className={className}
      onClick={handleClick}
      onSelected={handleSelect}
      useResizeHandler
    />
  );
};

PlotlyRenderer.displayName = 'PlotlyRenderer';
