import React from "react";
import { SelectionProvider } from "../context/SelectionContext";
import { RendererProvider } from "../context/RendererContext";
import type { RendererType } from "../context/RendererContext";

interface ChartProviderProps {
  children: React.ReactNode;
  /**
   * The rendering backend to use for all DynamicChart instances inside this
   * provider.
   *
   *  - 'plotly' — Plotly.js (default, feature-complete, interactive)
   *  - 'd3'     — D3.js raw SVG (maximum customisability)
   *
   * Both backends consume the same ChartConfig schema.
   * Defaults to 'plotly' when omitted.
   */
  renderer?: RendererType;
}

/**
 * Wraps one or more `DynamicChart` instances with:
 *   - `RendererProvider`  — makes the chosen rendering library available
 *   - `SelectionProvider` — cross-filtering state shared across all charts
 *
 * Usage:
 * ```tsx
 * <ChartProvider renderer="d3">
 *   <DynamicChart config={lineConfig} data={data} />
 *   <DynamicChart config={barConfig}  data={data} />
 * </ChartProvider>
 * ```
 */
export const ChartProvider: React.FC<ChartProviderProps> = ({
  children,
  renderer = "plotly",
}) => (
  <RendererProvider renderer={renderer}>
    <SelectionProvider>{children}</SelectionProvider>
  </RendererProvider>
);

ChartProvider.displayName = "ChartProvider";
