import React, { createContext, useContext } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * The two supported rendering backends.
 *
 *  - 'plotly' — react-plotly.js  (default, feature-complete)
 *  - 'd3'     — D3.js raw SVG (maximum customisability)
 *
 * Both backends consume the same `ChartConfig` schema — only the
 * rendering engine differs.
 */
export type RendererType = "plotly" | "d3";

export interface RendererMeta {
  type: RendererType;
  label: string;
  description: string;
  badge: string;
  color: string;
}

export const RENDERER_META: Record<RendererType, RendererMeta> = {
  plotly: {
    type: "plotly",
    label: "Plotly",
    description: "Plotly.js — interactive, zoom, pan, hover tooltips",
    badge: "PLT",
    color: "#3f4f75",
  },
  d3: {
    type: "d3",
    label: "D3.js",
    description: "D3.js — custom SVG rendering, full control",
    badge: "D3",
    color: "#f68026",
  },
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface RendererContextValue {
  renderer: RendererType;
}

const RendererContext = createContext<RendererContextValue>({
  renderer: "plotly",
});

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export interface RendererProviderProps {
  renderer: RendererType;
  children: React.ReactNode;
}

export const RendererProvider: React.FC<RendererProviderProps> = ({
  renderer,
  children,
}) => (
  <RendererContext.Provider value={{ renderer }}>
    {children}
  </RendererContext.Provider>
);

RendererProvider.displayName = "RendererProvider";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRenderer(): RendererContextValue {
  return useContext(RendererContext);
}
