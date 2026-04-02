// Components
export { DynamicChart } from "./components/DynamicChart";
export type { DynamicChartProps } from "./components/DynamicChart";
export { ChartProvider } from "./components/ChartProvider";

// Renderer context & types
export {
  useRenderer,
  RendererProvider,
  RENDERER_META,
} from "./context/RendererContext";
export type {
  RendererType,
  RendererMeta,
  RendererProviderProps,
} from "./context/RendererContext";

// Selection context & hooks
export { useSelection } from "./context/SelectionContext";
export type { SelectionState } from "./context/SelectionContext";

// Data hooks & cache utilities
export {
  prefetchData,
  clearDataCache,
  isStreamUrl,
  useStreamStatus,
  useChartData,
} from "./hooks/useChartData";
export type { StreamStatus } from "./hooks/useChartData";

// Types
export {
  ChartConfigSchema,
  AxisConfigSchema,
  MarkerConfigSchema,
} from "./types/ChartConfig";
export type {
  ChartConfig,
  AxisConfig,
  MarkerConfig,
} from "./types/ChartConfig";

export { ChartDataSchema, ChartDataRowSchema } from "./types/ChartData";
export type { ChartData, ChartDataRow } from "./types/ChartData";

// Adapters (exported for advanced consumers who want to build their own renderer)
export { buildTraces, buildLayout } from "./adapters/plotlyAdapter";

// Shared data utilities
export {
  PALETTE,
  aggregate,
  groupByField,
  col,
  buildColorArray,
  applyAggregation,
  buildSankeyData,
} from "./utils/dataUtils";
export type {
  SankeyNodeData,
  SankeyLinkData,
  SankeyData,
} from "./utils/dataUtils";

// Renderers (exported for advanced consumers who want to use them directly)
export { PlotlyRenderer } from "./renderers/PlotlyRenderer";
export type { PlotlyRendererProps } from "./renderers/PlotlyRenderer";

export { D3Renderer } from "./renderers/D3Renderer";
export type { D3RendererProps } from "./renderers/D3Renderer";
