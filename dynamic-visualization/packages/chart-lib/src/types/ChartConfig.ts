import { z } from "zod";

export const AxisConfigSchema = z.object({
  field: z.string(),
  title: z.string().optional(),
  axisType: z.enum(["linear", "log", "date", "category"]).optional(),
});

export type AxisConfig = z.infer<typeof AxisConfigSchema>;

export const MarkerConfigSchema = z.object({
  colorField: z.string().optional(),
  sizeField: z.string().optional(),
  colorScale: z
    .enum([
      "Viridis",
      "Plasma",
      "Blues",
      "Reds",
      "Greens",
      "YlOrRd",
      "RdBu",
      "Picnic",
    ])
    .optional(),
  opacity: z.number().min(0).max(1).optional(),
  size: z.number().positive().optional(),
});

export type MarkerConfig = z.infer<typeof MarkerConfigSchema>;

export const ChartConfigSchema = z.object({
  /** Unique identifier — used for cross-filtering source tracking */
  id: z.string(),

  /** The chart variety to render */
  type: z.enum([
    "line",
    "bar",
    "scatter",
    "pie",
    "area",
    "heatmap",
    "histogram",
    "box",
    "sankey",
  ]),

  /** Chart title displayed above the plot */
  title: z.string().optional(),

  /** Primary horizontal axis (required for all except pie) */
  xAxis: AxisConfigSchema,

  /** Primary vertical axis (optional for pie/histogram where it is derived) */
  yAxis: AxisConfigSchema.optional(),

  /** Z-axis only used by heatmap; maps to the colour-intensity value */
  zAxis: AxisConfigSchema.optional(),

  /**
   * Sankey-only: the field whose values become the *target* nodes.
   * The source nodes come from `xAxis.field` and the flow weight from `yAxis.field`.
   */
  sankeyTarget: z.string().optional(),

  /** Marker and colour styling options */
  marker: MarkerConfigSchema.optional(),

  /**
   * Pass-through Plotly layout overrides merged last — lets callers fine-tune
   * anything via the config file without touching the component.
   */
  layout: z.record(z.string(), z.unknown()).optional(),

  /** Enable box-select / lasso on the chart (default: true) */
  enableSelection: z.boolean().default(true),

  /** Selection interaction mode (default: 'box') */
  selectionMode: z.enum(["box", "lasso"]).default("box"),

  /**
   * Pre-aggregation applied to the data before building Plotly traces.
   * Only relevant when groupBy is also set.
   */
  aggregation: z.enum(["sum", "count", "mean", "median"]).optional(),

  /** Field name used to split data into groups / traces */
  groupBy: z.string().optional(),

  /** Assigns all area-chart traces to a shared stack */
  stackGroup: z.string().optional(),

  /** Field whose distinct values each get their own colour/trace */
  colorBy: z.string().optional(),
});

export type ChartConfig = z.infer<typeof ChartConfigSchema>;
