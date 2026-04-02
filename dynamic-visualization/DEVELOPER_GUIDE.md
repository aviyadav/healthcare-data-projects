# Developer Integration Guide — dynamic-visualization / chart-lib

> **Audience:** Engineers embedding the `chart-lib` visualization utility into their own React
> applications. Assumes familiarity with React, TypeScript, and REST APIs.  
> **Assumption:** Chart data is delivered from a backend API at runtime, and chart configurations
> (type, axis mappings, aggregation rules) are constructed dynamically in code rather than
> hard-coded in static JSON files.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Package Overview](#2-package-overview)
3. [Core Concepts](#3-core-concepts)
   - 3.1 [ChartConfig — the single source of truth](#31-chartconfig--the-single-source-of-truth)
   - 3.2 [ChartData — what your API must return](#32-chartdata--what-your-api-must-return)
   - 3.3 [Derived fields added automatically](#33-derived-fields-added-automatically)
   - 3.4 [The rendering pipeline](#34-the-rendering-pipeline)
4. [Step 1 — Wrap your app with ChartProvider](#4-step-1--wrap-your-app-with-chartprovider)
5. [Step 2 — Build configs dynamically](#5-step-2--build-configs-dynamically)
   - 5.1 [Config builder pattern](#51-config-builder-pattern)
   - 5.2 [Discovering fields from a live API response](#52-discovering-fields-from-a-live-api-response)
6. [Step 3 — Mount DynamicChart](#6-step-3--mount-dynamicchart)
7. [Chart Type Reference — config recipes for all 9 types](#7-chart-type-reference--config-recipes-for-all-9-types)
   - 7.1 [Line](#71-line)
   - 7.2 [Bar](#72-bar)
   - 7.3 [Scatter](#73-scatter)
   - 7.4 [Pie / Donut](#74-pie--donut)
   - 7.5 [Area](#75-area)
   - 7.6 [Heatmap](#76-heatmap)
   - 7.7 [Histogram](#77-histogram)
   - 7.8 [Box Plot](#78-box-plot)
   - 7.9 [Sankey](#79-sankey)
8. [Aggregation in depth](#8-aggregation-in-depth)
9. [Renderer selection](#9-renderer-selection)
10. [Data fetching & caching](#10-data-fetching--caching)
    - 10.1 [JSON REST endpoints](#101-json-rest-endpoints)
    - 10.2 [NDJSON streaming endpoints](#102-ndjson-streaming-endpoints)
    - 10.3 [Prefetching for faster first render](#103-prefetching-for-faster-first-render)
    - 10.4 [Paginated APIs — Clinical Data pattern](#104-paginated-apis--clinical-data-pattern)
    - 10.5 [Cache invalidation](#105-cache-invalidation)
11. [Cross-filtering](#11-cross-filtering)
    - 11.1 [How it works](#111-how-it-works)
    - 11.2 [Reading selection state](#112-reading-selection-state)
    - 11.3 [Programmatic selection and clearing](#113-programmatic-selection-and-clearing)
12. [Dynamic dashboard — full worked example](#12-dynamic-dashboard--full-worked-example)
12.5 [Adding a new data source to the demo app](#125-adding-a-new-data-source-to-the-demo-app)
13. [Config validation at runtime](#13-config-validation-at-runtime)
14. [Authentication & custom fetch headers](#14-authentication--custom-fetch-headers)
15. [Error handling](#15-error-handling)
16. [Performance guide](#16-performance-guide)
17. [Troubleshooting](#17-troubleshooting)
18. [Full ChartConfig field reference](#18-full-chartconfig-field-reference)

---

## 1. Prerequisites & Installation

### Bun requirements

| Tool | Minimum version |
|---|---|
| Bun | 1.1 |
| React | 18 |
| TypeScript | 5 (recommended) |

### Install peer dependencies and the library

```bash
# bun
bun add react react-dom plotly.js react-plotly.js d3 d3-sankey chart-lib

# npm
npm install react react-dom plotly.js react-plotly.js d3 d3-sankey chart-lib

# yarn
yarn add react react-dom plotly.js react-plotly.js d3 d3-sankey chart-lib
```

**Why each package:**

| Package | Role |
|---|---|
| `react` / `react-dom` | UI framework — peer dependency, not bundled |
| `plotly.js` | Plotly rendering engine — peer dependency |
| `react-plotly.js` | React wrapper for Plotly — peer dependency |
| `d3` | D3.js SVG rendering + layout utilities — bundled but externalized |
| `d3-sankey` | Sankey layout engine — bundled but externalized |
| `chart-lib` | The visualization utility itself |

### TypeScript configuration

Add `"moduleResolution": "bundler"` (or `"node16"`) to your `tsconfig.json` so named exports
from the ESM bundle resolve correctly:

```json
{
  "compilerOptions": {
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx"
  }
}
```

---

## 2. Package Overview

`chart-lib` exports everything you need from a single entry point:

```typescript
import {
  // ── Core components ───────────────────────────────────────────────────
  DynamicChart,       // The chart component — resolves config + data, picks renderer
  ChartProvider,      // Context wrapper — renderer choice + cross-filter state

  // ── Renderer control ──────────────────────────────────────────────────
  RENDERER_META,      // Metadata (label, colour, badge) for each renderer
  useRenderer,        // Hook — read the active renderer inside a ChartProvider
  RendererProvider,   // Low-level context provider (ChartProvider wraps this)

  // ── Cross-filter ──────────────────────────────────────────────────────
  useSelection,       // Hook — read/write cross-filter selection state

  // ── Data utilities ────────────────────────────────────────────────────
  prefetchData,       // Warm the cache before charts mount
  clearDataCache,     // Evict one URL (or all URLs) from the cache
  isStreamUrl,        // Boolean — true when URL is an NDJSON stream endpoint
  useStreamStatus,    // Hook — live stream progress (rowCount, streaming flag, timing)

  // ── Validation schemas (Zod) ──────────────────────────────────────────
  ChartConfigSchema,  // Zod schema — validates a ChartConfig at runtime
  ChartDataSchema,    // Zod schema — validates an array of data rows

  // ── Shared data utilities ─────────────────────────────────────────────
  buildSankeyData,    // Prepare {nodes, links} for a Sankey renderer
  PALETTE,            // 10-colour array used by all renderers
  aggregate,          // Group + reduce rows by a single field
  groupByField,       // Split rows into a Map keyed by a categorical field
  applyAggregation,   // Apply aggregation from a ChartConfig to raw rows

  // ── Low-level renderer components ─────────────────────────────────────
  PlotlyRenderer,     // Use directly only for custom wrappers
  D3Renderer,         // Use directly only for custom wrappers

  // ── Adapter (advanced) ────────────────────────────────────────────────
  buildTraces,        // Convert ChartConfig + rows → Plotly Data[]
  buildLayout,        // Convert ChartConfig → Plotly Layout
} from 'chart-lib';

// ── Types ──────────────────────────────────────────────────────────────────
import type {
  ChartConfig,        // Full config type inferred from ChartConfigSchema
  AxisConfig,         // { field, title?, axisType? }
  MarkerConfig,       // { colorField?, sizeField?, colorScale?, opacity?, size? }
  ChartData,          // ChartDataRow[]
  ChartDataRow,       // Record<string, string | number | null>
  RendererType,       // 'plotly' | 'd3'
  RendererMeta,       // label, description, badge, color per renderer
  SelectionState,     // { sourceChartId, selectedValues, field, selectedIndices }
  StreamStatus,       // { rowCount, streaming, firstRowMs, lastRowMs, error }
  SankeyData,         // { nodes: SankeyNodeData[], links: SankeyLinkData[] }
  SankeyNodeData,     // { name: string }
  SankeyLinkData,     // { source: number, target: number, value: number }
} from 'chart-lib';
```

---

## 3. Core Concepts

### 3.1 ChartConfig — the single source of truth

Every chart is fully described by a `ChartConfig` object. The same object drives both rendering
backends (Plotly and D3), so swapping the renderer requires zero config changes.

```typescript
type ChartConfig = {
  // ── Required ────────────────────────────────────────────────────────────
  id:    string;                    // Unique string — used for cross-filter tracking
  type:  ChartType;                 // One of 9 values — see §7
  xAxis: AxisConfig;                // Source nodes for Sankey; horizontal axis elsewhere

  // ── Optional — axis ─────────────────────────────────────────────────────
  title?:        string;
  yAxis?:        AxisConfig;        // Required for all types except histogram
  zAxis?:        AxisConfig;        // Heatmap only — maps to colour intensity
  sankeyTarget?: string;            // Sankey only — target-node field name

  // ── Optional — visual ───────────────────────────────────────────────────
  marker?: {
    colorField?:  string;           // Drive point colour from a data column
    sizeField?:   string;           // Drive point size from a data column
    colorScale?:  ColorScaleName;   // 'Viridis' | 'Plasma' | 'Blues' | 'Reds' |
                                    // 'Greens' | 'YlOrRd' | 'RdBu' | 'Picnic'
    opacity?:     number;           // 0–1
    size?:        number;           // Fixed marker size in px
  };
  layout?: Record<string, unknown>; // Plotly layout overrides (merged last, Plotly only)

  // ── Optional — interaction ──────────────────────────────────────────────
  enableSelection?: boolean;        // default: true
  selectionMode?:   'box' | 'lasso'; // default: 'box'

  // ── Optional — data shaping ─────────────────────────────────────────────
  aggregation?: 'sum' | 'count' | 'mean' | 'median';
  groupBy?:     string;             // Field to group by when aggregating
  stackGroup?:  string;             // Area chart stacking group name
  colorBy?:     string;             // Split into traces by distinct values of this field
};

type AxisConfig = {
  field:     string;
  title?:    string;
  axisType?: 'linear' | 'log' | 'date' | 'category';
};
```

`ChartConfig` objects can be constructed entirely in TypeScript at runtime — they do not need to
live in JSON files. This is the recommended approach when the chart schema is driven by API
metadata or user configuration.

### 3.2 ChartData — what your API must return

Data arrives as a flat array of row objects. Every value must be a primitive — `string`, `number`,
or `null`. Nested objects are not supported; flatten your API response before passing it in.

```typescript
// Acceptable response shapes:

// Shape 1 — bare array (simplest)
[
  { "month": "2025-01", "region": "North", "revenue": 15200, "units": 152 },
  { "month": "2025-01", "region": "South", "revenue":  9800, "units": 196 }
]

// Shape 2 — wrapped object { data: [...] }
// The library unwraps this automatically.
{
  "data": [
    { "month": "2025-01", "region": "North", "revenue": 15200, "units": 152 }
  ]
}
```

The library validates both shapes with a Zod schema at runtime. Invalid or unexpected rows render
a styled error card — they never crash the page.

### 3.3 Derived fields added automatically

The library enriches every row with two computed fields after fetch/parse:

| Field | Value | Use case |
|---|---|---|
| `_count` | Always `1` | Enables `aggregation: "count"` on any `groupBy` field without needing a pre-counted column in your API |
| `_timestamp` | Epoch milliseconds parsed from `created_ts`, or `null` | Use as a numeric Y-axis for time-series box plots |

If your API returns a `created_ts` ISO string field, `_timestamp` is populated automatically and
can be used in axis config immediately:

```json
{ "field": "_timestamp", "title": "Created (epoch ms)" }
```

### 3.4 The rendering pipeline

```
Your code
  │
  ├─ builds ChartConfig (inline object)
  ├─ provides data URL (string) or inline ChartDataRow[]
  │
  ▼
DynamicChart
  ├─ useChartConfig  → validates + caches config
  ├─ useChartData    → fetches + caches data (one request shared by all charts)
  ├─ SelectionContext → applies cross-filter from sibling charts
  │
  ▼  reads RendererContext.renderer
  ├─ 'plotly' → buildTraces(config, data) → buildLayout(config) → <Plot />
  └─ 'd3'     → renderXxx(svg, config, data, w, h, onSelect) → SVG DOM
```

---

## 4. Step 1 — Wrap your app with ChartProvider

Every `DynamicChart` must be a descendant of a `<ChartProvider>`. The provider supplies two
things:

1. The active **renderer** (`'plotly'` or `'d3'`) to all descendant charts
2. The shared **cross-filter selection state** so charts filter each other automatically

```tsx
// src/Dashboard.tsx
import { ChartProvider } from 'chart-lib';

export function Dashboard() {
  return (
    // All charts inside use Plotly (default). Change to renderer="d3" for D3.
    <ChartProvider renderer="plotly">
      {/* your DynamicChart components go here */}
    </ChartProvider>
  );
}
```

**Multiple independent dashboards on the same page** each get their own `<ChartProvider>`.
Charts inside separate providers never cross-filter each other:

```tsx
<ChartProvider renderer="plotly">
  {/* Sales dashboard — cross-filters only within this group */}
  <SalesDashboard />
</ChartProvider>

<ChartProvider renderer="d3">
  {/* Operations dashboard — completely isolated */}
  <OperationsDashboard />
</ChartProvider>
```

**Switching the renderer at runtime** — key the provider on the renderer value so React fully
remounts it (which also clears any active cross-filter selection):

```tsx
const [renderer, setRenderer] = useState<RendererType>('plotly');

<ChartProvider key={renderer} renderer={renderer}>
  {/* ... */}
</ChartProvider>
```

---

## 5. Step 2 — Build configs dynamically

### 5.1 Config builder pattern

Rather than maintaining static JSON files, construct `ChartConfig` objects in TypeScript. The
full type is exported, so you get autocomplete and compile-time checking.

```typescript
import type { ChartConfig } from 'chart-lib';

// ── Simple factory function ────────────────────────────────────────────────

function makeLineConfig(params: {
  id: string;
  title: string;
  xField: string;
  yField: string;
  colorByField?: string;
  xAxisType?: 'linear' | 'log' | 'date' | 'category';
}): ChartConfig {
  return {
    id: params.id,
    type: 'line',
    title: params.title,
    xAxis: {
      field: params.xField,
      title: params.xField,
      axisType: params.xAxisType,
    },
    yAxis: {
      field: params.yField,
      title: params.yField,
    },
    ...(params.colorByField ? { colorBy: params.colorByField } : {}),
    enableSelection: true,
    selectionMode: 'box',
  };
}

// Usage:
const revenueOverTime = makeLineConfig({
  id:          'revenue-trend',
  title:       'Monthly Revenue by Region',
  xField:      'month',
  xAxisType:   'date',
  yField:      'revenue',
  colorByField:'region',
});
```

### 5.2 Discovering fields from a live API response

A common pattern is to inspect the first row of an API response and derive axis mappings
programmatically — for example, letting a user pick X and Y axes from a dropdown list of
available column names.

```typescript
import { ChartDataRowSchema } from 'chart-lib';
import type { ChartDataRow, ChartConfig } from 'chart-lib';

// ── Step 1: Fetch a sample of your data ────────────────────────────────────

async function fetchColumnNames(apiUrl: string): Promise<string[]> {
  const res  = await fetch(apiUrl);
  const json = await res.json() as unknown;

  // Normalise both response shapes: bare array or { data: [...] }
  const rows = Array.isArray(json)
    ? json
    : (json as { data?: unknown[] }).data ?? [];

  const firstRow = rows[0];
  if (!firstRow || typeof firstRow !== 'object') return [];

  // Validate the first row against ChartDataRowSchema and extract keys
  const parsed = ChartDataRowSchema.safeParse(firstRow);
  return parsed.success ? Object.keys(parsed.data) : Object.keys(firstRow as object);
}

// ── Step 2: Classify columns by inferred type ──────────────────────────────

type ColumnKind = 'numeric' | 'categorical' | 'date';

function classifyColumn(rows: ChartDataRow[], field: string): ColumnKind {
  const sample = rows
    .slice(0, 20)
    .map(r => r[field])
    .filter(v => v !== null && v !== undefined);

  if (sample.every(v => typeof v === 'number')) return 'numeric';

  // ISO date heuristic: YYYY-MM or YYYY-MM-DD
  if (sample.every(v => typeof v === 'string' && /^\d{4}-\d{2}/.test(String(v)))) return 'date';

  return 'categorical';
}

// ── Step 3: Build a config from user selections ────────────────────────────

interface UserSelection {
  chartType: ChartConfig['type'];
  xField:    string;
  yField?:   string;
  groupBy?:  string;
  zField?:   string;         // Heatmap
  targetField?: string;      // Sankey
}

function buildConfigFromSelection(
  id: string,
  rows: ChartDataRow[],
  sel: UserSelection,
): ChartConfig {
  const xKind = classifyColumn(rows, sel.xField);

  const base: ChartConfig = {
    id,
    type: sel.chartType,
    xAxis: {
      field:    sel.xField,
      title:    sel.xField,
      axisType: xKind === 'date'        ? 'date'
               : xKind === 'numeric'    ? 'linear'
               : /* categorical */        'category',
    },
    enableSelection: sel.chartType !== 'heatmap' && sel.chartType !== 'sankey',
    selectionMode:   'box',
  };

  if (sel.yField) {
    base.yAxis = { field: sel.yField, title: sel.yField };
  }

  if (sel.zField) {
    base.zAxis = { field: sel.zField, title: sel.zField };
  }

  if (sel.groupBy) {
    base.groupBy     = sel.groupBy;
    base.colorBy     = sel.groupBy;
    base.aggregation = classifyColumn(rows, sel.yField ?? sel.xField) === 'numeric'
      ? 'sum'
      : 'count';
  }

  if (sel.targetField) {
    base.sankeyTarget = sel.targetField;
    base.aggregation  = 'sum';
  }

  return base;
}
```

---

## 6. Step 3 — Mount DynamicChart

`DynamicChart` accepts either an **inline `ChartConfig` object** (for dynamic configs you build
in code) or a **URL string** (for configs served from a config API). The same applies to data.

```tsx
import { DynamicChart } from 'chart-lib';
import type { ChartConfig } from 'chart-lib';

// ── Option A: inline config + data URL (most common for dynamic dashboards) ─

<DynamicChart
  config={myConfig}                    // ChartConfig object built in code
  data="/api/v1/sales/monthly"         // URL → library fetches + caches it
  style={{ height: 360 }}
/>

// ── Option B: config URL + data URL (fully server-driven) ──────────────────

<DynamicChart
  config="/api/v1/charts/revenue/config"
  data="/api/v1/charts/revenue/data"
  style={{ height: 360 }}
/>

// ── Option C: inline config + inline data (for pre-loaded data) ────────────

<DynamicChart
  config={myConfig}
  data={myRows}                        // ChartDataRow[] already in memory
  style={{ height: 360 }}
/>

// ── Option D: NDJSON stream (URL whose last path segment starts with "stream") ──

<DynamicChart
  config={myConfig}
  data="/api/v1/sales/stream"          // Library detects stream automatically
  style={{ height: 360 }}
/>
```

**Available props:**

| Prop | Type | Required | Description |
|---|---|---|---|
| `config` | `ChartConfig \| string` | ✅ | Inline config object or URL to a config JSON endpoint |
| `data` | `ChartDataRow[] \| string` | ✅ | Inline rows or URL to a data endpoint |
| `onChartClick` | `(event, config) => void` | — | Point click callback — Plotly renderer only |
| `onChartSelect` | `(event, config) => void` | — | Box/lasso complete callback — Plotly renderer only |
| `style` | `React.CSSProperties` | — | Applied to the chart container |
| `className` | `string` | — | CSS class on the chart container |

The component manages its own **loading** (spinner) and **error** (styled card) states. It never
throws or crashes its parent.

---

## 7. Chart Type Reference — config recipes for all 9 types

Every recipe below shows:
- The minimal required config
- All optional tuning fields for that type
- A realistic dynamic builder function
- Notes on what the API data shape should look like

---

### 7.1 Line

**Purpose:** Continuous trends over time or across an ordered X axis. Supports one or many
coloured traces via `colorBy`.

**Data shape:**
```
[{ month: "2025-01", region: "North", revenue: 15200 }, ...]
```

**Minimal config:**
```typescript
const lineConfig: ChartConfig = {
  id:    'revenue-line',
  type:  'line',
  xAxis: { field: 'month', axisType: 'date' },
  yAxis: { field: 'revenue' },
};
```

**Full options:**
```typescript
const lineConfig: ChartConfig = {
  id:    'revenue-line',
  type:  'line',
  title: 'Monthly Revenue by Region',

  xAxis: {
    field:    'month',
    title:    'Month',
    axisType: 'date',          // 'linear' | 'log' | 'date' | 'category'
  },
  yAxis: {
    field:    'revenue',
    title:    'Revenue ($)',
  },

  // Split into one trace per distinct value of this field
  colorBy: 'region',

  // Interactive selection
  enableSelection: true,
  selectionMode:   'box',      // 'box' | 'lasso'

  // Plotly layout overrides (merged last; Plotly renderer only)
  layout: { showlegend: true },
};
```

**Dynamic builder:**
```typescript
function makeLineConfig(opts: {
  id: string;
  xField: string;
  yField: string;
  colorByField?: string;
  isTimeSeries?: boolean;
}): ChartConfig {
  return {
    id:    opts.id,
    type:  'line',
    xAxis: { field: opts.xField, axisType: opts.isTimeSeries ? 'date' : 'category' },
    yAxis: { field: opts.yField },
    ...(opts.colorByField ? { colorBy: opts.colorByField } : {}),
    enableSelection: true,
    selectionMode:   'box',
  };
}
```

**Key rules:**
- `xAxis` and `yAxis` are both required.
- `axisType: 'date'` requires the field to contain ISO date strings (`YYYY-MM-DD` or `YYYY-MM`).
- Without `colorBy`, a single trace is drawn. With `colorBy`, one trace is created per distinct
  category value — no aggregation is applied, so ensure your data is already grouped or use
  `aggregation` + `groupBy` together.

---

### 7.2 Bar

**Purpose:** Compare discrete categories. Supports grouped bars (`colorBy`) and pre-aggregation
(`aggregation` + `groupBy`).

**Data shape:**
```
[{ status: "open", priority: "high", _count: 1 }, ...]
```

**Minimal config:**
```typescript
const barConfig: ChartConfig = {
  id:    'status-bar',
  type:  'bar',
  xAxis: { field: 'status', axisType: 'category' },
  yAxis: { field: '_count' },
};
```

**Full options with aggregation:**
```typescript
const barConfig: ChartConfig = {
  id:    'status-bar',
  type:  'bar',
  title: 'Issues by Status',

  xAxis: { field: 'status',  title: 'Status',  axisType: 'category' },
  yAxis: { field: '_count',  title: 'Count' },

  // Group raw rows by 'status' and sum '_count' (which is always 1 per row,
  // so sum == count of rows in each group)
  aggregation: 'sum',
  groupBy:     'status',

  // Colour each bar by status
  colorBy: 'status',

  enableSelection: true,
  selectionMode:   'box',
};
```

**Dynamic builder:**
```typescript
function makeBarConfig(opts: {
  id:          string;
  categoryField: string;
  valueField:    string;
  aggregation?:  ChartConfig['aggregation'];
  colorBy?:      string;
  title?:        string;
}): ChartConfig {
  return {
    id:          opts.id,
    type:        'bar',
    title:       opts.title,
    xAxis:       { field: opts.categoryField, axisType: 'category' },
    yAxis:       { field: opts.valueField },
    aggregation: opts.aggregation ?? 'sum',
    groupBy:     opts.categoryField,
    colorBy:     opts.colorBy ?? opts.categoryField,
    enableSelection: true,
    selectionMode:   'box',
  };
}
```

**Key rules:**
- `aggregation` requires `groupBy` to also be set. Without both, raw rows are plotted as-is.
- Use `_count` as `yAxis.field` with `aggregation: 'sum'` to count rows per group.
- `colorBy` without `groupBy` splits already-aggregated rows into separate bar traces
  (Plotly: side-by-side bars via `barmode: 'group'`).

---

### 7.3 Scatter

**Purpose:** Correlate two numeric dimensions. Optionally encode a third dimension via marker
colour or size.

**Data shape:**
```
[{ volume: 152, revenue: 15200, category: "Electronics" }, ...]
```

**Minimal config:**
```typescript
const scatterConfig: ChartConfig = {
  id:    'vol-rev-scatter',
  type:  'scatter',
  xAxis: { field: 'volume' },
  yAxis: { field: 'revenue' },
};
```

**Full options:**
```typescript
const scatterConfig: ChartConfig = {
  id:    'vol-rev-scatter',
  type:  'scatter',
  title: 'Volume vs Revenue by Category',

  xAxis: { field: 'volume',  title: 'Volume (units)', axisType: 'linear' },
  yAxis: { field: 'revenue', title: 'Revenue ($)',     axisType: 'linear' },

  marker: {
    colorField: 'category',  // Colour each point by a categorical field
    size:       10,           // Fixed size in px (omit to use sizeField instead)
    opacity:    0.7,          // 0–1
    // colorScale: 'Viridis', // Use a continuous colour scale with a numeric colorField
  },

  // Lasso selection is often more useful than box for scatter
  enableSelection: true,
  selectionMode:   'lasso',
};
```

**Dynamic builder:**
```typescript
function makeScatterConfig(opts: {
  id:          string;
  xField:      string;
  yField:      string;
  colorField?: string;
  sizeField?:  string;
  title?:      string;
}): ChartConfig {
  return {
    id:    opts.id,
    type:  'scatter',
    title: opts.title,
    xAxis: { field: opts.xField, axisType: 'linear' },
    yAxis: { field: opts.yField, axisType: 'linear' },
    marker: {
      ...(opts.colorField ? { colorField: opts.colorField } : {}),
      ...(opts.sizeField  ? { sizeField:  opts.sizeField  } : {}),
      size:    8,
      opacity: 0.7,
    },
    enableSelection: true,
    selectionMode:   'lasso',
  };
}
```

**Key rules:**
- `marker.colorField` with a **categorical** column colours by category using `PALETTE`.
- `marker.colorField` with a **numeric** column requires `marker.colorScale` to render a gradient.
- `marker.sizeField` must reference a numeric column; values are used directly as pixel radii.
- Neither `aggregation` nor `groupBy` applies to scatter — raw rows are plotted individually.

---

### 7.4 Pie / Donut

**Purpose:** Show proportions of a whole. Renders as a donut (hole in the centre) by default.

**Data shape — pre-aggregated (simplest):**
```
[{ category: "Electronics", total: 42000 }, { category: "Clothing", total: 31000 }]
```

**Data shape — raw rows (with aggregation):**
```
[{ category: "Electronics", revenue: 15200 }, { category: "Electronics", revenue: 9800 }, ...]
```

**Minimal config (pre-aggregated data):**
```typescript
const pieConfig: ChartConfig = {
  id:    'category-pie',
  type:  'pie',
  xAxis: { field: 'category' },   // Label field
  yAxis: { field: 'total' },      // Value field
};
```

**With aggregation (raw row data):**
```typescript
const pieConfig: ChartConfig = {
  id:    'category-pie',
  type:  'pie',
  title: 'Revenue by Category',

  xAxis: { field: 'category', title: 'Category' },
  yAxis: { field: 'revenue',  title: 'Revenue ($)' },

  aggregation: 'sum',
  groupBy:     'category',

  // Pie charts are typically not interactive for cross-filtering
  enableSelection: false,
};
```

**Dynamic builder:**
```typescript
function makePieConfig(opts: {
  id:           string;
  labelField:   string;
  valueField:   string;
  aggregation?: ChartConfig['aggregation'];
  title?:       string;
}): ChartConfig {
  return {
    id:          opts.id,
    type:        'pie',
    title:       opts.title,
    xAxis:       { field: opts.labelField },
    yAxis:       { field: opts.valueField },
    aggregation: opts.aggregation ?? 'sum',
    groupBy:     opts.labelField,
    enableSelection: false,
  };
}
```

**Key rules:**
- `xAxis.field` provides the **slice labels**; `yAxis.field` provides the **slice values**.
- `enableSelection: false` is recommended — pie slices do not participate meaningfully in
  cross-filtering.
- The donut hole is fixed; there is no config option to render a filled pie (use `layout` overrides
  if you need to override this at the Plotly level).

---

### 7.5 Area

**Purpose:** Visualise cumulative or stacked quantities over time. Most useful with `colorBy` and
`stackGroup` to create a stacked area chart.

**Data shape:**
```
[{ month: "2025-01", category: "Electronics", sales: 15200 }, ...]
```

**Minimal config:**
```typescript
const areaConfig: ChartConfig = {
  id:    'sales-area',
  type:  'area',
  xAxis: { field: 'month' },
  yAxis: { field: 'sales' },
};
```

**Stacked area:**
```typescript
const areaConfig: ChartConfig = {
  id:    'sales-area',
  type:  'area',
  title: 'Monthly Sales by Category (Stacked)',

  xAxis: { field: 'month',    title: 'Month',      axisType: 'category' },
  yAxis: { field: 'sales',    title: 'Sales ($)' },

  // One filled area trace per category, all stacked into the same group
  colorBy:    'category',
  stackGroup: 'one',           // Arbitrary name — all traces with the same name stack together

  enableSelection: true,
  selectionMode:   'box',
};
```

**Dynamic builder:**
```typescript
function makeAreaConfig(opts: {
  id:          string;
  xField:      string;
  yField:      string;
  colorBy?:    string;
  stacked?:    boolean;
  title?:      string;
}): ChartConfig {
  return {
    id:    opts.id,
    type:  'area',
    title: opts.title,
    xAxis: { field: opts.xField },
    yAxis: { field: opts.yField },
    ...(opts.colorBy ? { colorBy: opts.colorBy } : {}),
    ...(opts.stacked ? { stackGroup: 'one' } : {}),
    enableSelection: true,
    selectionMode:   'box',
  };
}
```

**Key rules:**
- Without `colorBy`, a single filled area trace is drawn filling to Y=0.
- With `colorBy`, each category gets a separate trace. Add `stackGroup: 'one'` to stack them.
- Without `stackGroup` but with `colorBy`, traces overlap rather than stack.
- `aggregation` + `groupBy` can pre-aggregate the data before splitting by `colorBy` — useful
  when your API returns raw event rows rather than pre-summed time-series data.

---

### 7.6 Heatmap

**Purpose:** Show intensity of a numeric value across two categorical dimensions laid out as a
grid.

**Data shape:**
```
[{ region: "North", category: "Electronics", sales: 15200 }, ...]
```

**Minimal config:**
```typescript
const heatmapConfig: ChartConfig = {
  id:    'region-category-heatmap',
  type:  'heatmap',
  xAxis: { field: 'region' },
  yAxis: { field: 'category' },
  zAxis: { field: 'sales' },
};
```

**Full options:**
```typescript
const heatmapConfig: ChartConfig = {
  id:    'region-category-heatmap',
  type:  'heatmap',
  title: 'Sales Intensity: Region × Category',

  xAxis: { field: 'region',   title: 'Region' },
  yAxis: { field: 'category', title: 'Category' },
  zAxis: { field: 'sales',    title: 'Sales ($)' },   // The intensity value

  marker: {
    colorScale: 'YlOrRd',   // One of: Viridis | Plasma | Blues | Reds |
                             //         Greens  | YlOrRd | RdBu  | Picnic
  },

  // Heatmaps are not typically used for cross-filtering
  enableSelection: false,
};
```

**Dynamic builder:**
```typescript
function makeHeatmapConfig(opts: {
  id:          string;
  xField:      string;   // Horizontal axis — categorical
  yField:      string;   // Vertical axis — categorical
  zField:      string;   // Cell intensity — numeric
  colorScale?: ChartConfig['marker'] extends { colorScale?: infer S } ? S : never;
  title?:      string;
}): ChartConfig {
  return {
    id:    opts.id,
    type:  'heatmap',
    title: opts.title,
    xAxis: { field: opts.xField },
    yAxis: { field: opts.yField },
    zAxis: { field: opts.zField },
    marker: { colorScale: opts.colorScale ?? 'Viridis' },
    enableSelection: false,
  };
}
```

**Key rules:**
- All three axes (`xAxis`, `yAxis`, `zAxis`) are required.
- The library builds the Z matrix automatically by iterating the rows: for each `(xVal, yVal)`
  pair it sums all `zAxis.field` values found in the data.
- Both X and Y dimensions must be **categorical** (string values).
- `axisType` on `xAxis`/`yAxis` is ignored for heatmaps — the library always uses a `scaleBand`
  in D3 and categorical scale in Plotly.

---

### 7.7 Histogram

**Purpose:** Show the frequency distribution of a single numeric or categorical dimension.
Supports overlaid groups via `colorBy`.

**Data shape:**
```
[{ price: 99.99, department: "Electronics" }, ...]
```

**Minimal config:**
```typescript
const histConfig: ChartConfig = {
  id:    'price-histogram',
  type:  'histogram',
  xAxis: { field: 'price' },
};
```

**With overlaid groups:**
```typescript
const histConfig: ChartConfig = {
  id:    'price-histogram',
  type:  'histogram',
  title: 'Price Distribution by Department',

  xAxis: { field: 'price', title: 'Price ($)', axisType: 'linear' },

  // Separate overlapping histogram bars per department
  colorBy: 'department',

  marker: { opacity: 0.75 },   // Reduce opacity so overlapping bars are visible

  enableSelection: true,
  selectionMode:   'box',
};
```

**Dynamic builder:**
```typescript
function makeHistogramConfig(opts: {
  id:       string;
  xField:   string;
  colorBy?: string;
  opacity?: number;
  title?:   string;
}): ChartConfig {
  return {
    id:      opts.id,
    type:    'histogram',
    title:   opts.title,
    xAxis:   { field: opts.xField },
    colorBy: opts.colorBy,
    marker:  { opacity: opts.opacity ?? 0.75 },
    enableSelection: true,
    selectionMode:   'box',
  };
}
```

**Key rules:**
- `yAxis` is **not used** — the Y axis always represents bin count, computed by the rendering
  engine.
- `colorBy` creates one overlapping histogram per distinct value; set `marker.opacity` < 1 so
  all groups remain visible.
- `aggregation` / `groupBy` do not apply to histogram — binning is handled internally by Plotly
  or D3's `d3.bin()`.

---

### 7.8 Box Plot

**Purpose:** Show the statistical distribution (median, IQR, whiskers, outliers) of a numeric
field across categories.

**Data shape:**
```
[{ department: "Engineering", salary: 95000 }, { department: "Sales", salary: 72000 }, ...]
```

**Minimal config:**
```typescript
const boxConfig: ChartConfig = {
  id:    'salary-box',
  type:  'box',
  xAxis: { field: 'department' },
  yAxis: { field: 'salary' },
};
```

**Full options:**
```typescript
const boxConfig: ChartConfig = {
  id:    'salary-box',
  type:  'box',
  title: 'Salary Distribution by Department',

  xAxis: { field: 'department', title: 'Department',  axisType: 'category' },
  yAxis: { field: 'salary',     title: 'Salary ($)',   axisType: 'linear' },

  // One colour-coded box per department
  colorBy: 'department',

  enableSelection: true,
  selectionMode:   'box',
};
```

**Time-series box plot (using the derived `_timestamp` field):**
```typescript
const timeBoxConfig: ChartConfig = {
  id:    'created-box',
  type:  'box',
  title: 'Issue Creation Timeline by Department',

  xAxis: { field: 'department', axisType: 'category' },
  yAxis: { field: '_timestamp', title: 'Created (epoch ms)' },

  colorBy: 'department',
  enableSelection: true,
  selectionMode:   'box',
};
```

**Dynamic builder:**
```typescript
function makeBoxConfig(opts: {
  id:           string;
  categoryField: string;
  valueField:    string;
  colorBy?:      string;
  title?:        string;
}): ChartConfig {
  return {
    id:      opts.id,
    type:    'box',
    title:   opts.title,
    xAxis:   { field: opts.categoryField, axisType: 'category' },
    yAxis:   { field: opts.valueField },
    colorBy: opts.colorBy ?? opts.categoryField,
    enableSelection: true,
    selectionMode:   'box',
  };
}
```

**Key rules:**
- The library computes Q1, Q2 (median), Q3, IQR, whiskers (1.5 × IQR), and outliers from raw
  rows — your API does **not** need to pre-calculate statistics.
- `colorBy` creates one box per category (in both Plotly and D3 renderers).
- Without `colorBy`, a single box is drawn using all Y values.
- `xAxis` and `yAxis` are both required for a grouped box plot.

---

### 7.9 Sankey

**Purpose:** Visualise flows or proportions between two sets of categorical nodes. Link width is
proportional to the aggregated flow weight.

**Data shape:**
```
[{ priority: "high", status: "open",   _count: 1 }, ...]   // raw event rows
[{ region: "North",  category: "Electronics", sales: 15200 }] // pre-aggregated
```

**Minimal config:**
```typescript
const sankeyConfig: ChartConfig = {
  id:           'priority-status-sankey',
  type:         'sankey',
  xAxis:        { field: 'priority' },   // Source node field
  sankeyTarget: 'status',                // Target node field
  yAxis:        { field: '_count' },     // Flow weight field
  aggregation:  'sum',                   // How to combine rows sharing the same pair
};
```

**Full options:**
```typescript
const sankeyConfig: ChartConfig = {
  id:    'priority-status-sankey',
  type:  'sankey',
  title: 'Issue Flow: Priority → Status',

  // Source nodes (left column)
  xAxis: { field: 'priority', title: 'Priority' },

  // Target nodes (right column) — Sankey-only field
  sankeyTarget: 'status',

  // Flow weight per (source, target) pair after aggregation
  yAxis: { field: '_count', title: 'Count' },

  // 'sum' works for both _count (row-counting) and genuine numeric fields
  aggregation: 'sum',

  // Sankey diagrams visualise the full dataset — disable selection
  enableSelection: false,
};
```

**Dynamic builder:**
```typescript
function makeSankeyConfig(opts: {
  id:            string;
  sourceField:   string;
  targetField:   string;
  valueField:    string;
  aggregation?:  ChartConfig['aggregation'];
  sourceTitle?:  string;
  targetTitle?:  string;
  valueTitle?:   string;
  title?:        string;
}): ChartConfig {
  return {
    id:           opts.id,
    type:         'sankey',
    title:        opts.title,
    xAxis:        { field: opts.sourceField, title: opts.sourceTitle },
    sankeyTarget: opts.targetField,
    yAxis:        { field: opts.valueField,  title: opts.targetTitle },
    aggregation:  opts.aggregation ?? 'sum',
    enableSelection: false,
  };
}
```

**How the data is prepared internally:**

`buildSankeyData` (exported from `chart-lib` for custom use) groups all rows by the composite
key `(sourceField, targetField)`, applies the aggregation function within each group, deduplicates
all node names, and returns `{ nodes: [{ name }], links: [{ source, target, value }] }`.
Source names always appear first in the node list so they sit on the left column of the diagram.

**Key rules:**
- `xAxis.field`, `sankeyTarget`, and `yAxis.field` are all required.
- `aggregation` must be set — the library does **not** fall back to raw rows for Sankey, because
  multiple rows sharing the same `(source, target)` pair must be combined.
- `groupBy` is **not** used for Sankey — the two-field grouping is handled automatically by
  `buildSankeyData`.
- Use `_count` as `yAxis.field` with `aggregation: 'sum'` to count rows per `(source, target)`
  pair.
- Nodes can appear as both source and target (intermediate nodes) — the library handles this
  automatically.
- `enableSelection: false` is strongly recommended. Sankey shows aggregate flows across the full
  dataset; filtering the data changes the diagram itself rather than highlighting a slice.

---

## 8. Aggregation in depth

Aggregation pre-processes raw rows **before** they reach the renderer. It is controlled by two
config fields that must be set together:

| Field | Values | Description |
|---|---|---|
| `aggregation` | `'sum'` `'count'` `'mean'` `'median'` | How to combine values within each group |
| `groupBy` | any field name | The field to group rows by |

The aggregation result is a new set of rows with exactly two fields: `{ [groupBy]: value, [yAxis.field]: aggregatedValue }`. All other fields from the original rows are dropped.

### When to use each aggregation

| Scenario | `yAxis.field` | `aggregation` | Result |
|---|---|---|---|
| Count rows per category | `_count` | `'sum'` | Row count per group (since `_count = 1` per row) |
| Sum a numeric column | `revenue` | `'sum'` | Total revenue per group |
| Average a numeric column | `price` | `'mean'` | Average price per group |
| Median of a distribution | `salary` | `'median'` | Median salary per group |

### Aggregation + colorBy

`aggregation` + `groupBy` pre-aggregate into one row per group. If you then also set `colorBy`,
the aggregated rows are split into separate traces by `colorBy`. This means `groupBy` and
`colorBy` should be **the same field** when pre-aggregating:

```typescript
// Correct: group by region, then colour each bar by region
{
  aggregation: 'sum',
  groupBy:     'region',
  colorBy:     'region',
  yAxis:       { field: 'revenue' },
}

// Incorrect: groupBy and colorBy differ — colorBy tries to split already-aggregated rows
// by a field that no longer exists in the output
{
  aggregation: 'sum',
  groupBy:     'region',
  colorBy:     'category',   // ⚠ 'category' field is lost after aggregation
}
```

### Using `_count` for row counting

The `_count` derived field (always `1` per row) lets you count rows without needing a pre-counted
column from your API:

```typescript
// Count issues per status
{
  type:        'bar',
  xAxis:       { field: 'status' },
  yAxis:       { field: '_count' },
  aggregation: 'sum',      // sum of _count (1 per row) = row count
  groupBy:     'status',
}
```

---

## 9. Renderer selection

Both renderers consume the **same `ChartConfig`**. Switching requires only changing the `renderer`
prop on `<ChartProvider>`.

```tsx
import { ChartProvider, RENDERER_META } from 'chart-lib';
import type { RendererType } from 'chart-lib';

function Dashboard() {
  const [renderer, setRenderer] = useState<RendererType>('plotly');

  return (
    <>
      {/* Renderer picker */}
      <div>
        {(Object.values(RENDERER_META) as RendererMeta[]).map((meta) => (
          <button key={meta.type} onClick={() => setRenderer(meta.type)}>
            {meta.label}
          </button>
        ))}
      </div>

      {/* Key forces full remount when renderer changes, clearing selection state */}
      <ChartProvider key={renderer} renderer={renderer}>
        <DynamicChart config={barConfig}     data="/api/v1/data" style={{ height: 360 }} />
        <DynamicChart config={sankeyConfig}  data="/api/v1/data" style={{ height: 360 }} />
      </ChartProvider>
    </>
  );
}
```

### Renderer capability comparison

| Capability | Plotly | D3 |
|---|---|---|
| Line | ✅ | ✅ |
| Bar | ✅ | ✅ |
| Scatter | ✅ | ✅ |
| Pie / Donut | ✅ | ✅ |
| Area | ✅ | ✅ |
| Heatmap | ✅ | ✅ |
| Histogram | ✅ | ✅ |
| Box Plot | ✅ | ✅ |
| Sankey | ✅ | ✅ |
| Zoom / Pan | ✅ built-in | ❌ |
| Hover tooltips | ✅ built-in | ✅ SVG `<title>` |
| Box-select / lasso | ✅ built-in | ❌ |
| Download as PNG | ✅ via modebar | ❌ |
| Cross-filtering | ✅ | ✅ |
| Output | SVG / WebGL canvas | Pure SVG |

### When to choose each renderer

**Choose Plotly** when:
- You need zoom, pan, scroll-zoom, or downloadable PNG out of the box
- You want hover tooltips with rich formatting
- You want Plotly's built-in box-select and lasso tools for cross-filtering
- Your users expect an interactive, publication-quality chart experience

**Choose D3** when:
- You need maximum control over the SVG output for custom styling or animation
- You are embedding charts into a PDF or server-side rendered context
- Your design system requires exact pixel-level consistency
- You want to extend the chart with custom SVG overlays using standard D3 idioms

---

## 10. Data fetching & caching

### 10.1 JSON REST endpoints

Pass any URL string as the `data` prop. The library fetches it once and caches the result in a
module-level Map. Every chart sharing the same URL gets the same data from a **single network
request** — even if they mount at different times.

```tsx
// All three charts share one fetch — one request, one parse, one Zod validation
<ChartProvider>
  <DynamicChart config={lineConfig}    data="/api/v1/sales" style={{ height: 360 }} />
  <DynamicChart config={barConfig}     data="/api/v1/sales" style={{ height: 360 }} />
  <DynamicChart config={sankeyConfig}  data="/api/v1/sales" style={{ height: 360 }} />
</ChartProvider>
```

Your API must return either a bare `ChartDataRow[]` array or a `{ data: ChartDataRow[] }` wrapper.
Both are accepted and unwrapped automatically.

### 10.2 NDJSON streaming endpoints

Any URL whose last path segment **starts with** `stream` is automatically treated as an NDJSON
stream (newline-delimited JSON, one object per line):

```
/api/v1/events/stream          ← detected as stream
/api/v1/events/stream?from=7d  ← detected as stream (query string ignored)
/api/v1/events/latest          ← treated as regular JSON
```

The library opens a single `ReadableStream` reader, parses each line as it arrives, and notifies
all subscribed charts. Charts render incrementally as rows arrive — they do not wait for the
stream to close.

A pulsing **● LIVE** badge overlays each chart while the stream is open. Use `useStreamStatus`
to show stream progress in your own UI:

```tsx
import { useStreamStatus } from 'chart-lib';

function StreamBadge({ url }: { url: string }) {
  const { rowCount, streaming, firstRowMs } = useStreamStatus(url);

  if (firstRowMs === null) return null;

  return streaming
    ? <span>🔴 Streaming — {rowCount.toLocaleString()} rows so far</span>
    : <span>✅ Done — {rowCount.toLocaleString()} rows in {(firstRowMs / 1000).toFixed(2)}s</span>;
}
```

**Server requirements for NDJSON streaming:**
- `Content-Type: application/x-ndjson` (or `application/jsonlines`)
- Each line must be a valid JSON object matching `Record<string, string | number | null>`
- Lines are delimited by `\n`; a trailing newline is optional

### 10.3 Prefetching for faster first render

Call `prefetchData(url)` as early as possible — ideally when the user navigates to a page or
hovers over a tab — to warm the cache before any chart mounts. The function routes automatically
to the JSON or stream path based on `isStreamUrl(url)`:

```tsx
import { prefetchData, isStreamUrl } from 'chart-lib';
import { useEffect } from 'react';

function App() {
  const DATA_URL = '/api/v1/sales';

  // Start the fetch immediately when App mounts — before any chart renders
  useEffect(() => {
    prefetchData(DATA_URL);
  }, []);

  return (
    <ChartProvider>
      <DynamicChart config={lineConfig} data={DATA_URL} style={{ height: 360 }} />
      <DynamicChart config={barConfig}  data={DATA_URL} style={{ height: 360 }} />
    </ChartProvider>
  );
}
```

When the charts mount, the cache is already warm — they read data synchronously and skip the
loading spinner entirely.

### 10.4 Paginated APIs — Clinical Data pattern

For paginated endpoints that return a `{ data, meta }` envelope, `chart-lib`'s `ChartDataSchema`
automatically unwraps the envelope — so `DynamicChart` only ever receives a flat `ChartDataRow[]`.
You are responsible for constructing the correct URL, fetching pagination metadata, and evicting
stale pages from the cache before each navigation.

The demo app's **Clinical Data API** source illustrates the full pattern:

```tsx
import { ChartProvider, DynamicChart, prefetchData, clearDataCache } from 'chart-lib';
import { useState, useCallback, useEffect } from 'react';

interface PaginationMeta {
  page:          number;
  page_size:     number;
  total_records: number;
  total_pages:   number;
}

type ClinicalDomain = 'AE' | 'CM' | 'DM' | 'LB' | 'TV' | 'VS';

function buildClinicalUrl(
  domain: ClinicalDomain,
  page: number,
  pageSize: number,
  filters: Record<string, string>,
): string {
  const params = new URLSearchParams();
  params.set('page', String(page));
  params.set('page_size', String(pageSize));
  for (const [key, value] of Object.entries(filters)) {
    if (value.trim()) params.set(key, value.trim());
  }
  return `/api/v1/${domain.toLowerCase()}?${params.toString()}`;
}

const DOMAIN_CHART_CONFIGS: Record<ClinicalDomain, { config: object; label: string }[]> = {
  AE: [ /* 9 configs for Adverse Events domain */ ],
  DM: [ /* 9 configs for Demographics domain */ ],
  // ... other domains
};

function ClinicalDashboard() {
  const [domain,   setDomain]   = useState<ClinicalDomain>('AE');
  const [page,     setPage]     = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [filters,  setFilters]  = useState({ study: '', site: '', subject: '', visit: '', form: '' });
  const [meta,     setMeta]     = useState<PaginationMeta | null>(null);

  const dataUrl = buildClinicalUrl(domain, page, pageSize, filters);

  // Fetch pagination metadata whenever the URL changes
  useEffect(() => {
    fetch(dataUrl)
      .then((r) => r.json())
      .then((body) => 'meta' in body && setMeta(body.meta))
      .catch(() => {});
  }, [dataUrl]);

  const navigate = useCallback((newPage: number) => {
    clearDataCache(dataUrl);         // evict old page so a fresh fetch fires
    setPage(newPage);
  }, [dataUrl]);

  const handleDomainChange = useCallback((newDomain: ClinicalDomain) => {
    clearDataCache(dataUrl);
    setDomain(newDomain);
    setPage(1);                      // reset to page 1 on domain switch
    setMeta(null);
  }, [dataUrl]);

  // Debounce filter changes to avoid hammering the API on every keystroke
  const handleFilterChange = useCallback((field: string, value: string) => {
    setFilters((prev) => ({ ...prev, [field]: value }));
    setPage(1);                      // reset to page 1 when filter changes
  }, []);

  const configs = DOMAIN_CHART_CONFIGS[domain];

  return (
    <>
      {/* Domain selector */}
      {(['AE', 'CM', 'DM', 'LB', 'TV', 'VS'] as ClinicalDomain[]).map((d) => (
        <button key={d} onClick={() => handleDomainChange(d)}>{d}</button>
      ))}

      {/* Filter inputs */}
      {['study', 'site', 'subject', 'visit', 'form'].map((field) => (
        <input
          key={field}
          placeholder={field}
          value={filters[field as keyof typeof filters]}
          onChange={(e) => handleFilterChange(field, e.target.value)}
        />
      ))}

      {/* Pagination controls */}
      <button disabled={page <= 1} onClick={() => navigate(page - 1)}>← Prev</button>
      <input
        type="number" value={page}
        onBlur={(e) => navigate(Number(e.target.value))}
      />
      {meta && <span>/ {meta.total_pages} pages ({meta.total_records} records)</span>}
      <button disabled={meta !== null && page >= meta.total_pages} onClick={() => navigate(page + 1)}>
        Next →
      </button>
      <select value={pageSize} onChange={(e) => { clearDataCache(dataUrl); setPageSize(Number(e.target.value)); setPage(1); }}>
        {[50, 100, 200, 500, 1000].map((s) => <option key={s} value={s}>{s} / page</option>)}
      </select>

      {/* Charts re-keyed on domain so ChartProvider remounts on domain switch */}
      <ChartProvider key={domain}>
        {configs.map((c) => (
          <DynamicChart key={c.label} config={c.config} data={dataUrl} style={{ height: 360 }} />
        ))}
      </ChartProvider>
    </>
  );
}
```

**Key points:**

| Concern | Implementation |
|---|---|
| Envelope unwrap | `ChartDataSchema` already transforms `{ data: [...] }` — no adapter needed |
| Metadata | Fetch the same URL independently (or cache `meta` from `useChartData` options) |
| Cache eviction | Call `clearDataCache(oldUrl)` before `setPage` / `setDomain` / `setPageSize` |
| Filter debounce | Use `setTimeout` / `useDebounce` — always reset `page` to 1 |
| Domain remount | Pass `domain` as the `key` prop on `<ChartProvider>` to clear cross-filter state |

### 10.5 Cache invalidation

```typescript
import { clearDataCache } from 'chart-lib';

// Evict a single URL — next render will trigger a fresh fetch
clearDataCache('/api/v1/sales');

// Evict everything — use on logout or full dashboard refresh
clearDataCache();
```

Call `clearDataCache(url)` before user-triggered refreshes (e.g., a "Reload" button) to guarantee
a fresh network request even if the URL has not changed.

---

## 11. Cross-filtering

### 11.1 How it works

When any chart inside a `<ChartProvider>` is interacted with (clicked, box-selected, or lasso-
selected), it records the selected X-axis values into a shared `SelectionContext`. Every other
chart in the same provider automatically re-renders using only the rows whose X-axis field value
matches the selection. The source chart is always excluded from filtering — it always shows
its own full data.

Cross-filtering is **renderer-agnostic**:
- **Plotly** uses `plotly_selected` / `plotly_click` events via `setSelectionFromEvent`
- **D3** uses `click` handlers on SVG elements via `setSelectionByValues`

The `SelectionContext` stores the same `SelectionState` shape regardless of which renderer
produced it:

```typescript
interface SelectionState {
  sourceChartId:   string;                        // ID of the originating chart
  selectedIndices: number[];                      // Plotly point indices (empty for D3)
  selectedValues:  (string | number | null)[];    // The distinct X-axis values selected
  field:           string;                        // The xAxis.field used to filter siblings
}
```

### 11.2 Reading selection state

Use `useSelection()` from within any component inside a `<ChartProvider>`:

```tsx
import { useSelection } from 'chart-lib';

function SelectionPanel() {
  const { selection, clearSelection, filterData } = useSelection();

  if (!selection) return <p>No selection active — click a chart to cross-filter.</p>;

  return (
    <div>
      <p>
        <strong>Filtering by:</strong> {selection.field} ={' '}
        {selection.selectedValues.join(', ')}
      </p>
      <p>Originated from chart: <code>{selection.sourceChartId}</code></p>
      <button onClick={clearSelection}>Clear filter</button>
    </div>
  );
}
```

### 11.3 Programmatic selection and clearing

You can set a selection from your own UI (not from a chart interaction) using
`setSelectionByValues`:

```tsx
import { useSelection } from 'chart-lib';

function RegionFilter({ regions }: { regions: string[] }) {
  const { setSelectionByValues, clearSelection } = useSelection();

  return (
    <div>
      {regions.map((region) => (
        <button
          key={region}
          onClick={() =>
            // Cross-filter all charts to rows where 'region' === this value
            setSelectionByValues('region-filter-ui', 'region', [region])
          }
        >
          {region}
        </button>
      ))}
      <button onClick={clearSelection}>Show all</button>
    </div>
  );
}
```

The first argument to `setSelectionByValues` is the `sourceChartId` — charts whose `config.id`
matches this value will **not** be filtered (they are the "source"). Pass any unique string for a
non-chart selection origin (e.g., `'region-filter-ui'`).

**Clearing selection:**
```tsx
const { clearSelection } = useSelection();
clearSelection(); // All charts revert to full data
```

---

## 12. Dynamic dashboard — full worked example

This is a self-contained example of a dashboard where:
- The data API URL comes from a runtime prop
- Configs are built programmatically from known field names
- The renderer can be switched from a UI toggle
- Cross-filter state is exposed in a sidebar

```tsx
// src/SalesDashboard.tsx

import React, { useState, useEffect, useCallback } from 'react';
import {
  ChartProvider,
  DynamicChart,
  useSelection,
  prefetchData,
  clearDataCache,
  RENDERER_META,
} from 'chart-lib';
import type { ChartConfig, RendererType } from 'chart-lib';

// ── 1. Build all configs in code ───────────────────────────────────────────

function buildDashboardConfigs(): ChartConfig[] {
  return [
    // Line — monthly revenue, one trace per region
    {
      id:    'revenue-trend',
      type:  'line',
      title: 'Monthly Revenue by Region',
      xAxis: { field: 'month',   title: 'Month',   axisType: 'date' },
      yAxis: { field: 'revenue', title: 'Revenue ($)' },
      colorBy: 'region',
      enableSelection: true,
      selectionMode:   'box',
    },

    // Bar — total revenue per region, pre-aggregated
    {
      id:          'revenue-by-region',
      type:        'bar',
      title:       'Revenue by Region',
      xAxis:       { field: 'region',  title: 'Region', axisType: 'category' },
      yAxis:       { field: 'revenue', title: 'Revenue ($)' },
      aggregation: 'sum',
      groupBy:     'region',
      colorBy:     'region',
      enableSelection: true,
      selectionMode:   'box',
    },

    // Scatter — volume vs revenue, colour-coded by category
    {
      id:    'vol-rev-scatter',
      type:  'scatter',
      title: 'Volume vs Revenue',
      xAxis: { field: 'units',   title: 'Units Sold', axisType: 'linear' },
      yAxis: { field: 'revenue', title: 'Revenue ($)', axisType: 'linear' },
      marker: { colorField: 'category', size: 9, opacity: 0.7 },
      enableSelection: true,
      selectionMode:   'lasso',
    },

    // Pie — revenue share by product category
    {
      id:          'category-pie',
      type:        'pie',
      title:       'Revenue Share by Category',
      xAxis:       { field: 'category' },
      yAxis:       { field: 'revenue' },
      aggregation: 'sum',
      groupBy:     'category',
      enableSelection: false,
    },

    // Area — monthly revenue by category, stacked
    {
      id:         'revenue-area',
      type:       'area',
      title:      'Monthly Revenue by Category (Stacked)',
      xAxis:      { field: 'month',    title: 'Month',      axisType: 'date' },
      yAxis:      { field: 'revenue',  title: 'Revenue ($)' },
      colorBy:    'category',
      stackGroup: 'one',
      enableSelection: true,
      selectionMode:   'box',
    },

    // Heatmap — revenue intensity: region × category
    {
      id:    'region-cat-heatmap',
      type:  'heatmap',
      title: 'Revenue: Region × Category',
      xAxis: { field: 'region',   title: 'Region' },
      yAxis: { field: 'category', title: 'Category' },
      zAxis: { field: 'revenue',  title: 'Revenue ($)' },
      marker: { colorScale: 'YlOrRd' },
      enableSelection: false,
    },

    // Histogram — unit distribution by category
    {
      id:      'units-hist',
      type:    'histogram',
      title:   'Unit Distribution by Category',
      xAxis:   { field: 'units', title: 'Units Sold' },
      colorBy: 'category',
      marker:  { opacity: 0.75 },
      enableSelection: true,
      selectionMode:   'box',
    },

    // Box — revenue spread per region
    {
      id:      'revenue-box',
      type:    'box',
      title:   'Revenue Distribution by Region',
      xAxis:   { field: 'region',  title: 'Region',       axisType: 'category' },
      yAxis:   { field: 'revenue', title: 'Revenue ($)',   axisType: 'linear' },
      colorBy: 'region',
      enableSelection: true,
      selectionMode:   'box',
    },

    // Sankey — flow from region to product category, weighted by revenue
    {
      id:           'region-cat-sankey',
      type:         'sankey',
      title:        'Revenue Flow: Region → Category',
      xAxis:        { field: 'region',   title: 'Region' },
      sankeyTarget: 'category',
      yAxis:        { field: 'revenue',  title: 'Revenue ($)' },
      aggregation:  'sum',
      enableSelection: false,
    },
  ];
}

// ── 2. Selection sidebar ───────────────────────────────────────────────────

function SelectionSidebar() {
  const { selection, clearSelection } = useSelection();

  return (
    <aside style={{ width: 220, padding: 16, background: '#f9f9f9' }}>
      <h3 style={{ margin: '0 0 8px' }}>Active Filter</h3>
      {selection ? (
        <>
          <p style={{ fontSize: 13 }}>
            <strong>Field:</strong> {selection.field}<br />
            <strong>Values:</strong> {selection.selectedValues.join(', ')}
          </p>
          <button onClick={clearSelection}>✕ Clear</button>
        </>
      ) : (
        <p style={{ fontSize: 13, color: '#888' }}>
          Click or select a chart to cross-filter all others.
        </p>
      )}
    </aside>
  );
}

// ── 3. Inner dashboard (must be inside ChartProvider) ──────────────────────

function DashboardInner({ dataUrl }: { dataUrl: string }) {
  const configs = buildDashboardConfigs();

  return (
    <div style={{ display: 'flex', gap: 16 }}>
      {/* Chart grid */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {configs.map((cfg) => (
          <div key={cfg.id} style={{ background: '#fff', borderRadius: 8, padding: 8, boxShadow: '0 1px 4px rgba(0,0,0,.1)' }}>
            <DynamicChart
              config={cfg}
              data={dataUrl}
              style={{ height: 300 }}
            />
          </div>
        ))}
      </div>

      {/* Selection sidebar */}
      <SelectionSidebar />
    </div>
  );
}

// ── 4. Top-level dashboard with renderer toggle ────────────────────────────

interface SalesDashboardProps {
  dataUrl: string;   // e.g. '/api/v1/sales' or '/api/v1/sales/stream'
}

export function SalesDashboard({ dataUrl }: SalesDashboardProps) {
  const [renderer, setRenderer] = useState<RendererType>('plotly');

  // Prefetch data as soon as the component mounts
  useEffect(() => {
    prefetchData(dataUrl);
  }, [dataUrl]);

  // When dataUrl changes, evict the old URL from cache and prefetch the new one
  useEffect(() => {
    return () => { clearDataCache(dataUrl); };
  }, [dataUrl]);

  return (
    <div>
      {/* Renderer toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {Object.values(RENDERER_META).map((meta) => (
          <button
            key={meta.type}
            onClick={() => setRenderer(meta.type as RendererType)}
            style={{
              fontWeight: renderer === meta.type ? 700 : 400,
              borderBottom: renderer === meta.type ? `2px solid ${meta.color}` : 'none',
            }}
          >
            {meta.label}
          </button>
        ))}
      </div>

      {/* Key forces full remount when renderer changes */}
      <ChartProvider key={`${renderer}-${dataUrl}`} renderer={renderer}>
        <DashboardInner dataUrl={dataUrl} />
      </ChartProvider>
    </div>
  );
}
```

**Usage in your app:**

```tsx
// Regular JSON endpoint
<SalesDashboard dataUrl="/api/v1/sales?from=2025-01&to=2025-12" />

// NDJSON stream endpoint
<SalesDashboard dataUrl="/api/v1/sales/stream" />
```

---

## 13. Config validation at runtime

`ChartConfigSchema` (a Zod schema) is exported so you can validate configs received from your own
backend config API before rendering:

```typescript
import { ChartConfigSchema } from 'chart-lib';
import type { ChartConfig } from 'chart-lib';

async function fetchAndValidateConfig(url: string): Promise<ChartConfig> {
  const res  = await fetch(url);
  const json = await res.json() as unknown;

  const result = ChartConfigSchema.safeParse(json);

  if (!result.success) {
    // result.error is a ZodError with detailed field-level messages
    console.error('Invalid chart config from API:', result.error.flatten());
    throw new Error(`Config validation failed: ${result.error.message}`);
  }

  return result.data;
}
```

`DynamicChart` itself runs this validation when `config` is passed as a URL string. When you
pass an inline object, Zod still validates it on every render via `useMemo` — type errors that
TypeScript misses at compile time (e.g., a wrong `type` value coming from user input) are caught
and shown as an in-card error rather than crashing the page.

### Config validation on a config-builder API response

If your backend serves chart configurations from a database (where users have configured their own
dashboards), validate every config before rendering:

```typescript
async function fetchDashboardConfigs(dashboardId: string): Promise<ChartConfig[]> {
  const res  = await fetch(`/api/v1/dashboards/${dashboardId}/configs`);
  const json = await res.json() as unknown[];

  return json.flatMap((raw) => {
    const parsed = ChartConfigSchema.safeParse(raw);
    if (parsed.success) return [parsed.data];
    console.warn('Skipping invalid config:', parsed.error.flatten().fieldErrors);
    return [];   // skip invalid configs rather than breaking the entire dashboard
  });
}
```

---

## 14. Authentication & custom fetch headers

The library uses the browser's native `fetch` internally. `useChartData` and `useChartConfig`
do not currently accept a `fetchOptions` parameter. To pass authentication headers (Bearer
tokens, cookies, API keys), use one of these approaches:

### Option A — Use cookies (preferred for same-origin APIs)

Configure your backend to set a `HttpOnly` session cookie. The browser sends it automatically
with every `fetch` request to the same origin — no code changes needed.

### Option B — Proxy through Vite (development)

In `vite.config.ts`, proxy `/api` to your backend. Vite injects any required headers or
re-writes the origin at the proxy layer:

```typescript
// vite.config.ts
export default {
  server: {
    proxy: {
      '/api': {
        target: 'https://my-backend.internal',
        changeOrigin: true,
        headers: { 'X-Internal-Token': process.env.INTERNAL_TOKEN ?? '' },
      },
    },
  },
};
```

### Option C — Fork useChartData (advanced)

If you need per-request Bearer tokens, copy `useChartData.ts` into your project, add a
`headers` parameter to `getOrFetchData`, and pass it through every `fetch` call:

```typescript
// In your fork of useChartData.ts
async function getOrFetchData(
  url: string,
  headers: Record<string, string>,
): Promise<ChartDataRow[]> {
  const res = await fetch(url, { headers });
  // ... rest unchanged
}
```

Then pass your auth token from a React context or Zustand store into the hook before calling
`DynamicChart` with inline data.

---

## 15. Error handling

### In-card error display

`DynamicChart` never throws. All errors — network failures, Zod validation failures, missing
required config fields — are caught and rendered as a styled error card in place of the chart:

```
┌──────────────────────────────────────────────┐
│  Chart error:                                │
│  Failed to fetch data: 403 Forbidden         │
└──────────────────────────────────────────────┘
```

### Detecting errors in your own code

If you need to react to errors (e.g., show a global toast), validate inline before passing to
`DynamicChart`:

```typescript
const result = ChartConfigSchema.safeParse(rawConfig);
if (!result.success) {
  showToast(`Invalid chart config: ${result.error.issues[0]?.message}`);
  return null;
}
// result.data is type-safe ChartConfig
return <DynamicChart config={result.data} data={dataUrl} />;
```

### Common validation errors and causes

| Error message fragment | Cause | Fix |
|---|---|---|
| `"Invalid enum value ... type"` | `type` field has an unsupported chart type string | Ensure `type` is one of the 9 valid values |
| `"Required"` on `xAxis` | `xAxis` is missing entirely | Every config must include `xAxis` |
| `"Expected number, received string"` on `marker.opacity` | Opacity passed as `"0.7"` (string) instead of `0.7` (number) | Parse to float before building the config |
| `"Failed to fetch data: 404"` | Data URL returns 404 | Check the API endpoint path and any Vite proxy config |
| `"Unexpected token"` | API returns HTML error page instead of JSON | The API is returning an error — check the network tab |

---

## 16. Performance guide

### Share one data URL across all charts

All charts pointing at the same URL share a single fetch and a single Zod parse. Avoid using
different URLs for the same dataset:

```tsx
// ✅ One fetch, all charts get data simultaneously
const DATA_URL = '/api/v1/sales';
<DynamicChart config={lineConfig}  data={DATA_URL} />
<DynamicChart config={barConfig}   data={DATA_URL} />
<DynamicChart config={pieConfig}   data={DATA_URL} />

// ❌ Three fetches, three Zod validations
<DynamicChart config={lineConfig}  data="/api/v1/sales?chart=line" />
<DynamicChart config={barConfig}   data="/api/v1/sales?chart=bar"  />
<DynamicChart config={pieConfig}   data="/api/v1/sales?chart=pie"  />
```

### Prefetch before mounting

Call `prefetchData(url)` when the user navigates to a route or hovers over a navigation item,
before the chart components mount. This eliminates the loading flash entirely:

```typescript
// React Router v6 example — prefetch on link hover
<Link
  to="/dashboard/sales"
  onMouseEnter={() => prefetchData('/api/v1/sales')}
>
  Sales Dashboard
</Link>
```

### Memoize config objects

Config objects are compared by reference inside `DynamicChart`. If you build configs inline in
JSX, wrap them in `useMemo` to avoid unnecessary re-renders:

```tsx
// ❌ New object on every render → DynamicChart re-processes config each time
<DynamicChart config={{ id: 'bar', type: 'bar', xAxis: { field: 'status' }, ... }} data={url} />

// ✅ Stable reference — only recomputes when dependencies change
const barConfig = useMemo<ChartConfig>(() => ({
  id:    'status-bar',
  type:  'bar',
  xAxis: { field: selectedXField, axisType: 'category' },
  yAxis: { field: selectedYField },
  aggregation: 'sum',
  groupBy:     selectedXField,
}), [selectedXField, selectedYField]);

<DynamicChart config={barConfig} data={url} />
```

### Use aggregation on large datasets

If your API returns raw event rows (thousands of records), use `aggregation` + `groupBy` to
reduce the data client-side rather than requesting pre-aggregated data from the server. This
keeps the API simple while keeping render performance fast:

```typescript
// API returns 50 000 raw order rows
// The library pre-aggregates to one row per (region, month) before rendering
{
  aggregation: 'sum',
  groupBy:     'region',
  yAxis:       { field: 'revenue' },
}
```

### Limit chart count per provider

Each chart subscribes to the `SelectionContext` via `useMemo`. In practice, 10–15 charts in one
`<ChartProvider>` is comfortable. Beyond 20, consider splitting into multiple independent
providers or virtualizing the chart grid.

---

## 17. Troubleshooting

### Chart shows nothing / blank area

1. **Check the container height.** `DynamicChart` fills its parent. If the parent has no height
   the chart renders at 0 px. Always set an explicit `height` on the container or via `style`:
   ```tsx
   <DynamicChart config={cfg} data={url} style={{ height: 360 }} />
   ```
2. **Check the browser Network tab.** Confirm the data URL returns 200 with valid JSON.
3. **Check the console.** Zod validation errors are logged as warnings.

### D3 Sankey renders nothing

The most common cause is that `sankeyTarget` is missing or the `(source, target)` pairs in the
data all have empty string values after coercion. Verify:
- `sankeyTarget` is set in the config (not just `groupBy`)
- `xAxis.field` and `sankeyTarget` both refer to columns that contain non-empty string values
- `yAxis.field` refers to a column with numeric values (or use `_count` with `aggregation: 'sum'`)

### Charts update on every render

Your `config` prop is a new object on every render. Wrap it in `useMemo` — see §16.

### Cross-filter does not affect a chart

Check that:
1. The chart is inside the **same `<ChartProvider>`** as the chart that originated the selection.
   Charts in different providers are isolated.
2. `enableSelection: true` is set on the source chart (the one being interacted with).
3. The `xAxis.field` on the source chart matches a field name that exists in all target charts'
   data — cross-filter matches by comparing `row[sourceField]` against `selectedValues`.

### Data is stale after an API update

The library caches by URL. If your API changes its response without changing the URL, call
`clearDataCache(url)` before triggering a re-render:

```typescript
async function refreshData() {
  clearDataCache('/api/v1/sales');
  // Re-render will trigger a fresh fetch
  setRefreshKey(k => k + 1);
}
```

### Aggregation produces unexpected results

- Confirm `groupBy` and `aggregation` are both set. Either alone has no effect.
- `aggregation: 'count'` counts the number of **rows** in each group (not the sum of a column).
  Use `aggregation: 'sum'` with `yAxis.field: '_count'` to count rows while summing `_count = 1`.
- After aggregation, only the `groupBy` field and `yAxis.field` are preserved in the output rows.
  `colorBy` must equal `groupBy` if you want per-group colouring after aggregation.

---

## 18. Full ChartConfig field reference

| Field | Type | Required | Default | Applies to |
|---|---|---|---|---|
| `id` | `string` | ✅ | — | all |
| `type` | `'line'\|'bar'\|'scatter'\|'pie'\|'area'\|'heatmap'\|'histogram'\|'box'\|'sankey'` | ✅ | — | all |
| `xAxis.field` | `string` | ✅ | — | all (source nodes for Sankey) |
| `xAxis.title` | `string` | — | — | all |
| `xAxis.axisType` | `'linear'\|'log'\|'date'\|'category'` | — | — | all except heatmap |
| `title` | `string` | — | — | all |
| `yAxis.field` | `string` | ✅ for most | — | all except histogram (flow weight for Sankey) |
| `yAxis.title` | `string` | — | — | all |
| `yAxis.axisType` | `'linear'\|'log'\|'date'\|'category'` | — | — | line, bar, scatter, area, box |
| `zAxis.field` | `string` | ✅ for heatmap | — | heatmap only |
| `zAxis.title` | `string` | — | — | heatmap only |
| `sankeyTarget` | `string` | ✅ for sankey | — | sankey only — target node column |
| `marker.colorField` | `string` | — | — | scatter, histogram |
| `marker.sizeField` | `string` | — | — | scatter |
| `marker.colorScale` | see below | — | `'Viridis'` | heatmap, scatter (with numeric colorField) |
| `marker.opacity` | `number` 0–1 | — | — | histogram, scatter |
| `marker.size` | `number` > 0 | — | — | scatter |
| `layout` | `Record<string,unknown>` | — | — | Plotly renderer only — merged last |
| `enableSelection` | `boolean` | — | `true` | all |
| `selectionMode` | `'box'\|'lasso'` | — | `'box'` | Plotly renderer only |
| `aggregation` | `'sum'\|'count'\|'mean'\|'median'` | — | — | all except histogram (required for sankey) |
| `groupBy` | `string` | — | — | line, bar, scatter, pie, area, box (not sankey) |
| `stackGroup` | `string` | — | — | area only |
| `colorBy` | `string` | — | — | line, bar, scatter, area, histogram, box |

**Color scale values:** `'Viridis'` · `'Plasma'` · `'Blues'` · `'Reds'` · `'Greens'` · `'YlOrRd'` · `'RdBu'` · `'Picnic'`

---

*End of Developer Integration Guide*