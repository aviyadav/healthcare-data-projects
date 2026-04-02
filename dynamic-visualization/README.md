# dynamic-visualization

A Bun monorepo that delivers a **dynamic, embeddable React visualization utility**. Chart type, axis mappings, styling, and data are all driven by a single `ChartConfig` JSON schema — no component code changes required to switch chart types, data sources, or even the entire rendering library.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Compile](#compile)
- [Test](#test)
- [Run the Demo App](#run-the-demo-app)
- [Packages](#packages)
  - [chart-lib](#chart-lib)
  - [demo](#demo)
- [Configuration API](#configuration-api)
  - [ChartConfig Schema](#chartconfig-schema)
  - [ChartData Schema](#chartdata-schema)
  - [Chart Type Reference](#chart-type-reference)
- [Component API](#component-api)
  - [DynamicChart](#dynamicchart)
  - [ChartProvider](#chartprovider)
  - [useSelection](#useselection)
  - [useRenderer](#userenderer)
  - [Renderer Utilities](#renderer-utilities)
  - [Data Cache Utilities](#data-cache-utilities)
  - [Low-level Adapters](#low-level-adapters)
- [Rendering Libraries](#rendering-libraries)
  - [Plotly (default)](#plotly-default)
  - [D3.js](#d3js)
  - [Renderer Comparison](#renderer-comparison)
- [Cross-filtering](#cross-filtering)
- [Data Sources](#data-sources)
  - [Shared-fetch / Stream Cache](#shared-fetch--stream-cache)
  - [Issues API Source](#issues-api-source)
  - [Clinical Data API Source](#clinical-data-api-source)
  - [Sample File Source](#sample-file-source)
- [Demo App Features](#demo-app-features)
- [Embedding in a Web Application](#embedding-in-a-web-application)
- [Switching from Mock Files to Real APIs](#switching-from-mock-files-to-real-apis)
- [Technology Decisions](#technology-decisions)

---

## Overview

The utility solves the problem of rendering different chart types from a single reusable component, across multiple rendering libraries, against multiple data sources — all without touching component code.

You declare **what** to show in a `ChartConfig` object (or a URL to one), and the library figures out **how** to show it using whichever rendering backend is active. The same config JSON drives Plotly and D3.js equally.

```
ChartConfig (unchanged schema — single source of truth)
       │
       ▼
 DynamicChart  ─── resolves config + data, applies cross-filter
       │
       ▼  RendererContext.renderer
  ┌────┴───────────────────────────────────────────────┐
  │ 'plotly' → PlotlyRenderer  (react-plotly.js)       │
  │ 'd3'     → D3Renderer      (d3 raw SVG + d3-sankey) │
  └────────────────────────────────────────────────────┘
```

Multiple `DynamicChart` instances sharing a `<ChartProvider>` automatically participate in **cross-filtering**: selecting data in one chart filters all sibling charts, regardless of which renderer each chart uses.

---

## Key Features

| Feature | Description |
|---|---|
| **Config-driven charts** | Chart type, axes, colours, aggregation — all from a JSON config or URL |
| **2 rendering libraries** | Plotly.js and D3.js — switch at runtime with one prop |
| **9 chart types** | Line, Bar, Scatter, Pie/Donut, Area, Heatmap, Histogram, Box Plot, **Sankey** |
| **Cross-filtering** | Selection in one chart filters all sibling charts automatically |
| **Per-chart reset** | Reset button per card clears selection and restores default zoom |
| **Global reset** | "Reset All Charts" clears every chart simultaneously |
| **3 data sources** | Streaming Issues API, Clinical Data API (6 CDISC SDTM domains, paginated), or static sample-data.json |
| **NDJSON streaming** | Issues API streams rows via `ReadableStream`; charts render live as rows arrive |
| **Single-fetch cache** | All 9 charts sharing a URL make exactly one network request (stream or JSON) |
| **Prefetch on switch** | Data fetch/stream starts before charts mount — near-instant render on switch |
| **Live streaming badge** | Pulsing **LIVE** overlay on each chart while the stream is still open |
| **Loading gate** | One banner replaces 9 individual spinners; all charts appear on first rows |
| **Runtime validation** | Zod schemas validate config and data; errors shown in-card, never crash |

---

## Repository Structure

```
dynamic-visualization/
├── package.json                    # Root — workspace scripts, shared devDeps
├── bunfig.toml                     # Bun workspace & runtime configuration
├── tsconfig.base.json              # Shared strict TypeScript config
│
├── packages/
│   ├── chart-lib/                  # Publishable React component library
│   │   ├── src/
│   │   │   ├── adapters/
│   │   │   │   └── plotlyAdapter.ts        # buildTraces + buildLayout (Plotly)
│   │   │   ├── components/
│   │   │   │   ├── DynamicChart.tsx        # Routing shell — picks renderer
│   │   │   │   └── ChartProvider.tsx       # renderer + cross-filter wrapper
│   │   │   ├── context/
│   │   │   │   ├── SelectionContext.tsx    # Cross-filter state
│   │   │   │   └── RendererContext.tsx     # Active renderer type
│   │   │   ├── hooks/
│   │   │   │   ├── useChartConfig.ts       # Config fetch + Zod validation
│   │   │   │   └── useChartData.ts         # Data fetch/stream + shared cache + pub/sub
│   │   │   ├── renderers/
│   │   │   │   ├── PlotlyRenderer.tsx      # Plotly.js renderer
│   │   │   │   └── D3Renderer.tsx          # D3.js SVG renderer (all 9 types)
│   │   │   ├── types/
│   │   │   │   ├── ChartConfig.ts          # Zod schema + ChartConfig type
│   │   │   │   └── ChartData.ts            # Zod schema + ChartDataRow type
│   │   │   ├── utils/
│   │   │   │   └── dataUtils.ts            # Shared: aggregate, groupByField,
│   │   │   │                               #   col, PALETTE, applyAggregation,
│   │   │   │                               #   buildSankeyData
│   │   │   └── index.ts                    # Public exports
│   │   ├── dist/                           # Built artefacts (generated)
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── vite.config.ts                  # Library build (ESM + CJS)
│   │   │
│   └── demo/                       # Vite demo application
│       ├── src/
│       │   ├── components/
│       │   │   └── ChartCard.tsx           # Card wrapper with reset button
│       │   ├── config/
│       │   │   ├── api/                    # Configs for Issues API source
│       │   │   │   ├── line-chart.json
│       │   │   │   ├── bar-chart.json
│       │   │   │   ├── scatter-chart.json
│       │   │   │   ├── pie-chart.json
│       │   │   │   ├── area-chart.json
│       │   │   │   ├── heatmap-chart.json
│       │   │   │   ├── histogram-chart.json
│       │   │   │   ├── box-chart.json
│       │   │   │   └── sankey-chart.json
│       │   │   ├── clinical/               # Configs for Clinical Data API source
│       │   │   │   ├── ae/                 # Adverse Events — 9 chart configs
│       │   │   │   ├── cm/                 # Concomitant Medications — 9 chart configs
│       │   │   │   ├── dm/                 # Demographics — 9 chart configs
│       │   │   │   ├── lb/                 # Laboratory Results — 9 chart configs
│       │   │   │   ├── tv/                 # Trial Visits — 9 chart configs
│       │   │   │   └── vs/                 # Vital Signs — 9 chart configs
│       │   │   └── file/                   # Configs for sample-data source
│       │   │       ├── line-chart.json
│       │   │       ├── bar-chart.json
│       │   │       ├── scatter-chart.json
│       │   │       ├── pie-chart.json
│       │   │       ├── area-chart.json
│       │   │       ├── heatmap-chart.json
│       │   │       ├── histogram-chart.json
│       │   │       ├── box-chart.json
│       │   │       └── sankey-chart.json
│       │   ├── data/
│       │   │   └── sample-data.json        # 24-row sales dataset
│       │   ├── App.tsx                     # Main page — source + renderer picker
│       │   └── main.tsx
│       ├── index.html
│       ├── package.json
│       ├── tsconfig.json
│       └── vite.config.ts
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Bun | ≥ 1.1 (install via `curl -fsSL https://bun.sh/install \| bash`) |

Verify:

```sh
bun --version
```

---

## Quick Start

```sh
# 1. Enter the repo
cd dynamic-visualization-v3

# 2. Install all workspace dependencies (single lockfile)
bun install

# 3. Start the demo app
bun dev
# → http://localhost:5173
```

---

## Compile

### Build the component library

Produces ESM (`.mjs`), CommonJS (`.cjs`), and TypeScript declarations inside `packages/chart-lib/dist/`:

```sh
bun build
# or explicitly:
bun run --filter chart-lib build
```

Build output:

```
packages/chart-lib/dist/
├── index.mjs              # ES module bundle
├── index.cjs              # CommonJS bundle
└── types/
    ├── index.d.ts
    ├── adapters/
    │   └── plotlyAdapter.d.ts
    ├── components/
    │   ├── DynamicChart.d.ts
    │   └── ChartProvider.d.ts
    ├── context/
    │   ├── SelectionContext.d.ts
    │   └── RendererContext.d.ts
    ├── hooks/
    │   ├── useChartConfig.d.ts
    │   └── useChartData.d.ts
    ├── renderers/
    │   ├── PlotlyRenderer.d.ts
    │   └── D3Renderer.d.ts
    ├── types/
    │   ├── ChartConfig.d.ts
    │   └── ChartData.d.ts
    └── utils/
        └── dataUtils.d.ts
```

**Peer dependencies** (not bundled — must be provided by the consuming application):
`react`, `react-dom`, `plotly.js`, `react-plotly.js`

**Bundled dependencies** (externalized in the Rollup output but declared as `dependencies`, so auto-installed with the package):
`d3`, `d3-sankey`, `zod`

### Build the demo app (static production bundle)

```sh
bun run --filter demo build
# output → packages/demo/dist/
```

Preview the production build locally:

```sh
bun run --filter demo preview
# → http://localhost:4173
```

### TypeScript type-check only (no emit)

```sh
bun lint
```

---

## Test

Unit tests live in `packages/chart-lib/src/__tests__/` and run with [Bun's built-in test runner](https://bun.sh/docs/cli/test).

```sh
# Run all tests once
bun test
# or explicitly:
bun run --filter chart-lib test

# Watch mode (re-runs on file change)
bun run --filter chart-lib test:watch
```

### Test coverage

| Test file | What is tested |
|---|---|
| `plotlyAdapter.test.ts` | `buildTraces` — one assertion per chart type for Plotly `type`, `mode`, `fill`, axis field mapping; `colorBy` trace splitting; `aggregation: sum`. `buildLayout` — title, `dragmode`, `clickmode`, `barmode`, axis titles, layout overrides. |
| `filterData.test.ts` | `filterData` — no-selection passthrough, field matching, empty-match result, numeric coercion, null handling. |

**24 / 24 tests pass.**

---

## Run the Demo App

```sh
bun dev
# → http://localhost:5173
```

The demo presents a header with two selectors:

1. **Library** — choose the rendering engine (Plotly / D3.js)
2. **Data Source** — choose between the streaming Issues API, the Clinical Data API, or the static sample-data file

Below the header, a responsive grid shows all 9 chart types, all sharing one `<ChartProvider>`. An **Event Log** panel on the right shows click and select events in real time.

---

## Packages

### chart-lib

> `packages/chart-lib/`

The publishable library. True peer dependencies (consumers must install these separately):

```json
{
  "react": "^18",
  "react-dom": "^18",
  "plotly.js": "^2.35",
  "react-plotly.js": "^2.6"
}
```

Bundled runtime dependencies (externalized in the build output, auto-installed via `dependencies`):
`d3 ^7`, `d3-sankey ^0.12`, `zod ^3`

#### Scripts

| Script | Command | Purpose |
|---|---|---|
| `build` | `vite build && tsc --emitDeclarationOnly` | ESM + CJS + `.d.ts` |
| `test` | `vitest run` | Run unit tests once |
| `test:watch` | `vitest` | Watch mode |

---

### demo

> `packages/demo/`

A Vite + React app that exercises the library. Uses either local JSON files (served as Vite static assets) or a live REST API. All chart configs live in `src/config/{api,file}/`.

#### Scripts

| Script | Command | Purpose |
|---|---|---|
| `dev` | `vite` | Start dev server with HMR |
| `build` | `vite build` | Production static bundle |
| `preview` | `vite preview` | Serve production bundle locally |

---

## Configuration API

### ChartConfig Schema

A `ChartConfig` is a plain JSON object (or TypeScript value) that fully describes a chart. It is **rendering-library-agnostic**: the same config drives Plotly and D3 equally.

```ts
type ChartConfig = {
  // Required
  id: string;                        // Unique ID — used for cross-filter tracking
  type: ChartType;                   // See chart type table below
  xAxis: AxisConfig;                 // Primary horizontal axis (source nodes for Sankey)

  // Optional
  title?: string;
  yAxis?: AxisConfig;                // Required for all types except histogram
  zAxis?: AxisConfig;                // Heatmap only — maps to colour intensity
  sankeyTarget?: string;             // Sankey only — field whose values become target nodes
  marker?: MarkerConfig;
  layout?: Record<string, unknown>;  // Plotly layout overrides (merged last, Plotly only)
  enableSelection?: boolean;         // default: true
  selectionMode?: 'box' | 'lasso';  // default: 'box'
  aggregation?: 'sum' | 'count' | 'mean' | 'median';
  groupBy?: string;                  // Field to aggregate/group by (used with aggregation)
  stackGroup?: string;               // Area chart stacking group name
  colorBy?: string;                  // Split traces by distinct values of this field
};

type AxisConfig = {
  field: string;                     // Data field name to map to this axis
  title?: string;                    // Axis label displayed on the chart
  axisType?: 'linear' | 'log' | 'date' | 'category';
};

type MarkerConfig = {
  colorField?: string;               // Data field to drive point colour
  sizeField?: string;                // Data field to drive point size
  colorScale?: 'Viridis' | 'Plasma' | 'Blues' | 'Reds' | 'Greens' | 'YlOrRd' | 'RdBu' | 'Picnic';
  opacity?: number;                  // 0–1
  size?: number;                     // Fixed marker size in pixels
};
```

**Example — bar chart:**

```json
{
  "id": "bar-sales",
  "type": "bar",
  "title": "Sales by Region",
  "xAxis": { "field": "region", "title": "Region", "axisType": "category" },
  "yAxis": { "field": "sales", "title": "Total Sales ($)" },
  "aggregation": "sum",
  "groupBy": "region",
  "colorBy": "region",
  "enableSelection": true,
  "selectionMode": "box"
}
```

**Example — Sankey diagram:**

```json
{
  "id": "sankey-flow",
  "type": "sankey",
  "title": "Sales Flow: Region → Category",
  "xAxis": { "field": "region", "title": "Region" },
  "sankeyTarget": "category",
  "yAxis": { "field": "sales", "title": "Sales ($)" },
  "aggregation": "sum",
  "enableSelection": false
}
```

For a Sankey chart:
- `xAxis.field` defines the **source** node column
- `sankeyTarget` defines the **target** node column
- `yAxis.field` defines the numeric **flow weight** column
- `aggregation` controls how multiple rows sharing the same `(source, target)` pair are combined (typically `"sum"` or `"count"`)

These config JSON files work unchanged when you switch the rendering library from Plotly to D3.

---

### ChartData Schema

Chart data is an **array of flat row objects**. Each value is a `string`, `number`, or `null`.

```ts
type ChartDataRow = Record<string, string | number | null>;
type ChartData = ChartDataRow[];
```

The library also accepts the API-wrapper shape `{ data: ChartDataRow[] }` and unwraps it automatically.

**Example:**

```json
[
  { "date": "2025-01", "region": "North", "sales": 15200, "volume": 152, "price": 100 },
  { "date": "2025-01", "region": "South", "sales": 9800,  "volume": 196, "price": 50  }
]
```

The following **derived fields** are automatically added to every row when the source looks like the Issues API:

| Derived field | Source field | Description |
|---|---|---|
| `_count` | — | Always `1`; enables `aggregation: "count"` on any `groupBy` |
| `_timestamp` | `created_ts` | Epoch ms from the ISO timestamp; used by box plots on time axes |
| `status_label` | `status` | Human-readable status (`"Open"` / `"Closed"`) |
| `priority_label` | `priority` | Stripped priority label (`"High"` / `"Medium"` / `"Low"`) |
| `created_month` | `created_ts` | `"YYYY-MM"` substring; groups issues by calendar month |
| `created_hour` | `created_ts` | Integer 0–23; groups issues by hour of day |

---

### Chart Type Reference

| `type` | Plotly trace | D3 SVG approach | Required fields | Key config options |
|---|---|---|---|---|
| `line` | `scatter` mode `lines` | `d3.line()` path | `xAxis`, `yAxis` | `colorBy` splits into multiple traces |
| `bar` | `bar` | `scaleBand` + `rect` | `xAxis`, `yAxis` | `aggregation`, `groupBy`, `colorBy` |
| `scatter` | `scatter` mode `markers` | `circle` elements | `xAxis`, `yAxis` | `marker.colorField`, `marker.sizeField`, `marker.size` |
| `pie` | `pie` | `d3.pie()` + `d3.arc()` | `xAxis` (labels), `yAxis` (values) | Renders as donut by default |
| `area` | `scatter` fill `tonexty` | `d3.area()` path | `xAxis`, `yAxis` | `stackGroup`, `colorBy` |
| `heatmap` | `heatmap` | `scaleBand` rect grid | `xAxis`, `yAxis`, `zAxis` | `marker.colorScale` |
| `histogram` | `histogram` | `d3.bin()` + `rect` | `xAxis` | `colorBy` overlays groups |
| `box` | `box` | Custom IQR box + whiskers | `xAxis`, `yAxis` | `colorBy` creates one box per category |
| `sankey` | `sankey` | `d3-sankey` layout + SVG paths | `xAxis` (source), `sankeyTarget` (target), `yAxis` (value) | `aggregation` combines rows per `(source, target)` pair |

---

## Component API

### DynamicChart

The main component. Resolves its config and data (from inline values or URLs), applies cross-filtering, then delegates rendering to whichever renderer is active in the surrounding `<ChartProvider>`.

```tsx
import { DynamicChart } from 'chart-lib';

<DynamicChart
  config={configObjectOrUrl}
  data={dataArrayOrUrl}
  onChartClick={(event, config) => { /* PlotMouseEvent — Plotly only */ }}
  onChartSelect={(event, config) => { /* PlotSelectionEvent — Plotly only */ }}
  style={{ height: 400 }}
  className="my-chart"
/>
```

| Prop | Type | Required | Description |
|---|---|---|---|
| `config` | `ChartConfig \| string` | ✅ | Inline config object or URL to a config JSON file |
| `data` | `ChartDataRow[] \| string` | ✅ | Inline data array or URL to a data JSON endpoint |
| `onChartClick` | `(event: PlotMouseEvent, config: ChartConfig) => void` | — | Fired on point click — **Plotly renderer only** |
| `onChartSelect` | `(event: PlotSelectionEvent, config: ChartConfig) => void` | — | Fired on box/lasso completion — **Plotly renderer only** |
| `style` | `React.CSSProperties` | — | Applied to the chart container element |
| `className` | `string` | — | CSS class applied to the chart container element |

The component handles its own loading and error states. A spinner is shown while config/data fetch; a styled error card is shown on validation failure. Neither crashes the page.

---

### ChartProvider

Wraps one or more `DynamicChart` instances with:
- **`RendererProvider`** — makes the chosen rendering library available to all descendant charts
- **`SelectionProvider`** — cross-filtering state shared across all sibling charts

```tsx
import { ChartProvider, DynamicChart } from 'chart-lib';

// All charts use Plotly (default):
<ChartProvider>
  <DynamicChart config={barConfig} data={data} />
  <DynamicChart config={sankeyConfig} data={data} />
</ChartProvider>

// Switch all charts to D3.js:
<ChartProvider renderer="d3">
  <DynamicChart config={barConfig} data={data} />
  <DynamicChart config={sankeyConfig} data={data} />
</ChartProvider>
```

| Prop | Type | Default | Description |
|---|---|---|---|
| `renderer` | `'plotly' \| 'd3'` | `'plotly'` | Rendering backend for all charts inside this provider |
| `children` | `React.ReactNode` | — | One or more `DynamicChart` components |

Charts inside the same `<ChartProvider>` automatically cross-filter each other. Charts in separate providers are fully isolated.

To switch the renderer at runtime (e.g. from a UI toggle), change the `renderer` prop. Because `DynamicChart` reads the renderer from context, all charts update simultaneously without any config or data changes.

---

### useSelection

Access the current cross-filter selection from anywhere inside a `<ChartProvider>`:

```tsx
import { useSelection } from 'chart-lib';

const {
  selection,            // SelectionState | null
  clearSelection,       // () => void
  filterData,           // (rows: ChartDataRow[], field: string) => ChartDataRow[]
  setSelectionByValues, // renderer-agnostic selection setter
} = useSelection();
```

| Member | Type | Description |
|---|---|---|
| `selection` | `SelectionState \| null` | Current selection, or `null` when nothing is selected |
| `clearSelection()` | `() => void` | Clears the active selection — all charts return to full data |
| `filterData(rows, field)` | function | Returns the subset of `rows` where `row[field]` is in the selected values. Returns `rows` unchanged when no selection is active. |
| `setSelectionByValues(chartId, field, values)` | function | Renderer-agnostic way to set a selection. Used internally by the D3 renderer. |

**`SelectionState` type:**

```ts
interface SelectionState {
  sourceChartId: string;                    // ID of the chart that originated the selection
  selectedIndices: number[];                // Plotly point indices (empty for non-Plotly renderers)
  selectedValues: (string | number | null)[]; // Distinct x-axis values selected
  field: string;                            // The xAxis field name used to cross-filter
}
```

---

### useRenderer

Read the active renderer type from anywhere inside a `<ChartProvider>`:

```tsx
import { useRenderer } from 'chart-lib';

const { renderer } = useRenderer();
// renderer: 'plotly' | 'd3'
```

---

### Renderer Utilities

```tsx
import { RENDERER_META, RendererProvider } from 'chart-lib';
import type { RendererType, RendererMeta } from 'chart-lib';

// RENDERER_META: Record<RendererType, RendererMeta>
// Contains label, description, badge text, and brand colour for each renderer.
// Useful for building your own renderer picker UI.

const meta: RendererMeta = RENDERER_META['d3'];
// { type: 'd3', label: 'D3.js', description: '...', badge: 'D3', color: '#f68026' }
```

---

### Data Cache Utilities

```tsx
import { prefetchData, clearDataCache, isStreamUrl, useStreamStatus } from 'chart-lib';

// Start fetching (JSON) or streaming (NDJSON) a URL before any chart mounts.
// Routes automatically based on whether the URL is a stream endpoint.
// Subsequent useChartData calls for the same URL share the single request.
prefetchData('/api/v1/issues/stream');   // starts NDJSON stream
prefetchData('/api/v1/issues/');        // starts regular JSON fetch

// Evict a URL from all caches (JSON resolved cache, in-flight promise, stream cache).
clearDataCache('/api/v1/issues/stream');

// Evict everything.
clearDataCache();

// Detect whether a URL will be treated as a stream endpoint.
// Matches any URL whose last path segment starts with "stream".
isStreamUrl('/api/v1/issues/stream'); // → true
isStreamUrl('/api/v1/issues/');       // → false

// React hook — returns a live snapshot of a stream's progress.
// Re-renders the consumer on every new chunk. Safe to call with any URL;
// returns a zeroed snapshot for non-stream URLs.
const { rowCount, streaming, firstRowMs, lastRowMs, error } =
  useStreamStatus('/api/v1/issues/stream');
```

| `StreamStatus` field | Type | Description |
|---|---|---|
| `rowCount` | `number` | Total rows received so far |
| `streaming` | `boolean` | `true` while the stream is still open |
| `firstRowMs` | `number \| null` | ms from fetch start until first row parsed |
| `lastRowMs` | `number \| null` | ms from fetch start until stream closed |
| `error` | `Error \| null` | Set if the stream request failed |

---

### Low-level Adapters

For advanced use — building a custom renderer or wrapping Plotly directly:

```ts
import { buildTraces, buildLayout } from 'chart-lib'; // Plotly adapter
import {
  PALETTE,
  aggregate,
  groupByField,
  col,
  applyAggregation,
  buildSankeyData,        // prepares {nodes, links} graph from flat rows
} from 'chart-lib';
import type { SankeyData, SankeyNodeData, SankeyLinkData } from 'chart-lib';
```

`buildSankeyData(rows, sourceField, targetField, valueField, aggName?)` groups flat data rows
by the `(sourceField, targetField)` pair, applies the chosen aggregation function, deduplicates
nodes (source names first so they sit on the left column), and returns a fully-prepared
`{ nodes: SankeyNodeData[], links: SankeyLinkData[] }` graph ready to hand to any Sankey
renderer.

The renderer components are also exported for consumers who want to use them directly:

```tsx
import { PlotlyRenderer, D3Renderer } from 'chart-lib';
// Props types:
import type { PlotlyRendererProps, D3RendererProps } from 'chart-lib';
```

---

## Rendering Libraries

### Plotly (default)

`renderer="plotly"` uses [Plotly.js](https://plotly.com/javascript/) via `react-plotly.js`.

**Supports:** all 9 chart types  
**Strengths:** interactive zoom/pan/hover tooltips, built-in selection events (box-select, lasso), downloadable PNG, scroll-zoom, rich built-in modebar. Plotly's native `sankey` trace type handles layout and curved-link rendering automatically.  
**Cross-filter trigger:** `plotly_selected` / `plotly_click` events via `setSelectionFromEvent`

```tsx
<ChartProvider renderer="plotly">
  <DynamicChart config={...} data={...} />
</ChartProvider>
```

---

### D3.js

`renderer="d3"` uses [D3.js](https://d3js.org/) v7 for custom SVG rendering, with [d3-sankey](https://github.com/d3/d3-sankey) v0.12 for the Sankey layout.

**Supports:** all 9 chart types (including full box plot with IQR, whiskers, outliers; heatmap with configurable colour scale; and Sankey with hover tooltips and node-click cross-filtering)  
**Strengths:** maximum control over every visual element, hand-crafted SVG output, consistent colours via shared `PALETTE`  
**Cross-filter trigger:** `click` event handlers on SVG elements via `setSelectionByValues`. For Sankey, clicking a **node** selects all rows whose source-field value matches that node name.

```tsx
<ChartProvider renderer="d3">
  <DynamicChart config={...} data={...} />
</ChartProvider>
```

Each D3 chart clears its container (`innerHTML = ''`) and redraws on every data/config change. Chart dimensions are read from the rendered container at draw time.

**D3 Sankey implementation details:**

- `buildSankeyData` in `dataUtils.ts` groups flat rows by `(xAxis.field, sankeyTarget)` and aggregates the `yAxis.field` value, producing `{ nodes, links }` with integer-indexed links
- `d3-sankey`'s layout engine positions nodes and computes Bézier curve control points for each link
- Node rectangles are colour-coded from `PALETTE`; link strokes derive from the source-node colour at 38% opacity, rising to 65% on hover
- SVG `<title>` tooltips are attached to both links (`source → target\nvalue`) and nodes (`name\ntotal`)
- Node labels are anchored outside the rectangle — right-side for source nodes, left-side for target nodes — and display the node name with its total flow value

---

### Renderer Comparison

| Capability | Plotly | D3.js |
|---|---|---|
| Line | ✅ | ✅ |
| Bar | ✅ | ✅ |
| Scatter | ✅ | ✅ |
| Pie / Donut | ✅ | ✅ |
| Area | ✅ | ✅ |
| Histogram | ✅ | ✅ |
| Box Plot | ✅ | ✅ |
| Heatmap | ✅ | ✅ |
| **Sankey** | ✅ | ✅ |
| Zoom / Pan | ✅ built-in | ❌ |
| Hover tooltips | ✅ built-in | ✅ SVG `<title>` |
| Cross-filtering | ✅ | ✅ |
| Output format | WebGL/SVG canvas | SVG |

---

## Cross-filtering

Cross-filtering works automatically for both renderers when charts share a `<ChartProvider>`. The mechanism:

1. User interacts with **Chart A** (click, box-select, lasso, or node click on Sankey)
2. The active renderer calls either `setSelectionFromEvent` (Plotly) or `setSelectionByValues` (D3) on the shared `SelectionContext`
3. The context stores `{ sourceChartId, selectedValues, field }` — the distinct x-axis values of the selected points
4. **Chart B** (and all other siblings) re-compute `displayData` by filtering rows where `row[sourceField]` is in `selectedValues`
5. Charts re-render with filtered data; **the source chart is excluded from filtering** (it is never filtered by its own selection)

> **Note:** The demo Sankey configs ship with `"enableSelection": false` because Sankey flows represent aggregate totals across the full dataset — filtering the dataset being visualised would change the diagram itself rather than highlighting a slice. You can enable Sankey node selection for cross-filtering sibling charts by setting `"enableSelection": true` in the config.

**Clearing the selection:**
- Click the **↺ Reset** button on any individual chart card (also resets that card's zoom)
- Click **↺ Reset All Charts** in the toolbar (clears selection and remounts all cards)
- Switch data source or renderer — the `<ChartProvider>` remounts, clearing all state

---

## Data Sources

The demo supports three data sources. Each source maps to its own set of chart configs. The mapping is defined in `App.tsx` and is intentionally explicit to serve as a living example of how to configure the library for different backends.

### Shared-fetch / Stream Cache

All 9 charts on the page share one `<ChartProvider>` and all point at the same data URL. The `useChartData` hook routes each URL to the appropriate cache path based on `isStreamUrl()`.

**JSON sources** use two module-level Maps:

```
resolvedDataCache   URL → ChartDataRow[]          (populated after fetch + parse completes)
fetchPromiseCache   URL → Promise<ChartDataRow[]>  (populated while a fetch is in-flight)
```

- Chart 1 starts a fetch and registers its Promise in `fetchPromiseCache`
- Charts 2–9 find the existing Promise and attach to it — **one network request**
- When the fetch resolves, all 9 hooks update simultaneously
- On subsequent mounts, `resolvedDataCache` returns data synchronously — **zero loading flash**

**NDJSON stream sources** use a third Map:

```
streamCache   URL → StreamEntry   (created synchronously on first call; holds rows[], listeners, timing)
```

- `getOrStartStream(url)` creates the `StreamEntry` synchronously and fires `fetch()` once
- The `ReadableStream` body is read chunk-by-chunk; each newline-delimited JSON object is parsed with `ChartDataRowSchema`, enriched via `transformRows`, and appended to `entry.rows`
- Every time new rows land, all `entry.listeners` callbacks are invoked — every subscribed `useChartData` instance re-renders its chart with the latest partial dataset
- `entry.firstRowMs` / `entry.lastRowMs` record elapsed time from fetch start to first row / stream close
- When the stream closes, the final array is promoted to `resolvedDataCache` so future mounts after the stream finishes are synchronous

**Loading behaviour differs per source type:**

| Event | JSON | NDJSON stream |
|---|---|---|
| Loading banner shown until… | Full body parsed + Zod validated | First row arrives |
| Charts appear… | All 9 at once after full load | All 9 at once after first rows |
| Charts update after appearing | No (static) | Yes — live as rows arrive |
| `streaming` flag on `useChartData` | Always `false` | `true` until stream closes |
| LIVE badge on each chart | Not shown | Shown while stream is open |
| Toolbar badge | `⚡ loaded in Xs` | `⚡ first row in Xs · N rows (streaming…)` |

### Issues API Source

```
Data URL:    http://127.0.0.1:8090/api/v1/issues/stream  (proxied via Vite → /api/v1/issues/stream)
Config set:  src/config/api/*.json
```

The Issues API endpoint streams rows as **NDJSON** (newline-delimited JSON) — one `ChartDataRow` object per line. The app detects this automatically because the URL's last path segment is `stream`.

Charts are configured around the Issues data model:

| Chart | Title | Key fields |
|---|---|---|
| Line | Issues Over Time by Source System | `created_ts`, `_count`, `src_sys_nm` |
| Bar | Issues by Status | `status`, `_count` |
| Scatter | Issue Type vs Priority | `type`, `priority` |
| Pie | Issues by Priority | `priority`, `_count` |
| Area | Issues by Review Category | `review_ctgy`, `_count` |
| Heatmap | Issue Count: Review Category × Dept | `review_ctgy`, `fn_dept`, `_count` |
| Histogram | Issues by Source System | `src_sys_nm`, `fn_dept` |
| Box Plot | Issue Creation Timeline by Dept | `fn_dept`, `_timestamp` |
| **Sankey** | **Issue Flow: Priority → Status** | **`priority` → `status`, `_count`** |

---

### Clinical Data API Source

```
Base URL:    http://127.0.0.1:8090/api/v1/{domain}  (proxied via Vite → /api/v1/{domain})
Config set:  src/config/clinical/{domain}/*.json  (9 configs per domain)
```

The Clinical Data API exposes six [CDISC SDTM](https://www.cdisc.org/standards/foundational/sdtm) clinical trial domains, each available as a **paginated JSON** endpoint at `/api/v1/{domain}` (where `{domain}` is lowercase, e.g. `/api/v1/ae`). The response envelope is:

```json
{
  "data": [ /* ChartDataRow[] — the current page */ ],
  "meta": {
    "page": 1,
    "page_size": 100,
    "total_records": 37763,
    "total_pages": 378
  }
}
```

`ChartDataSchema` in `chart-lib` automatically unwraps the `{ data: [...] }` envelope, so no adapter changes are required.

**Supported domains:**

| Domain | Full Name | Key fields |
|---|---|---|
| `AE` | Adverse Events | `AESEV`, `AEREL`, `AEOUT`, `AEBODSYS`, `AESTDTC`, `AESEQ`, `AE_INCIDENT_GROUP` |
| `CM` | Concomitant Medications | `CMCAT`, `CMROUTE`, `CMDOSE`, `CMDOSFRM`, `CMSTDTC` |
| `DM` | Demographics | `AGE`, `SEX`, `RACE`, `COUNTRY`, `ARM`, `DMDTC` |
| `LB` | Laboratory Results | `LBTESTCD`, `LBORRES`, `LBORRESU`, `LBSTNRLO`, `LBSTNRHI`, `LBDTC` |
| `TV` | Trial Visits | `VISIT`, `VISITNUM`, `TVSTRL`, `TVENRL`, `ARMCD` |
| `VS` | Vital Signs | `VSTESTCD`, `VSORRES`, `VSORRESU`, `VSDTC` |

All six share common fields: `STUDY`, `SITE`, `SUBJECT`, `VISIT`, `FORM`, `DOMAIN`, `SITEID`, `STUDYID`, `USUBJID`.

**Common query parameters (all domains):**

| Parameter | Description |
|---|---|
| `study` | Filter by STUDY identifier |
| `site` | Filter by SITE identifier |
| `subject` | Filter by SUBJECT identifier |
| `visit` | Filter by VISIT name |
| `form` | Filter by FORM name |
| `page` | Page number (1-indexed, default 1) |
| `page_size` | Records per page (1–1000, default 100) |

**UI controls:**

- **Domain selector** — a tab strip of six buttons (AE / CM / DM / LB / TV / VS) that switches the active endpoint and config set. Changing domain resets to page 1 and clears pagination meta.
- **Filters** — five text inputs (`study`, `site`, `subject`, `visit`, `form`) with a 400 ms debounce. Changing any filter resets to page 1 automatically.
- **Pagination navigation** — Prev / Next buttons, a direct page number input, and a page size selector (50 / 100 / 200 / 500 / 1000 rows). The total pages and total records fetched from `meta` are displayed alongside the page input.
- **Toolbar badge** — shows the current page and total records when the Clinical source is active.

**Chart config sets** — each domain has 9 chart configs under `src/config/clinical/{domain}/`, using the domain-specific CDISC field names:

| Chart type | AE example | LB example | VS example |
|---|---|---|---|
| Bar | AEs by Severity (`AESEV`) | Lab Tests by Test Code (`LBTESTCD`) | Vital Signs by Test (`VSTESTCD`) |
| Scatter | AE Relatedness vs Outcome | Lab Result vs Upper Normal Limit | Result by Visit |
| Box Plot | AE Sequence by Body System | Lab Result Distribution | Result Distribution by Test |
| Line | AEs Over Time by Body System | Lab Tests Over Time | Vital Signs Over Time |
| Histogram | AEs by Outcome | Lab Tests by Visit | Vital Signs by Visit |
| Pie | AEs by Drug Relatedness | Lab Tests by Test Code | Measurements by Test |
| Heatmap | Body System × Severity | Test Code × Visit | Test × Visit |
| Area | AEs by Incident Group | Tests by Code (Stacked) | Tests by Code (Stacked) |
| Sankey | AE Flow: Severity → Outcome | LB Flow: Test Code → Visit | VS Flow: Test → Visit |

---

### Sample File Source

```
Data URL:    /src/data/sample-data.json  (24-row sales dataset served as static asset)
Config set:  src/config/file/*.json
```

The sales dataset has these fields: `date`, `region`, `category`, `sales`, `volume`, `price`.

| Chart | Title | Key fields |
|---|---|---|
| Line | Monthly Sales by Region | `date`, `sales`, `region` |
| Bar | Total Sales by Region | `region`, `sales` |
| Scatter | Volume vs Sales by Category | `volume`, `sales`, `category` |
| Pie | Sales by Category | `category`, `sales` |
| Area | Monthly Sales by Category (Stacked) | `date`, `sales`, `category` |
| Heatmap | Sales Intensity: Region × Category | `region`, `category`, `sales` |
| Histogram | Volume Distribution by Category | `volume`, `category` |
| Box Plot | Sales Distribution by Region | `region`, `sales` |
| **Sankey** | **Sales Flow: Region → Category** | **`region` → `category`, `sales`** |

---

## Demo App Features

### Renderer selector

A **Library** tab strip in the header lets you switch the rendering backend for all 9 charts simultaneously. The two options are:

| Badge | Label | Colour |
|---|---|---|
| `PLT` | Plotly | `#3f4f75` (navy) |
| `D3` | D3.js | `#f68026` (orange) |

Switching bumps `globalResetKey`, which remounts the `<ChartProvider>` (clearing selection state) and passes the new `renderer` prop. All 9 charts appear with the new renderer simultaneously.

### Data source selector

A **Data Source** tab strip below the Library selector lets you switch between the three sources: the streaming Issues API, the Clinical Data API, and the sample data file. Switching:
1. Calls `prefetchData(newUrl)` immediately (before re-rendering) to start the fetch
2. Replaces the config set with the configs appropriate for that source
3. Remounts `<ChartProvider>` to clear cross-filter state
4. Clears the event log

### Per-chart reset button

Every chart card has a **↺ Reset** button in its header. Clicking it:
1. Calls `clearSelection()` — removes the global cross-filter affecting all charts
2. Increments the card's internal `chartKey` — remounts `DynamicChart`, restoring default zoom and pan

A **"● filtered"** badge appears on the card header whenever a cross-filter is active, making it immediately obvious which charts are affected.

### Reset All Charts

A global **↺ Reset All Charts** button above the chart grid:
1. Calls `clearSelection()`
2. Increments `globalResetKey` — remounts the entire `<ChartProvider>` tree, resetting all charts at once

The button glows amber when a cross-filter is active.

### Loading gate

While the data is loading, a single **`DataLoadingBanner`** replaces the chart grid. It shows:
- A dual-ring animated spinner
- The data source label (colour-coded to match the source badge)
- The exact data URL being fetched/streamed
- An elapsed-time counter updating every 100 ms
- An indeterminate progress bar
- For **JSON sources**: *"One shared request for all 9 charts. Subsequent renders will be instant (cached)."*
- For **stream sources**: *"Streaming rows from the server — charts appear as soon as the first batch arrives."*

All 9 charts appear **simultaneously** once data is ready:
- **JSON**: the moment the full body is parsed and Zod-validated
- **Stream**: the moment the first NDJSON rows arrive (charts then update live)

After the first load, switching back to the same source uses the module-level cache — charts appear instantly with no loading flash.

### Live streaming badge

While the NDJSON stream is still open, each chart card displays a pulsing green **● LIVE** badge in its top-right corner. The badge disappears automatically when the stream closes. This makes it immediately clear that chart data is still growing.

### Streaming toolbar counter

Once charts appear for a stream source, the toolbar load-time badge updates live:

```
⚡ first row in 0.18s · 4,231 rows (streaming…)
```

The row count increments with every chunk. The `(streaming…)` suffix disappears when the stream closes.

### Event log

A sidebar panel records the last 20 chart interactions:
- **click** events (blue badge) — fired when a user clicks a single data point (Plotly renderer)
- **select** events (green badge) — fired when a box or lasso selection completes (Plotly renderer)

Each entry shows the chart title, number of points, and timestamp. A **Clear** button empties the log.

### Clinical Data API controls

When the **Clinical Data API** source is active, three additional control areas appear below the data source selector:

**Domain selector** — a tab strip of six CDISC SDTM domain buttons:

| Tab | Domain | Description |
|---|---|---|
| `AE` | Adverse Events | Safety events, severity, relatedness |
| `CM` | Concomitant Medications | Drug categories, dosage, routes |
| `DM` | Demographics | Age, sex, race, arm, country |
| `LB` | Laboratory Results | Test codes, values, reference ranges |
| `TV` | Trial Visits | Visit schedule, start/end rules |
| `VS` | Vital Signs | Test codes, values, units |

Switching domain resets to page 1, clears pagination meta, and swaps in the 9 chart configs from `src/config/clinical/{domain}/`.

**Filter inputs** — five text fields with a **400 ms debounce** that append query parameters to the API URL:

| Field | Query param | Effect |
|---|---|---|
| Study | `study` | Filter by study identifier |
| Site | `site` | Filter by site identifier |
| Subject | `subject` | Filter by subject identifier |
| Visit | `visit` | Filter by visit name |
| Form | `form` | Filter by form name |

Typing in any filter field resets pagination to page 1 automatically. A **Clear Filters** button resets all five fields at once.

**Pagination row** — appears directly below the filter fields:
- **← Prev** / **Next →** buttons (disabled at first/last page)
- Direct **page number input** (accepts a number; Enter or blur commits the navigation)
- **`/ N pages`** label showing total pages from `meta.total_pages`
- **`(M total records)`** label showing `meta.total_records`
- **Page size selector** dropdown (options: 50 / 100 / 200 / 500 / 1000 rows per page)

Changing page size or navigating resets the old URL in `resolvedDataCache` via `clearDataCache` to guarantee a fresh fetch.

---

## Embedding in a Web Application

### 1. Install dependencies

```sh
npm install plotly.js react-plotly.js react react-dom d3 d3-sankey
# When chart-lib is published to npm:
npm install chart-lib
```

### 2. Wrap your chart section with ChartProvider

```tsx
import { ChartProvider, DynamicChart } from 'chart-lib';

export default function Dashboard() {
  return (
    // All charts use Plotly by default; change renderer prop to switch all at once
    <ChartProvider renderer="plotly">
      <DynamicChart
        config="/api/charts/sales-bar/config"
        data="/api/charts/sales-bar/data"
        style={{ height: 350 }}
      />
      <DynamicChart
        config="/api/charts/sales-sankey/config"
        data="/api/charts/sales-sankey/data"
        style={{ height: 350 }}
      />
    </ChartProvider>
  );
}
```

### 3. Add a renderer picker (optional)

```tsx
import { ChartProvider, DynamicChart, RENDERER_META } from 'chart-lib';
import type { RendererType } from 'chart-lib';
import { useState } from 'react';

export default function Dashboard() {
  const [renderer, setRenderer] = useState<RendererType>('plotly'); // 'plotly' | 'd3'

  return (
    <>
      {/* Renderer selector */}
      <div>
        {Object.values(RENDERER_META).map((meta) => (
          <button key={meta.type} onClick={() => setRenderer(meta.type)}>
            {meta.label}
          </button>
        ))}
      </div>

      {/* Key changes when renderer changes — remounts provider + all charts */}
      <ChartProvider key={renderer} renderer={renderer}>
        <DynamicChart config="/api/config/bar"    data="/api/data" style={{ height: 350 }} />
        <DynamicChart config="/api/config/sankey" data="/api/data" style={{ height: 350 }} />
      </ChartProvider>
    </>
  );
}
```

### 4. Prefetch data for faster first render

`prefetchData` automatically routes to the right path — a regular JSON fetch or an NDJSON stream — based on the URL:

```tsx
import { prefetchData } from 'chart-lib';
import { useEffect } from 'react';

function App() {
  useEffect(() => {
    // JSON endpoint: starts a single shared fetch before charts mount
    prefetchData('/api/v1/data/');

    // NDJSON stream endpoint: starts the stream reader before charts mount;
    // charts render with partial data as rows arrive
    prefetchData('/api/v1/data/stream');
  }, []);

  // ...
}
```

### 4a. Show live stream progress (optional)

Use `useStreamStatus` to display row counts or a streaming indicator in your own UI:

```tsx
import { useStreamStatus } from 'chart-lib';

function StreamIndicator({ url }: { url: string }) {
  const { rowCount, streaming, firstRowMs } = useStreamStatus(url);

  if (!streaming && firstRowMs === null) return null;

  return (
    <span>
      {streaming
        ? `Streaming… ${rowCount.toLocaleString()} rows`
        : `Done — ${rowCount.toLocaleString()} rows in ${(firstRowMs! / 1000).toFixed(2)}s`}
    </span>
  );
}
```

### 5. Shape your config API

Your config endpoint must return a valid `ChartConfig` JSON object. For a Sankey chart:

```json
{
  "id": "pipeline-flow",
  "type": "sankey",
  "title": "Sales Pipeline: Stage → Outcome",
  "xAxis": { "field": "stage",   "title": "Stage" },
  "sankeyTarget": "outcome",
  "yAxis": { "field": "revenue", "title": "Revenue ($)" },
  "aggregation": "sum",
  "enableSelection": false
}
```

For all other chart types, the `sankeyTarget` field is simply omitted and `xAxis` maps to the horizontal axis as usual.

### 6. Shape your data API

Your data endpoint must return a `ChartDataRow[]` JSON array or `{ data: ChartDataRow[] }`:

```json
[
  { "stage": "Prospect", "outcome": "Won",  "revenue": 42000 },
  { "stage": "Prospect", "outcome": "Lost", "revenue": 18000 },
  { "stage": "Demo",     "outcome": "Won",  "revenue": 31000 }
]
```

The library validates both against Zod schemas at runtime. Invalid responses render a clean error card; they never crash the page.

---

## Switching from Mock Files to Real APIs

The demo uses local JSON files served as Vite static assets. The `useChartConfig` and `useChartData` hooks accept any valid URL, so switching to real endpoints requires only a URL change:

```tsx
// Before (mock files):
<DynamicChart
  config="/src/config/api/sankey-chart.json"
  data="/src/data/sample-data.json"
/>

// After (real JSON API endpoint):
<DynamicChart
  config="/api/v1/charts/pipeline-sankey/config"
  data="/api/v1/charts/pipeline-sankey/data"
/>

// After (NDJSON stream endpoint — URL last segment must start with "stream"):
<DynamicChart
  config="/api/v1/charts/pipeline-sankey/config"
  data="/api/v1/charts/pipeline-sankey/stream"
/>
```

The Vite dev server's proxy config (`vite.config.ts`) already forwards `/api` to `http://127.0.0.1:8090`, so both the Issues API JSON and stream endpoints work out of the box.

**Stream endpoint requirements:**
- Must respond with `Content-Type: application/x-ndjson` (or similar)
- Each line must be a valid JSON object matching `ChartDataRow` (`Record<string, string | number | null>`)
- Lines are delimited by `\n`; a trailing newline is optional

If your API requires authentication headers, extend `useChartConfig` and `useChartData` with a `fetchOptions` parameter:

```ts
// Inside useChartData.ts — add fetchOptions to every fetch() call:
fetch(url, { headers: { Authorization: `Bearer ${token}` } })
```

---

## Technology Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Primary visualization** | **Plotly.js** via `react-plotly.js` | Covers all 9 chart types including a native `sankey` trace; rich built-in selection events; interactive zoom/pan/download out of the box |
| **Secondary visualization** | **D3.js** | Maximum SVG control; covers all 9 types without plugins (bar, line, scatter, pie, area, histogram, box, heatmap use core d3; Sankey uses `d3-sankey`) |
| **Sankey layout** | **d3-sankey** v0.12 | Purpose-built Sankey layout engine for D3; computes node positions, link widths, and Bézier control points; integer-indexed link resolution is used directly — no custom nodeId accessor needed |
| **Renderer selection** | **React Context** (`RendererContext`) | A single prop on `ChartProvider` switches all descendant charts simultaneously — no per-chart wiring |
| **Cross-filter mechanism** | **React Context** (`SelectionContext`) | Shared state without a global store; scoped to a single `ChartProvider` so multiple independent dashboards on one page don't interfere |
| **Renderer-agnostic selection** | `setSelectionByValues(chartId, field, values[])` | D3 doesn't emit Plotly events; a separate setter accepts raw values and populates the same `SelectionState` shape. For Sankey, clicking a node emits the node name as the selected value |
| **Sankey data preparation** | `buildSankeyData` in `dataUtils.ts` | Shared utility used by both Plotly and D3 adapters; groups flat rows by `(source, target)` composite key, applies any aggregation function, deduplicates nodes with sources ordered first |
| **Data fetch deduplication** | Module-level `Map` cache in `useChartData` | 9 charts sharing one URL → 1 network request; cache is shared across React re-renders; subsequent mounts read synchronously |
| **NDJSON streaming** | `ReadableStream` + `TextDecoder` + pub/sub `Set<listener>` | Streams rows incrementally from the server; all N charts share one `ReadableStream` reader via a module-level `streamCache`; each chunk notifies every subscriber so charts update live without additional fetches |
| **Stream URL detection** | Last path-segment heuristic (`isStreamUrl`) | Zero configuration required; any endpoint whose path ends with `stream` is automatically routed through the stream reader; the same `prefetchData` call works for both JSON and stream URLs |
| **Partial-data rendering** | `streaming && rows.length > 0` guard in `DynamicChart` | Charts appear as soon as the first rows arrive rather than waiting for stream close; the loading spinner only blocks when there is truly nothing to render |
| **Prefetching** | `prefetchData(url)` | Fire-and-forget function that primes the cache (or starts the stream) before any chart mounts; dramatically reduces time-to-first-render on slow APIs |
| **Validation** | **Zod** | Runtime validation of config and data from external sources; infers TypeScript types from schemas at zero extra cost; per-row validation in the stream reader means a single malformed line never aborts the whole stream |
| **Monorepo tooling** | **Bun workspaces** | Single lockfile; fast installs; native `workspace:*` protocol for cross-package linking |
| **Build** | **Vite** (library mode + app mode) | Fast builds; `vite-plugin-dts` for `.d.ts` generation; tree-shakeable ESM output; all three rendering libraries externalized |
| **Testing** | **Bun test** | Built-in Jest-compatible test runner; no extra dependencies; fast execution |
| **TypeScript** | Strict + `exactOptionalPropertyTypes` | Catches undefined-vs-missing-property bugs that `strict` alone misses; particularly important for renderer prop spreading |
