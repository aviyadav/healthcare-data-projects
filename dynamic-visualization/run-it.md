# Demo Run-Book — Dynamic Visualization Utility

Step-by-step instructions for a live demo. Follow in order. Each step includes what to say and what the audience should see on screen.

---

## Before You Start (prep — do this before the audience arrives)

- [ ] Open a terminal and `cd` to the repo root:
  ```
  cd dynamic-visualization
  ```
- [ ] Confirm dependencies are installed (only needed once):
  ```
  bun install
  ```
- [ ] Have VS Code (or your editor) open with the project tree visible so you can flip to files quickly.
- [ ] Close unrelated browser tabs; keep one browser window ready.
- [ ] Optionally, have the Issues API server running at `http://127.0.0.1:8090` if you want to demo live API data. The server must expose `/api/v1/issues/stream` as an **NDJSON stream** (one JSON object per line, `Content-Type: application/x-ndjson`). The app works without it — it will fall back gracefully to the sample file source.

---

## Step 1 — Show the Repo Structure (1 min)

**Say:** *"This is a Bun monorepo with two packages."*

Point to in the editor:
- `packages/chart-lib/` — the publishable component library
- `packages/demo/` — the Vite demo application
- `bunfig.toml` — Bun workspace & runtime configuration
- `tsconfig.base.json` — shared strict TypeScript config

**Say:** *"The library ships with zero chart-type knowledge baked into the component. Everything — chart type, axis fields, colours, aggregation, interactivity — comes from a config object or a URL. The same config JSON drives two completely different rendering libraries."*

---

## Step 2 — Show the Architecture (1 min)

Open `packages/chart-lib/src/components/DynamicChart.tsx`.

**Point out** the `switch (renderer)` block near the bottom:

```tsx
switch (renderer) {
  case 'd3':   return <D3Renderer ... />;
  default:     return <PlotlyRenderer ... />;
}
```

**Say:** *"DynamicChart is a routing shell. It resolves the config, fetches or streams the data, applies cross-filtering — then hands the result to whichever renderer is active in the surrounding ChartProvider. The ChartConfig schema is unchanged regardless of which library renders it."*

**Point out** the loading guard above the switch:

```tsx
const showLoading =
  configLoading ||
  (dataLoading && !streaming) ||       // JSON: block until full load
  (streaming && rawData.length === 0); // Stream: block only until first rows arrive
```

**Say:** *"For streaming sources the chart renders as soon as the first rows land — it doesn't wait for the stream to close. A pulsing LIVE badge overlays the chart while the stream is still open."*

---

## Step 3 — Show a Chart Config File (1 min)

Open `packages/demo/src/config/file/bar-chart.json`.

**Point out:**
```json
{
  "id": "file-bar-chart",
  "type": "bar",
  "title": "Total Sales by Region",
  "xAxis": { "field": "region", "title": "Region", "axisType": "category" },
  "yAxis": { "field": "sales", "title": "Total Sales ($)" },
  "aggregation": "sum",
  "groupBy": "region",
  "colorBy": "region",
  "enableSelection": true,
  "selectionMode": "box"
}
```

**Say:** *"This JSON is what comes back from a config API. The component never hardcodes 'bar' anywhere. This exact config works unchanged whether Plotly or D3 is rendering it."*

Now open `packages/demo/src/config/file/sankey-chart.json` side-by-side.

**Point out:**
```json
{
  "id": "file-sankey-chart",
  "type": "sankey",
  "title": "Sales Flow: Region → Category",
  "xAxis": { "field": "region", "title": "Region" },
  "sankeyTarget": "category",
  "yAxis": { "field": "sales", "title": "Sales ($)" },
  "aggregation": "sum",
  "enableSelection": false
}
```

**Say:** *"The only new fields are `type: sankey` and `sankeyTarget`. The source nodes come from `xAxis.field`, the target nodes from `sankeyTarget`, and the flow weight from `yAxis.field`. Everything else — aggregation, title, renderer — works exactly as before. The library groups all rows by their `(region, category)` pair, sums the sales, and hands the result to Plotly's native sankey trace or the d3-sankey layout engine."*

Open `packages/demo/src/config/api/bar-chart.json` and `packages/demo/src/config/api/sankey-chart.json` to show the Issues API versions.

**Say:** *"Three data sources, three config sets — the streaming Issues API, the Clinical Data API (six CDISC domains), and the sample file. Different fields, different domain-appropriate charts, same config schema running through the same component."*

---

## Step 4 — Run the Tests (1 min)

In the terminal:
```
bun test
```

**Expected output:**
```
✓ src/__tests__/filterData.test.ts (5)
✓ src/__tests__/plotlyAdapter.test.ts (19)

Test Files  2 passed (2)
Tests       24 passed (24)
```

**Say:** *"24 tests — one per chart type for the Plotly adapter, plus the cross-filter data logic. All green. The adapter tests confirm that buildTraces emits the right Plotly trace type, mode, and axis field for each of the 9 chart types — including the new Sankey trace."*

---

## Step 5 — Build the Library (1 min)

```
bun build
```

**Expected output:**
```
dist/index.mjs   ~122 kB
dist/index.cjs   ~87 kB
✓ built in ~5s
```

**Say:** *"ESM and CommonJS with full TypeScript declarations. React and Plotly are peer dependencies — consumers provide those. D3, d3-sankey, and Zod ship as bundled dependencies so they install automatically, but they're externalized in the Rollup output so the bundle stays lean."*

Point to `packages/chart-lib/dist/` in the file tree.

---

## Step 6 — Start the Demo App (30 sec)

```
bun dev
```

Open browser → **http://localhost:5173**

**Say:** *"Let me start the demo. The default view is the Sample Data File source with Plotly. It doesn't just dump 9 charts on screen — it starts the data request first and shows a loading banner."*

Wait for the banner to appear, then point it out.

---

## Step 7 — Walk Through the Loading Gate (1 min)

While the loading banner is visible (or describe it if it resolves quickly):

**Point out:**
- The dual-ring spinner
- The data source URL shown in monospace
- The elapsed-time counter ticking up every 100 ms
- The indeterminate sliding progress bar

**Say:** *"This is the sample file source — a regular JSON fetch. Previously, 9 charts would each independently fetch the same URL — 9 network requests, 9 JSON parses, 9 Zod validations. Now there's exactly one network request. All charts attach to the same Promise. When it resolves, all 9 charts appear simultaneously. The banner replaces 9 individual loading spinners."*

Once the charts appear: **"There — all 9 at once."**

Point to the toolbar badge (e.g. **PLT Plotly**) confirming the active renderer, and the **⚡ loaded in Xs** timing badge.

---

## Step 8 — Walk Through the 9 Chart Types (2 min)

The default view uses the **Sample Data File** source (sales data). Point to each card:

1. **Line** — monthly sales by region (colour-coded traces per region)
2. **Bar** — total sales per region, pre-aggregated by `sum`
3. **Scatter** — volume vs sales, colour-coded by category
4. **Pie / Donut** — sales share by category
5. **Area** — monthly sales by category, stacked
6. **Heatmap** — sales intensity: region × category grid
7. **Histogram** — volume distribution, split by category
8. **Box Plot** — sales spread per region with IQR box, whiskers, and outliers
9. **Sankey** — sales flow from region (left) to category (right); link width is proportional to total sales; hover a link to see the exact source → target value

**Say:** *"Every one of these is the same `<DynamicChart>` component. The only difference is the JSON config file it reads. The Sankey is the newest — it groups all rows by their `(region, category)` pair, sums the sales value, builds a node-link graph, and hands it to Plotly's native sankey trace. Switch to D3 and the same data flows through the d3-sankey layout engine instead."*

---

## Step 9 — Demonstrate Cross-filtering (2 min)

**Say:** *"All charts inside one ChartProvider share selection state. Selecting in one filters all others."*

1. On the **Bar chart**, box-select one or two region bars (Plotly's select tool is active by default when `enableSelection: true`).
2. Watch the **Line**, **Area**, and **Sankey** charts — they immediately filter to the selected regions.
3. Notice the **"● filtered"** badge that appears in the card header.
4. Point to the **Event Log** panel on the right — it shows a `select` event with point count and chart title.

**Say:** *"No custom event wiring between charts. They all consume the same React context. The source chart records which x-axis values are selected, and every sibling filters its data automatically — including the Sankey, which redraws its flows for only the selected regions."*

---

## Step 10 — Demonstrate the Reset Button (1 min)

**Say:** *"Each chart card has its own reset button."*

1. With the cross-filter still active, point to the **↺ Reset** button in any card header.
2. Click it.
3. The cross-filter clears across **all** charts simultaneously.

**Say:** *"The reset button does two things: it clears the global cross-filter selection, and it remounts that specific chart — which also resets zoom and pan to the default view."*

Now point to the **↺ Reset All Charts** button in the toolbar above the grid.

**Say:** *"This version remounts the entire ChartProvider, resetting every chart at once. Notice it glows amber when a cross-filter is active — so it's always discoverable."*

---

## Step 11 — Switch Rendering Library to D3.js (2 min)

In the header, click the **D3 D3.js** tab in the Library selector.

**Say:** *"Switching from Plotly to D3.js — raw SVG, full control. Same configs, same data, different renderer."*

Wait for the charts to re-render (instant — data is already cached).

**Point out:**
- All 9 chart types render — including **Heatmap** (coloured cell grid with value labels), **Box Plot** (IQR box, dashed whiskers, whisker caps, outlier dots), and **Sankey** (SVG rectangles for nodes, Bézier-curve paths for links)
- Axes have tick marks, labels, and a dashed horizontal grid
- The **Sankey** node labels sit outside the rectangles, anchored left for source nodes and right for target nodes; link opacity rises on hover; SVG `<title>` tooltips show the flow value
- The **Box Plot** shows median lines with a different stroke weight
- The **Heatmap** uses the colour scale from the config's `marker.colorScale` field

Demonstrate cross-filtering: click a bar on the Bar chart. The other charts update.

**Say:** *"The D3 renderer calls setSelectionByValues — a renderer-agnostic context method — alongside the existing Plotly-specific setSelectionFromEvent. For the Sankey in D3 mode, clicking a node directly emits that node's name as the selected value. The filter logic and SelectionContext are identical; only the event source differs."*

---

## Step 12 — Switch to the Issues API Stream Source (2 min)

Switch back to **Plotly** renderer first (click PLT).

In the Data Source selector, click the **LIVE Issues API** tab.

**Say:** *"Switching to the live Issues API — but this time the endpoint is a stream. Watch the loading banner: it now says 'Streaming rows from the server' instead of the JSON cache message."*

Point to the loading banner:
- URL shows `/api/v1/issues/stream`
- Elapsed timer counts up
- Banner dismisses the moment the **first batch of rows** arrives — not when the stream closes

Once the charts appear, point out:
- Every chart card has a pulsing green **● LIVE** badge in its top-right corner
- The toolbar shows **⚡ first row in X.XXs · N rows (streaming…)**
- The row count in the toolbar ticks up live as more chunks arrive
- The **● LIVE** badges disappear and `(streaming…)` drops from the toolbar once the stream closes

**Point out** the different chart themes including the Sankey:
- **Line** — issues over time by source system
- **Bar** — issues by status
- **Scatter** — issue type vs priority
- **Heatmap** — issue count: review category × functional department
- **Box Plot** — issue creation timeline by department (epoch ms on Y axis)
- **Sankey** — issue flow from priority level (left) to resolution status (right); link width shows how many issues of each priority end up in each status

**Say:** *"The config set swapped. The data URL swapped. The component code didn't change — and now it incrementally renders live data straight off a stream, including a Sankey that redraws as new rows arrive. That's the entire point."*

---

## Step 13 — Switch Renderer on API Stream Data (30 sec)

With the Issues API source active, click **D3 D3.js**.

**Say:** *"Because the stream cache is warm — rows are already in memory — switching renderer is instant. No loading banner, no spinner. If the stream is still open when you switch, the charts keep updating live in the new renderer too — including the Sankey, now drawn as pure SVG."*

Switch back to **PLT Plotly** for the cleanest experience.

---

## Step 13b — Switch to the Clinical Data API Source (3 min)

In the Data Source selector, click the **CLIN Clinical Data API** tab.

**Say:** *"This is a completely different backend — a paginated clinical trial API exposing CDISC SDTM data. Six domains are available. Let's start with Demographics."*

Click the **DM** domain tab. Wait for the loading banner to resolve. Point out:
- The chart titles now say things like **"Demographics by ARM"**, **"Age Distribution"**, **"Race Distribution"**, **"Country Distribution"**
- All field names are CDISC SDTM — `AGE`, `SEX`, `RACE`, `COUNTRY`, `ARM`
- The toolbar badge shows the current page and total records

**Say:** *"54 chart configs — 6 domains × 9 chart types each. Every config was generated from the API's OpenAPI schema. The chart component itself has no idea it's looking at clinical data."*

Click the **AE** (Adverse Events) domain tab. Wait for charts to load. Point out:
- **Bar** — AEs by Severity (`AESEV`)
- **Heatmap** — Body System × Severity (`AEBODSYS` × `AESEV`)
- **Sankey** — AE Flow: Severity → Outcome (`AESEV` → `AEOUT`)

**Say:** *"Domain switch resets pagination to page 1. The chart provider remounts — cross-filter state is cleared — and a fresh fetch fires for the AE endpoint."*

**Demonstrate pagination:**
- Click the **Next →** button a few times; watch the toolbar badge page number increment
- Type `3` into the page number input and press Enter; charts reload at page 3
- Change the page size dropdown to **200**; watch the charts reload with more records

**Say:** *"The pagination meta comes from the `meta` field of every API response — `total_pages` and `total_records`. Old URL is evicted from cache before each page navigation so you always get a fresh fetch."*

**Demonstrate filters:**
- Type a study identifier into the **Study** field (e.g. `CDISC01`) — after a 400 ms debounce the URL updates and charts reload
- Point out that pagination reset back to page 1
- Click **Clear Filters** to reset

**Say:** *"Filters debounce at 400 ms so the API isn't hammered on every keystroke. Any filter change resets to page 1 automatically — stale page references are never sent."*

Click the **LB** (Laboratory Results) tab. Point out the Sankey: **LB Flow: Test Code → Visit**.

**Say:** *"Six domains, nine chart types each — all driven by config JSON. The component layer, the data hooks, the context — unchanged. Only the URL and the config files are different for each domain."*

---

## Step 14 — Show the Context Code (1 min)

Open `packages/chart-lib/src/context/SelectionContext.tsx`.

**Point out** the two selection setters:

```ts
setSelectionFromEvent(event, config)         // Plotly — extracts values from PlotMouseEvent
setSelectionByValues(chartId, field, values) // All other renderers — accepts raw values
```

**Say:** *"The SelectionState shape is identical either way. Once it's in context, the filterData function doesn't know or care how the selection was created — whether it came from a Plotly box-select, a D3 SVG click, or a D3 Sankey node click."*

---

## Step 15 — Show the Shared Data Cache & Stream Cache (1 min)

Open `packages/chart-lib/src/hooks/useChartData.ts`.

**Point out** the three module-level Maps:

```ts
const resolvedDataCache = new Map<string, ChartDataRow[]>();
const fetchPromiseCache = new Map<string, Promise<ChartDataRow[]>>();
const streamCache      = new Map<string, StreamEntry>();
```

**Say:** *"These three Maps live outside React and survive re-renders. For JSON sources: chart one stores its Promise in fetchPromiseCache, charts two through nine attach to it — one request. After resolve, resolvedDataCache gives subsequent mounts the data synchronously."*

**Point out** the `StreamEntry` type:

```ts
interface StreamEntry {
  rows: ChartDataRow[];
  done: boolean;
  firstRowMs: number | null;
  lastRowMs:  number | null;
  listeners:  Set<() => void>;
}
```

**Say:** *"For stream sources, getOrStartStream creates one StreamEntry per URL — synchronously, before the fetch even completes — then reads the ReadableStream body chunk by chunk. Every time a new line parses into a row, all the listener callbacks fire and every subscribed chart re-renders with the latest partial dataset. One stream, N live charts — including the Sankey, which redraws its entire flow diagram on every chunk. When the stream closes, the final array is promoted into resolvedDataCache, so a Reset All after the stream finishes is still instant."*

---

## Step 16 — Show the Sankey Data Utility (optional — 1 min)

Open `packages/chart-lib/src/utils/dataUtils.ts` and scroll to `buildSankeyData`.

**Point out:**

```ts
export function buildSankeyData(
  rows,
  sourceField,  // xAxis.field
  targetField,  // sankeyTarget
  valueField,   // yAxis.field
  aggName,      // 'sum' | 'count' | 'mean' | 'median'
): SankeyData { ... }
```

**Say:** *"This is the single data-preparation function shared by both the Plotly adapter and the D3 renderer. It groups flat data rows by the composite key `(sourceField, targetField)`, applies any aggregation function within each group, deduplicates nodes with source names ordered first so they sit naturally on the left column, and returns a `{ nodes, links }` graph with integer-indexed links. Both renderers call this before handing the result to their respective layout engines — Plotly's built-in sankey trace or d3-sankey's force-directed layout."*

---

## Step 17 — Live Config Change (optional — 1 min)

While the dev server is running with the Sample File source selected:

1. Open `packages/demo/src/config/file/sankey-chart.json`
2. Change `"aggregation": "sum"` to `"aggregation": "count"`
3. Save — Vite HMR reloads
4. The Sankey now shows flow counts (number of rows per pair) rather than summed sales
5. Revert the change

**Say:** *"Hot reload. Tune configs, see results instantly. No rebuild. This works for any of the 9 chart types — change type, axes, aggregation, or titles and the chart updates in place."*

---

## Wrap-Up Talking Points

- **Config-driven** — chart type, axes, colours, aggregation, selection mode — all from a JSON config or URL. No component changes needed.
- **9 chart types** — Line, Bar, Scatter, Pie/Donut, Area, Heatmap, Histogram, Box Plot, and Sankey. Adding the Sankey required no changes to `DynamicChart` — only a new type value in the schema, a new trace builder in the Plotly adapter, and a new render function in the D3 renderer.
- **Two rendering libraries, one config schema** — Plotly.js and D3.js (with d3-sankey for the Sankey layout) both consume `ChartConfig` identically. Switch at runtime with one prop on `ChartProvider`.
- **Shared Sankey data utility** — `buildSankeyData` in `dataUtils.ts` is called by both the Plotly adapter and the D3 renderer. Groups flat rows by `(source, target)` pair, applies any aggregation, returns a `{ nodes, links }` graph with integer-indexed links ready for either renderer.
- **Cross-filtering across both renderers** — shared React context with renderer-agnostic `setSelectionByValues`; works identically in Plotly and D3, including Sankey node clicks in the D3 renderer.
- **Single-fetch cache** — 9 charts, 1 network request. Module-level Maps deduplicate in-flight fetches and cache resolved data for instant subsequent renders.
- **Per-chart and global reset** — reset buttons clear cross-filter selection and restore default zoom; the global reset remounts the entire provider tree.
- **Three data sources, three config sets** — streaming Issues API, Clinical Data API (six CDISC SDTM domains × 9 charts each), and static file; each source has its own config folder and field names.
- **NDJSON streaming** — Issues API streams rows via a `ReadableStream` reader; one fetch, one shared `StreamEntry`, all 9 charts subscribe and update live as rows arrive.
- **Clinical Data API with domain selector, filters, and pagination** — six CDISC SDTM domains (AE, CM, DM, LB, TV, VS); five filter fields (study, site, subject, visit, form) with 400 ms debounce; Prev/Next buttons plus direct page-number input and page-size selector (50–1000); `meta.total_pages` + `meta.total_records` shown inline; old URL evicted from cache before each page navigation.
- **Partial-data rendering** — charts appear as soon as the first rows land; a pulsing **● LIVE** badge overlays each card while the stream is still open.
- **Live toolbar counter** — `⚡ first row in Xs · N rows (streaming…)` updates in real time; `(streaming…)` drops when the stream closes.
- **Unified prefetch** — `prefetchData(url)` routes automatically to a JSON fetch or stream start based on the URL; no caller changes needed when switching endpoint types.
- **Loading gate** — one animated banner while data loads or until first stream rows arrive; all 9 charts appear simultaneously.
- **Publishable library** — `chart-lib` builds to ESM + CJS with full TypeScript declarations, all rendering libraries externalized, ready for `npm publish`.