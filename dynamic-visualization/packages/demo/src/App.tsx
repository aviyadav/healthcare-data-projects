import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  ChartProvider,
  useSelection,
  prefetchData,
  clearDataCache,
  isStreamUrl,
  useStreamStatus,
  RENDERER_META,
} from "chart-lib";
import type { RendererType } from "chart-lib";
import { ChartCard } from "./components/ChartCard";
import { DataTable } from "./components/DataTable";

// ---------------------------------------------------------------------------
// Data source definitions — hardcoded for demo
// Two independent source → config set mappings.
// ---------------------------------------------------------------------------

type DataSourceId = "file" | "clinical";

interface DataSource {
  id: DataSourceId;
  label: string;
  description: string;
  badge: string;
  badgeColor: string;
  dataUrl: string;
  configs: Array<{ url: string; label: string }>;
}

// ---------------------------------------------------------------------------
// Clinical Data API — domain, filter and pagination configuration
// ---------------------------------------------------------------------------

type ClinicalDomain = "AE" | "CM" | "DM" | "LB" | "TV" | "VS";

const CLINICAL_DOMAINS: ClinicalDomain[] = ["AE", "CM", "DM", "LB", "TV", "VS"];

const CLINICAL_DOMAIN_LABELS: Record<ClinicalDomain, string> = {
  AE: "Adverse Events",
  CM: "Concomitant Medications",
  DM: "Demographics",
  LB: "Laboratory Results",
  TV: "Trial Visits",
  VS: "Vital Signs",
};

const CLINICAL_FILTER_FIELDS = ["study", "site", "subject", "visit", "form"] as const;
type ClinicalFilterKey = (typeof CLINICAL_FILTER_FIELDS)[number];

const EMPTY_CLINICAL_FILTERS: Record<ClinicalFilterKey, string> = {
  study: "",
  site: "",
  subject: "",
  visit: "",
  form: "",
};

interface PaginationMeta {
  page: number;
  page_size: number;
  total_records: number;
  total_pages: number;
}

const CLINICAL_PAGE_SIZE_OPTIONS = [100, 500, 1000, 2000, 3000, 4000, 5000] as const;

function buildClinicalUrl(
  domain: ClinicalDomain,
  page: number,
  pageSize: number,
  filters: Record<ClinicalFilterKey, string>,
): string {
  const params = new URLSearchParams();
  for (const k of CLINICAL_FILTER_FIELDS) {
    const v = filters[k].trim();
    if (v) params.set(k, v);
  }
  params.set("page", String(page));
  params.set("page_size", String(pageSize));
  return `/api/v1/${domain.toLowerCase()}?${params.toString()}`;
}

const CLINICAL_CONFIGS: Record<ClinicalDomain, Array<{ url: string; label: string }>> = {
  AE: [
    { url: "/src/config/clinical/ae/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/ae/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/ae/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/ae/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/ae/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/ae/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/ae/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/ae/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/ae/sankey-chart.json", label: "Sankey" },
  ],
  CM: [
    { url: "/src/config/clinical/cm/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/cm/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/cm/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/cm/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/cm/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/cm/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/cm/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/cm/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/cm/sankey-chart.json", label: "Sankey" },
  ],
  DM: [
    { url: "/src/config/clinical/dm/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/dm/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/dm/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/dm/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/dm/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/dm/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/dm/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/dm/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/dm/sankey-chart.json", label: "Sankey" },
  ],
  LB: [
    { url: "/src/config/clinical/lb/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/lb/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/lb/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/lb/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/lb/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/lb/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/lb/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/lb/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/lb/sankey-chart.json", label: "Sankey" },
  ],
  TV: [
    { url: "/src/config/clinical/tv/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/tv/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/tv/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/tv/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/tv/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/tv/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/tv/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/tv/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/tv/sankey-chart.json", label: "Sankey" },
  ],
  VS: [
    { url: "/src/config/clinical/vs/line-chart.json", label: "Line" },
    { url: "/src/config/clinical/vs/bar-chart.json", label: "Bar" },
    { url: "/src/config/clinical/vs/scatter-chart.json", label: "Scatter" },
    { url: "/src/config/clinical/vs/pie-chart.json", label: "Pie / Donut" },
    { url: "/src/config/clinical/vs/area-chart.json", label: "Area" },
    { url: "/src/config/clinical/vs/heatmap-chart.json", label: "Heatmap" },
    { url: "/src/config/clinical/vs/histogram-chart.json", label: "Histogram" },
    { url: "/src/config/clinical/vs/box-chart.json", label: "Box Plot" },
    { url: "/src/config/clinical/vs/sankey-chart.json", label: "Sankey" },
  ],
};

const DATA_SOURCES: Record<"file", DataSource> = {
  file: {
    id: "file",
    label: "Sample Data File",
    description: "Static sales dataset from /src/data/sample-data.json",
    badge: "FILE",
    badgeColor: "#8e44ad",
    dataUrl: "/src/data/sample-data.json",
    configs: [
      { url: "/src/config/file/line-chart.json", label: "Line" },
      { url: "/src/config/file/bar-chart.json", label: "Bar" },
      { url: "/src/config/file/scatter-chart.json", label: "Scatter" },
      { url: "/src/config/file/pie-chart.json", label: "Pie / Donut" },
      { url: "/src/config/file/area-chart.json", label: "Area" },
      { url: "/src/config/file/heatmap-chart.json", label: "Heatmap" },
      { url: "/src/config/file/histogram-chart.json", label: "Histogram" },
      { url: "/src/config/file/box-chart.json", label: "Box Plot" },
      { url: "/src/config/file/sankey-chart.json", label: "Sankey" },
    ],
  },
} as const;

// ---------------------------------------------------------------------------
// useDataLoader
//
// Tracks the loading lifecycle of a data URL so the App can show a single
// loading banner and reveal all charts simultaneously when data arrives.
//
// How it works with the chart-lib module-level cache:
//   1. prefetchData(url) is called first — starts the real fetch and stores
//      the Promise in chart-lib's fetchPromiseCache (one network request).
//   2. This hook fires an independent fetch() to detect completion. On
//      localhost the browser typically reuses the in-flight TCP connection.
//   3. After the probe's JSON parse resolves we delay one requestAnimationFrame.
//      rAF is a macrotask and runs AFTER all microtasks, so by the time it
//      fires chart-lib's Zod validation + transformRows have also completed
//      and resolvedDataCache is populated.
//   4. ready = true → charts mount → useChartData finds data in the cache
//      instantly — zero per-chart loading flash.
// ---------------------------------------------------------------------------

function useDataLoader(dataUrl: string) {
  const [ready, setReady] = useState(false);
  const [loadMs, setLoadMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isStream = isStreamUrl(dataUrl);

  // Always call useStreamStatus — hooks must never be conditional.
  // For JSON sources it returns a zeroed-out snapshot and is effectively a no-op.
  const streamStatus = useStreamStatus(dataUrl);

  // Reset all state whenever the URL changes (covers both stream and JSON paths).
  useEffect(() => {
    setReady(false);
    setLoadMs(null);
    setError(null);
  }, [dataUrl]);

  // Stream path: become "ready" as soon as the first row arrives so charts
  // can start rendering with partial data while the stream continues.
  useEffect(() => {
    if (!isStream) return;
    if (streamStatus.error) {
      setError(streamStatus.error.message);
      setReady(true);
    } else if (streamStatus.firstRowMs !== null) {
      setLoadMs(Math.round(streamStatus.firstRowMs));
      setReady(true);
    }
  }, [isStream, streamStatus.error, streamStatus.firstRowMs]);

  // JSON path: fire an independent probe fetch and wait for the full body to
  // parse before declaring ready.  On localhost the browser typically reuses
  // the in-flight TCP connection started by prefetchData.
  useEffect(() => {
    if (isStream) return;

    let cancelled = false;
    const t0 = performance.now();

    fetch(dataUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(() => {
        if (cancelled) return;
        const elapsed = Math.round(performance.now() - t0);
        // Defer one animation frame so chart-lib's Zod + transform microtasks
        // finish before we unmount the banner and mount the chart grid.
        requestAnimationFrame(() => {
          if (!cancelled) {
            setLoadMs(elapsed);
            setReady(true);
          }
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        requestAnimationFrame(() => {
          if (!cancelled) setReady(true);
        });
      });

    return () => {
      cancelled = true;
    };
  }, [dataUrl, isStream]);

  return { ready, loadMs, error };
}

// ---------------------------------------------------------------------------
// ResetAllButton
// Must live inside <ChartProvider> to access useSelection.
// ---------------------------------------------------------------------------

const ResetAllButton: React.FC<{ onResetAll: () => void }> = ({
  onResetAll,
}) => {
  const { clearSelection, selection } = useSelection();
  const [busy, setBusy] = useState(false);

  const handleClick = () => {
    setBusy(true);
    clearSelection();
    onResetAll();
    setTimeout(() => setBusy(false), 600);
  };

  const hasFilter = selection !== null;

  return (
    <button
      onClick={handleClick}
      style={{
        ...st.resetAllBtn,
        ...(hasFilter && !busy ? st.resetAllBtnFiltered : {}),
        ...(busy ? st.resetAllBtnBusy : {}),
      }}
      title="Clear all cross-filter selections and reset zoom on every chart"
    >
      <span style={{ fontSize: 14, lineHeight: 1 }}>↺</span>
      Reset All Charts
      {hasFilter && <span style={st.filterDot} />}
    </button>
  );
};

// ---------------------------------------------------------------------------
// DataLoadingBanner
// Replaces the chart grid while the initial data fetch is in progress.
// ---------------------------------------------------------------------------

interface BannerProps {
  sourceLabel: string;
  badgeColor: string;
  dataUrl: string;
  error: string | null;
}

const DataLoadingBanner: React.FC<BannerProps> = ({
  sourceLabel,
  badgeColor,
  dataUrl,
  error,
}) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    setElapsed(0);
    const id = setInterval(() => setElapsed((s) => s + 100), 100);
    return () => clearInterval(id);
  }, [dataUrl]);

  return (
    <>
      <style>{`
        @keyframes dv-spin    { to { transform: rotate(360deg);  } }
        @keyframes dv-spinRev { to { transform: rotate(-360deg); } }
        @keyframes dv-slide {
          0%   { left: -45%; width: 45%; }
          60%  { left: 25%;  width: 65%; }
          100% { left: 110%; width: 45%; }
        }
      `}</style>

      <div style={bn.root}>
        <div style={bn.card}>
          {error ? (
            <>
              <div style={bn.errorIcon}>✕</div>
              <p style={bn.title}>Failed to load data</p>
              <p style={bn.subtitle}>{error}</p>
              <code style={bn.url}>{dataUrl}</code>
            </>
          ) : (
            <>
              {/* Dual-ring spinner */}
              <div style={bn.spinWrap}>
                <div style={bn.ring1} />
                <div style={bn.ring2} />
              </div>

              <p style={bn.title}>
                Fetching{" "}
                <span style={{ color: badgeColor, fontWeight: 700 }}>
                  {sourceLabel}
                </span>{" "}
                data…
              </p>

              <p style={bn.subtitle}>
                {isStreamUrl(dataUrl) ? (
                  "Streaming rows from the server — charts appear as soon as the first batch arrives."
                ) : (
                  <>
                    One shared request for all 8 charts.
                    <br />
                    Subsequent renders will be <strong>instant</strong>{" "}
                    (cached).
                  </>
                )}
              </p>

              <code style={bn.url}>{dataUrl}</code>

              <span style={bn.elapsed}>
                {(elapsed / 1000).toFixed(1)}s elapsed
              </span>

              {/* Indeterminate progress bar */}
              <div style={bn.track}>
                <div style={bn.bar} />
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [activeSource, setActiveSource] = useState<DataSourceId>("clinical");
  const [activeRenderer, setActiveRenderer] = useState<RendererType>("plotly");
  const [globalResetKey, setGlobalResetKey] = useState(0);
  const [layoutMode, setLayoutMode] = useState<"grid" | "horizontal" | "vertical">("grid");
  const [chartWidth, setChartWidth] = useState(420);

  // ── Clinical Data API state ───────────────────────────────────────────────
  const [clinicalDomain, setClinicalDomain] = useState<ClinicalDomain>("AE");
  const [clinicalPage, setClinicalPage] = useState(1);
  const [clinicalPageSize, setClinicalPageSize] = useState(100);
  const [clinicalFilters, setClinicalFilters] =
    useState<Record<ClinicalFilterKey, string>>(EMPTY_CLINICAL_FILTERS);
  const [debouncedClinicalFilters, setDebouncedClinicalFilters] =
    useState<Record<ClinicalFilterKey, string>>(EMPTY_CLINICAL_FILTERS);
  const [clinicalMeta, setClinicalMeta] = useState<PaginationMeta | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedClinicalFilters(clinicalFilters);
      setClinicalPage(1); // reset to first page on filter change
    }, 400);
    return () => clearTimeout(timer);
  }, [clinicalFilters]);

  // Derived clinical URL
  const clinicalDataUrl = useMemo(
    () => buildClinicalUrl(clinicalDomain, clinicalPage, clinicalPageSize, debouncedClinicalFilters),
    [clinicalDomain, clinicalPage, clinicalPageSize, debouncedClinicalFilters],
  );

  // The URL actually fed to useDataLoader / ChartCards.
  const activeDataUrl = useMemo(() => {
    if (activeSource === "file") return DATA_SOURCES.file.dataUrl;
    return clinicalDataUrl;
  }, [activeSource, clinicalDataUrl]);

  // ── Active chart configs ──────────────────────────────────────────────────
  const activeConfigs =
    activeSource === "clinical"
      ? CLINICAL_CONFIGS[clinicalDomain]
      : DATA_SOURCES.file.configs;

  // Source label / badge used in the toolbar and loading banner
  const currentSourceLabel =
    activeSource === "clinical"
      ? `Clinical Data API — ${clinicalDomain} (${CLINICAL_DOMAIN_LABELS[clinicalDomain]})`
      : DATA_SOURCES.file.label;
  const currentSourceBadgeColor =
    activeSource === "clinical"
      ? "#1abc9c"
      : DATA_SOURCES.file.badgeColor;

  // Whenever the active URL changes (source switch or filter change), clear
  // stale cached data for the previous URL and prefetch the new one so charts
  // find data in the cache synchronously when they mount.
  const prevActiveDataUrl = useRef<string>("");
  useEffect(() => {
    const prev = prevActiveDataUrl.current;
    const curr = activeDataUrl;
    if (prev && prev !== curr) {
      clearDataCache(prev);
    }
    prevActiveDataUrl.current = curr;
    prefetchData(curr);
  }, [activeDataUrl]);

  // Fetch pagination meta whenever the clinical URL changes
  useEffect(() => {
    if (activeSource !== "clinical") return;
    let cancelled = false;
    fetch(clinicalDataUrl)
      .then((r) => r.json())
      .then((json: unknown) => {
        if (cancelled) return;
        if (
          json &&
          typeof json === "object" &&
          "meta" in json &&
          json.meta &&
          typeof json.meta === "object"
        ) {
          setClinicalMeta(json.meta as PaginationMeta);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [clinicalDataUrl, activeSource]);

  // Track data readiness at the app level → single loading banner, all
  // charts appear simultaneously when data is ready.
  const { ready, loadMs, error: loadError } = useDataLoader(activeDataUrl);

  const handleSourceSwitch = (id: DataSourceId) => {
    if (id === activeSource) return;
    if (id === "clinical") {
      prefetchData(clinicalDataUrl);
    } else {
      prefetchData(DATA_SOURCES.file.dataUrl);
    }
    setActiveSource(id);
  };

  const handleResetAll = useCallback(() => {
    setGlobalResetKey((k) => k + 1);
  }, []);

  const handleRendererSwitch = (r: RendererType) => {
    if (r === activeRenderer) return;
    setActiveRenderer(r);
    setGlobalResetKey((k) => k + 1);
  };

  // ── Clinical handlers ─────────────────────────────────────────────────────

  const handleDomainChange = useCallback(
    (domain: ClinicalDomain) => {
      if (domain === clinicalDomain) return;
      clearDataCache(clinicalDataUrl);
      setClinicalDomain(domain);
      setClinicalPage(1);
      setClinicalMeta(null);
    },
    [clinicalDomain, clinicalDataUrl],
  );

  const handleClinicalPageChange = useCallback(
    (newPage: number) => {
      if (newPage < 1) return;
      if (clinicalMeta && newPage > clinicalMeta.total_pages) return;
      clearDataCache(clinicalDataUrl);
      setClinicalPage(newPage);
    },
    [clinicalMeta, clinicalDataUrl],
  );

  const handleClinicalPageSizeChange = useCallback(
    (newSize: number) => {
      clearDataCache(clinicalDataUrl);
      setClinicalPageSize(newSize);
      setClinicalPage(1);
      setClinicalMeta(null);
    },
    [clinicalDataUrl],
  );

  const handleClinicalFilterChange = useCallback(
    (field: ClinicalFilterKey, value: string) => {
      setClinicalFilters((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  const handleClearClinicalFilters = useCallback(() => {
    setClinicalFilters(EMPTY_CLINICAL_FILTERS);
    setDebouncedClinicalFilters(EMPTY_CLINICAL_FILTERS);
    setClinicalPage(1);
  }, []);

  const hasClinicalFilters = CLINICAL_FILTER_FIELDS.some(
    (f) => clinicalFilters[f].trim() !== "",
  );

  return (
    <div style={st.root}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header style={st.header}>
        <div style={st.titleRow}>
          <img
            src="/random-logo.png"
            alt="Logo"
            style={st.logo}
          />
          <h1 style={st.title}>Clinical Data Visualization Demo</h1>
          <p style={st.subtitle}>
            9 chart types · 3 rendering libraries · Cross-filtering ·
            Single-fetch cache
          </p>
        </div>

        {/* ── Renderer selector ── */}
        <div style={st.rendererRow}>
          <span style={st.sourceRowLabel}>Library</span>
          <div style={st.tabs}>
            {Object.values(RENDERER_META).map((meta) => {
              const active = meta.type === activeRenderer;
              return (
                <button
                  key={meta.type}
                  onClick={() => handleRendererSwitch(meta.type)}
                  style={{
                    ...st.tab,
                    ...(active
                      ? {
                          ...st.tabActive,
                          boxShadow: `0 -2px 0 ${meta.color} inset`,
                        }
                      : st.tabInactive),
                  }}
                  title={meta.description}
                >
                  <span
                    style={{
                      ...st.tabBadge,
                      background: active ? meta.color : "#666",
                    }}
                  >
                    {meta.badge}
                  </span>
                  {meta.label}
                </button>
              );
            })}
          </div>
          <span style={st.rendererDesc}>
            {RENDERER_META[activeRenderer].description}
          </span>
        </div>

        {/* ── Source selector tab strip ── */}
        <div style={st.sourceRow}>
          <span style={st.sourceRowLabel}>Data Source</span>

          <div style={st.tabs}>
            {/* Clinical Data API tab — first */}
            <button
              onClick={() => handleSourceSwitch("clinical")}
              style={{
                ...st.tab,
                ...(activeSource === "clinical" ? st.tabActive : st.tabInactive),
              }}
              title="Paginated Clinical Trial Data API (AE, CM, DM, LB, TV, VS)"
            >
              <span
                style={{
                  ...st.tabBadge,
                  background: activeSource === "clinical" ? "#1abc9c" : "#777",
                }}
              >
                CLIN
              </span>
              Clinical Data API
            </button>
            <button
              onClick={() => handleSourceSwitch("file")}
              style={{
                ...st.tab,
                ...(activeSource === "file" ? st.tabActive : st.tabInactive),
              }}
              title={DATA_SOURCES.file.description}
            >
              <span
                style={{
                  ...st.tabBadge,
                  background: activeSource === "file" ? DATA_SOURCES.file.badgeColor : "#777",
                }}
              >
                {DATA_SOURCES.file.badge}
              </span>
              {DATA_SOURCES.file.label}
            </button>
          </div>

          <div style={st.sourceDesc}>
            <span style={{ opacity: 0.4 }}>→</span>
            <span>
              {activeSource === "clinical"
                ? `Clinical Data API — ${clinicalDomain}: ${CLINICAL_DOMAIN_LABELS[clinicalDomain]}`
                : DATA_SOURCES.file.description}
            </span>
            <code style={st.sourceUrl}>{activeDataUrl}</code>
          </div>
        </div>

        {/* ── Clinical Domain selector ── */}
        {activeSource === "clinical" && (
          <div style={st.clinicalDomainRow}>
            <span style={st.sourceRowLabel}>Domain</span>
            <div style={st.tabs}>
              {CLINICAL_DOMAINS.map((domain) => {
                const active = domain === clinicalDomain;
                return (
                  <button
                    key={domain}
                    onClick={() => handleDomainChange(domain)}
                    style={{
                      ...st.tab,
                      ...(active ? st.tabActive : st.tabInactive),
                      ...(active ? { boxShadow: "0 -2px 0 #1abc9c inset" } : {}),
                    }}
                    title={CLINICAL_DOMAIN_LABELS[domain]}
                  >
                    <span
                      style={{
                        ...st.tabBadge,
                        background: active ? "#1abc9c" : "#555",
                      }}
                    >
                      {domain}
                    </span>
                    {CLINICAL_DOMAIN_LABELS[domain]}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Clinical filter + pagination panel ── */}
        {activeSource === "clinical" && (
          <div style={st.filterRow}>
            {/* Filter inputs */}
            <div style={st.filterRowHead}>
              <span style={st.filterRowLabel}>Filters</span>
              {hasClinicalFilters && (
                <button
                  onClick={handleClearClinicalFilters}
                  style={st.clearFiltersBtn}
                  title="Clear all clinical filter values"
                >
                  ✕ Clear
                </button>
              )}
              {hasClinicalFilters && (
                <span style={st.filterPending}>
                  {JSON.stringify(debouncedClinicalFilters) !==
                  JSON.stringify(clinicalFilters) ? (
                    <span style={{ color: "#e67e22" }}>● applying…</span>
                  ) : (
                    <span style={{ color: "#27ae60" }}>● active</span>
                  )}
                </span>
              )}
            </div>
            <div style={st.filterGrid}>
              {CLINICAL_FILTER_FIELDS.map((field) => (
                <div key={field} style={st.filterItem}>
                  <label style={st.filterLabel} htmlFor={`cf-${field}`}>
                    {field}
                  </label>
                  <input
                    id={`cf-${field}`}
                    type="text"
                    value={clinicalFilters[field]}
                    onChange={(e) => handleClinicalFilterChange(field, e.target.value)}
                    style={{
                      ...st.filterInput,
                      ...(clinicalFilters[field] !== debouncedClinicalFilters[field]
                        ? st.filterInputPending
                        : clinicalFilters[field]
                          ? st.filterInputActive
                          : {}),
                    }}
                    placeholder="any"
                    autoComplete="off"
                    spellCheck={false}
                  />
                </div>
              ))}
            </div>

            {/* Pagination controls */}
            <div style={st.paginationRow}>
              <span style={st.filterRowLabel}>Page</span>

              {/* Prev button */}
              <button
                onClick={() => handleClinicalPageChange(clinicalPage - 1)}
                disabled={clinicalPage <= 1}
                style={{
                  ...st.pageBtn,
                  ...(clinicalPage <= 1 ? st.pageBtnDisabled : {}),
                }}
              >
                ‹ Prev
              </button>

              {/* Page number input */}
              <span style={st.pageInfo}>
                <input
                  type="number"
                  min={1}
                  max={clinicalMeta?.total_pages ?? 1}
                  value={clinicalPage}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!isNaN(v)) handleClinicalPageChange(v);
                  }}
                  style={st.pageInput}
                />
                <span style={{ opacity: 0.55, fontSize: 11 }}>
                  {clinicalMeta
                    ? ` / ${clinicalMeta.total_pages.toLocaleString()} pages (${clinicalMeta.total_records.toLocaleString()} records)`
                    : "/ …"}
                </span>
              </span>

              {/* Next button */}
              <button
                onClick={() => handleClinicalPageChange(clinicalPage + 1)}
                disabled={
                  clinicalMeta !== null && clinicalPage >= clinicalMeta.total_pages
                }
                style={{
                  ...st.pageBtn,
                  ...(clinicalMeta !== null && clinicalPage >= clinicalMeta.total_pages
                    ? st.pageBtnDisabled
                    : {}),
                }}
              >
                Next ›
              </button>

              {/* Page size selector */}
              <span style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 12 }}>
                <span style={st.filterRowLabel}>Page size</span>
                <select
                  value={clinicalPageSize}
                  onChange={(e) => handleClinicalPageSizeChange(Number(e.target.value))}
                  style={st.pageSizeSelect}
                >
                  {CLINICAL_PAGE_SIZE_OPTIONS.map((s) => (
                    <option key={s} value={s}>
                      {s.toLocaleString()} rows
                    </option>
                  ))}
                </select>
              </span>
            </div>
          </div>
        )}

      </header>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div style={st.body}>
        <ChartProvider
          key={`${activeSource}-${activeSource === "clinical" ? clinicalDomain : ""}-${activeRenderer}-${globalResetKey}`}
          renderer={activeRenderer}
        >
          {/* Toolbar above the chart grid */}
          <div style={st.toolbar}>
            <span style={st.toolbarInfo}>
              <span
                style={{
                  ...st.sourcePill,
                  background: currentSourceBadgeColor,
                }}
              >
                {activeSource === "clinical" ? clinicalDomain : DATA_SOURCES.file.badge}
              </span>
              {currentSourceLabel} ·{" "}
              <strong>{activeConfigs.length} charts</strong>
              {ready && loadMs !== null && (
                <span style={st.loadTimeBadge}>
                  ⚡ loaded in {(loadMs / 1000).toFixed(2)}s
                </span>
              )}
              {activeSource === "clinical" && clinicalMeta && (
                <span style={st.pageInfoBadge}>
                  Page {clinicalMeta.page} / {clinicalMeta.total_pages.toLocaleString()} · {clinicalMeta.total_records.toLocaleString()} records
                </span>
              )}
            </span>
            <span
              style={{
                ...st.tabBadge,
                background: RENDERER_META[activeRenderer].color,
                fontSize: 10,
                padding: "2px 8px",
              }}
            >
              {RENDERER_META[activeRenderer].label}
            </span>

            <ResetAllButton onResetAll={handleResetAll} />

            {/* ── Layout toggle ── */}
            <div style={st.layoutToggleGroup}>
              <button
                onClick={() => setLayoutMode("grid")}
                style={{
                  ...st.layoutBtn,
                  ...(layoutMode === "grid" ? st.layoutBtnActive : st.layoutBtnInactive),
                }}
                title="Grid layout"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <rect x="0" y="0" width="6" height="6" rx="1"/>
                  <rect x="8" y="0" width="6" height="6" rx="1"/>
                  <rect x="0" y="8" width="6" height="6" rx="1"/>
                  <rect x="8" y="8" width="6" height="6" rx="1"/>
                </svg>
                Grid
              </button>
              <button
                onClick={() => setLayoutMode("horizontal")}
                style={{
                  ...st.layoutBtn,
                  ...(layoutMode === "horizontal" ? st.layoutBtnActive : st.layoutBtnInactive),
                }}
                title="Horizontal scroll layout"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <rect x="0" y="2" width="4" height="10" rx="1"/>
                  <rect x="5" y="2" width="4" height="10" rx="1"/>
                  <rect x="10" y="2" width="4" height="10" rx="1"/>
                </svg>
                Horizontal
              </button>
              <button
                onClick={() => setLayoutMode("vertical")}
                style={{
                  ...st.layoutBtn,
                  ...(layoutMode === "vertical" ? st.layoutBtnActive : st.layoutBtnInactive),
                }}
                title="Vertical scroll layout"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <rect x="2" y="0" width="10" height="4" rx="1"/>
                  <rect x="2" y="5" width="10" height="4" rx="1"/>
                  <rect x="2" y="10" width="10" height="4" rx="1"/>
                </svg>
                Vertical
              </button>
            </div>

            {/* ── Chart width/height slider (scroll modes only) ── */}
            {(layoutMode === "horizontal" || layoutMode === "vertical") && (
              <div style={st.widthSliderGroup}>
                <span style={st.widthSliderLabel}>
                  {layoutMode === "horizontal" ? "Width" : "Height"}
                </span>
                <input
                  type="range"
                  min={280}
                  max={900}
                  step={10}
                  value={chartWidth}
                  onChange={(e) => setChartWidth(Number(e.target.value))}
                  style={st.widthSlider}
                />
                <span style={st.widthSliderValue}>{chartWidth}px</span>
              </div>
            )}
          </div>

          {/* Loading gate */}
          {!ready ? (
            <DataLoadingBanner
              sourceLabel={currentSourceLabel}
              badgeColor={currentSourceBadgeColor}
              dataUrl={activeDataUrl}
              error={loadError}
            />
          ) : layoutMode === "horizontal" ? (
            <div style={st.horizontalScroll}>
              {activeConfigs.map(({ url, label }) => (
                <div
                  key={`${activeSource}-${activeSource === "clinical" ? clinicalDomain : ""}-${url}`}
                  style={{ ...st.horizontalItem, width: chartWidth, flexShrink: 0 }}
                >
                  <ChartCard
                    label={label}
                    configUrl={url}
                    dataUrl={activeDataUrl}
                  />
                </div>
              ))}
            </div>
          ) : layoutMode === "vertical" ? (
            <div style={st.verticalScroll}>
              {activeConfigs.map(({ url, label }) => (
                <div
                  key={`${activeSource}-${activeSource === "clinical" ? clinicalDomain : ""}-${url}`}
                  style={{ ...st.verticalItem, height: chartWidth }}
                >
                  <ChartCard
                    label={label}
                    configUrl={url}
                    dataUrl={activeDataUrl}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div style={st.grid}>
              {activeConfigs.map(({ url, label }) => (
                <ChartCard
                  key={`${activeSource}-${activeSource === "clinical" ? clinicalDomain : ""}-${url}`}
                  label={label}
                  configUrl={url}
                  dataUrl={activeDataUrl}
                />
              ))}
            </div>
          )}

          {/* ── Data Table — full width below charts ── */}
          {ready && <DataTable dataUrl={activeDataUrl} />}
        </ChartProvider>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App styles
// ---------------------------------------------------------------------------

const st = {
  root: {
    fontFamily: "'Inter', system-ui, sans-serif",
    background: "#f0f2f5",
    minHeight: "100vh",
    color: "#1a1a2e",
  } satisfies React.CSSProperties,

  // Header
  header: {
    background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
    color: "#fff",
    padding: "20px 32px 0",
    boxShadow: "0 2px 12px rgba(0,0,0,.25)",
  } satisfies React.CSSProperties,

  titleRow: {
    marginBottom: 14,
  } satisfies React.CSSProperties,

  logo: {
    height: 36,
    width: "auto",
    display: "block",
    marginBottom: 10,
  } satisfies React.CSSProperties,

  // Renderer selector row
  rendererRow: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    flexWrap: "wrap" as const,
    borderTop: "1px solid rgba(255,255,255,.1)",
    paddingTop: 12,
    paddingBottom: 12,
  } satisfies React.CSSProperties,

  rendererDesc: {
    fontSize: 11,
    color: "rgba(255,255,255,.45)",
    flex: 1,
  } satisfies React.CSSProperties,

  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    letterSpacing: "-.01em",
  } satisfies React.CSSProperties,

  subtitle: {
    margin: "4px 0 0",
    fontSize: 12,
    opacity: 0.6,
    letterSpacing: ".02em",
  } satisfies React.CSSProperties,

  // Source selector
  sourceRow: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    flexWrap: "wrap" as const,
    borderTop: "1px solid rgba(255,255,255,.1)",
    paddingTop: 14,
    paddingBottom: 14,
  } satisfies React.CSSProperties,

  sourceRowLabel: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase" as const,
    letterSpacing: ".09em",
    opacity: 0.5,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  tabs: {
    display: "flex",
    gap: 6,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  tab: {
    display: "inline-flex",
    alignItems: "center",
    gap: 7,
    padding: "7px 16px",
    fontSize: 13,
    fontWeight: 600,
    borderRadius: "6px 6px 0 0",
    border: "none",
    cursor: "pointer",
    transition: "all 0.18s",
    outline: "none",
    letterSpacing: ".01em",
  } satisfies React.CSSProperties,

  tabActive: {
    background: "#fff",
    color: "#1a1a2e",
    boxShadow: "0 -2px 0 #3498db inset",
  } satisfies React.CSSProperties,

  tabInactive: {
    background: "rgba(255,255,255,.1)",
    color: "rgba(255,255,255,.7)",
  } satisfies React.CSSProperties,

  tabBadge: {
    fontSize: 9,
    fontWeight: 800,
    letterSpacing: ".08em",
    color: "#fff",
    borderRadius: 3,
    padding: "1px 5px",
  } satisfies React.CSSProperties,

  sourceDesc: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 11,
    color: "rgba(255,255,255,.5)",
    flex: 1,
    flexWrap: "wrap" as const,
  } satisfies React.CSSProperties,

  sourceUrl: {
    fontSize: 10,
    background: "rgba(255,255,255,.1)",
    borderRadius: 3,
    padding: "1px 6px",
    color: "rgba(255,255,255,.65)",
  } satisfies React.CSSProperties,

  // Paginated slider row — kept for reference but no longer rendered
  sliderRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap" as const,
    borderTop: "1px solid rgba(255,255,255,.1)",
    paddingTop: 12,
    paddingBottom: 12,
  } satisfies React.CSSProperties,

  sliderGroup: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  } satisfies React.CSSProperties,

  sliderLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: "#555",
    minWidth: 120,
  } satisfies React.CSSProperties,

  sliderInput: {
    width: 140,
    cursor: "pointer",
    accentColor: "#e67e22",
  } satisfies React.CSSProperties,

  sliderUrlPill: {
    fontSize: 10,
    background: "#fff3e0",
    borderRadius: 3,
    padding: "1px 6px",
    color: "#c0392b",
  } satisfies React.CSSProperties,

  // Clinical domain selector row
  clinicalDomainRow: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    flexWrap: "wrap" as const,
    borderTop: "1px solid rgba(255,255,255,.1)",
    paddingTop: 12,
    paddingBottom: 12,
  } satisfies React.CSSProperties,

  // Pagination controls row
  paginationRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    flexWrap: "wrap" as const,
    marginTop: 8,
  } satisfies React.CSSProperties,

  pageBtn: {
    display: "inline-flex",
    alignItems: "center",
    padding: "5px 14px",
    fontSize: 12,
    fontWeight: 700,
    color: "#fff",
    background: "#1abc9c",
    border: "none",
    borderRadius: 5,
    cursor: "pointer",
    outline: "none",
    transition: "background 0.15s",
    letterSpacing: ".02em",
  } satisfies React.CSSProperties,

  pageBtnDisabled: {
    background: "rgba(255,255,255,.15)",
    color: "rgba(255,255,255,.35)",
    cursor: "not-allowed",
  } satisfies React.CSSProperties,

  pageInfo: {
    display: "flex",
    alignItems: "center",
    gap: 4,
    fontSize: 12,
    color: "rgba(255,255,255,.8)",
    fontVariantNumeric: "tabular-nums" as const,
  } satisfies React.CSSProperties,

  pageInput: {
    width: 56,
    textAlign: "center" as const,
    fontSize: 12,
    fontWeight: 700,
    padding: "4px 6px",
    borderRadius: 4,
    border: "1px solid rgba(255,255,255,.25)",
    background: "rgba(255,255,255,.12)",
    color: "#fff",
    outline: "none",
  } satisfies React.CSSProperties,

  pageSizeSelect: {
    fontSize: 11,
    fontWeight: 600,
    padding: "4px 8px",
    borderRadius: 4,
    border: "1px solid rgba(255,255,255,.2)",
    background: "rgba(255,255,255,.1)",
    color: "#fff",
    outline: "none",
    cursor: "pointer",
  } satisfies React.CSSProperties,

  pageInfoBadge: {
    fontSize: 11,
    fontWeight: 600,
    color: "#1abc9c",
    background: "rgba(26,188,156,.12)",
    border: "1px solid rgba(26,188,156,.4)",
    borderRadius: 4,
    padding: "1px 7px",
    letterSpacing: ".02em",
    marginLeft: 2,
  } satisfies React.CSSProperties,

  // API filter panel
  filterRow: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 8,
    borderTop: "1px solid rgba(255,255,255,.1)",
    paddingTop: 12,
    paddingBottom: 14,
  } satisfies React.CSSProperties,

  filterRowHead: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  } satisfies React.CSSProperties,

  filterRowLabel: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase" as const,
    letterSpacing: ".09em",
    opacity: 0.5,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  clearFiltersBtn: {
    fontSize: 11,
    fontWeight: 700,
    color: "#e74c3c",
    background: "rgba(231,76,60,.12)",
    border: "1px solid rgba(231,76,60,.35)",
    borderRadius: 4,
    padding: "2px 9px",
    cursor: "pointer",
    outline: "none",
  } satisfies React.CSSProperties,

  filterPending: {
    fontSize: 11,
    fontWeight: 700,
  } satisfies React.CSSProperties,

  filterGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
    gap: "6px 12px",
  } satisfies React.CSSProperties,

  filterItem: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 2,
  } satisfies React.CSSProperties,

  filterLabel: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: "uppercase" as const,
    letterSpacing: ".06em",
    color: "rgba(255,255,255,.5)",
  } satisfies React.CSSProperties,

  filterInput: {
    fontSize: 12,
    padding: "4px 8px",
    borderRadius: 4,
    border: "1px solid rgba(255,255,255,.2)",
    background: "rgba(255,255,255,.08)",
    color: "#fff",
    outline: "none",
    transition: "border-color 0.15s, background 0.15s",
    width: "100%",
    boxSizing: "border-box" as const,
  } satisfies React.CSSProperties,

  filterInputPending: {
    borderColor: "#e67e22",
    background: "rgba(230,126,34,.15)",
  } satisfies React.CSSProperties,

  filterInputActive: {
    borderColor: "#27ae60",
    background: "rgba(39,174,96,.12)",
  } satisfies React.CSSProperties,

  // Body
  body: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 0,
    padding: "20px 24px",
  } satisfies React.CSSProperties,

  // Toolbar
  toolbar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 16,
    padding: "0 2px",
  } satisfies React.CSSProperties,

  toolbarInfo: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 13,
    color: "#555",
  } satisfies React.CSSProperties,

  sourcePill: {
    fontSize: 10,
    fontWeight: 800,
    color: "#fff",
    borderRadius: 4,
    padding: "2px 7px",
    letterSpacing: ".06em",
  } satisfies React.CSSProperties,

  loadTimeBadge: {
    fontSize: 11,
    fontWeight: 600,
    color: "#27ae60",
    background: "#eafaf1",
    border: "1px solid #a9dfbf",
    borderRadius: 4,
    padding: "1px 7px",
    letterSpacing: ".02em",
    marginLeft: 2,
  } satisfies React.CSSProperties,

  // Reset All button
  resetAllBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "7px 16px",
    fontSize: 12,
    fontWeight: 700,
    color: "#555",
    background: "#fff",
    border: "1.5px solid #ddd",
    borderRadius: 6,
    cursor: "pointer",
    outline: "none",
    transition: "all 0.15s",
    letterSpacing: ".02em",
  } satisfies React.CSSProperties,

  resetAllBtnFiltered: {
    borderColor: "#e67e22",
    color: "#e67e22",
    background: "#fef9f2",
  } satisfies React.CSSProperties,

  resetAllBtnBusy: {
    borderColor: "#3498db",
    color: "#3498db",
    background: "#eaf4fb",
  } satisfies React.CSSProperties,

  filterDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#e67e22",
    display: "inline-block",
    marginLeft: 2,
  } satisfies React.CSSProperties,

  // Chart grid — fixed 3-column layout
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 20,
  } satisfies React.CSSProperties,

  // Horizontal scroll layout
  horizontalScroll: {
    display: "flex",
    flexDirection: "row" as const,
    gap: 16,
    overflowX: "auto" as const,
    paddingBottom: 12,
    scrollbarWidth: "thin" as const,
  } satisfies React.CSSProperties,

  horizontalItem: {
    flexShrink: 0,
  } satisfies React.CSSProperties,

  // Vertical scroll layout
  verticalScroll: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 16,
  } satisfies React.CSSProperties,

  verticalItem: {
    width: "100%",
    display: "flex",
    flexDirection: "column" as const,
    minHeight: 0,
  } satisfies React.CSSProperties,

  // Layout toggle buttons
  layoutToggleGroup: {
    display: "flex",
    gap: 2,
    background: "#f0f0f0",
    borderRadius: 7,
    padding: 2,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  layoutBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    padding: "5px 11px",
    fontSize: 12,
    fontWeight: 600,
    borderRadius: 5,
    border: "none",
    cursor: "pointer",
    outline: "none",
    transition: "all 0.15s",
    letterSpacing: ".01em",
  } satisfies React.CSSProperties,

  layoutBtnActive: {
    background: "#fff",
    color: "#1a1a2e",
    boxShadow: "0 1px 4px rgba(0,0,0,.12)",
  } satisfies React.CSSProperties,

  layoutBtnInactive: {
    background: "transparent",
    color: "#999",
  } satisfies React.CSSProperties,

  // Width slider (horizontal mode)
  widthSliderGroup: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  widthSliderLabel: {
    fontSize: 11,
    fontWeight: 700,
    color: "#888",
    letterSpacing: ".05em",
  } satisfies React.CSSProperties,

  widthSlider: {
    width: 110,
    accentColor: "#3498db",
    cursor: "pointer",
  } satisfies React.CSSProperties,

  widthSliderValue: {
    fontSize: 11,
    fontWeight: 600,
    color: "#555",
    minWidth: 40,
    fontVariantNumeric: "tabular-nums" as const,
  } satisfies React.CSSProperties,
} as const;

// ---------------------------------------------------------------------------
// DataLoadingBanner styles
// ---------------------------------------------------------------------------

const bn = {
  root: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    minHeight: 460,
    padding: 24,
    flex: 1,
  } satisfies React.CSSProperties,

  card: {
    background: "#fff",
    borderRadius: 14,
    boxShadow: "0 4px 28px rgba(0,0,0,.09)",
    padding: "44px 52px",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 14,
    maxWidth: 460,
    width: "100%",
    textAlign: "center" as const,
  } satisfies React.CSSProperties,

  spinWrap: {
    position: "relative" as const,
    width: 60,
    height: 60,
    marginBottom: 6,
  } satisfies React.CSSProperties,

  ring1: {
    position: "absolute" as const,
    inset: 0,
    borderRadius: "50%",
    border: "4px solid #e8f4fd",
    borderTopColor: "#3498db",
    animation: "dv-spin 0.85s linear infinite",
  } satisfies React.CSSProperties,

  ring2: {
    position: "absolute" as const,
    inset: 11,
    borderRadius: "50%",
    border: "3px solid #fef3e2",
    borderBottomColor: "#e67e22",
    animation: "dv-spinRev 1.3s linear infinite",
  } satisfies React.CSSProperties,

  title: {
    margin: 0,
    fontSize: 19,
    fontWeight: 600,
    color: "#1a1a2e",
    lineHeight: 1.35,
  } satisfies React.CSSProperties,

  subtitle: {
    margin: 0,
    fontSize: 13,
    color: "#888",
    lineHeight: 1.65,
    maxWidth: 360,
  } satisfies React.CSSProperties,

  url: {
    fontSize: 11,
    color: "#aaa",
    background: "#f8f8f8",
    border: "1px solid #eee",
    borderRadius: 5,
    padding: "4px 12px",
  } satisfies React.CSSProperties,

  elapsed: {
    fontSize: 12,
    color: "#c0c0c0",
    fontVariantNumeric: "tabular-nums" as const,
    letterSpacing: ".04em",
  } satisfies React.CSSProperties,

  track: {
    width: "100%",
    height: 4,
    background: "#f0f0f0",
    borderRadius: 2,
    overflow: "hidden" as const,
    position: "relative" as const,
    marginTop: 2,
  } satisfies React.CSSProperties,

  bar: {
    position: "absolute" as const,
    top: 0,
    height: "100%",
    background: "linear-gradient(90deg, #3498db, #9b59b6)",
    borderRadius: 2,
    animation: "dv-slide 1.6s ease-in-out infinite",
  } satisfies React.CSSProperties,

  errorIcon: {
    width: 52,
    height: 52,
    borderRadius: "50%",
    background: "#fdf2f2",
    border: "2px solid #e74c3c",
    color: "#e74c3c",
    fontSize: 22,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  } satisfies React.CSSProperties,
} as const;
