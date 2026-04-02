import { useEffect, useMemo, useReducer, useState } from "react";
import {
  ChartDataSchema,
  ChartDataRowSchema,
  type ChartDataRow,
} from "../types/ChartData";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface UseChartDataResult {
  data: ChartDataRow[];
  loading: boolean;
  error: Error | null;
  /** True while an NDJSON stream for this URL is still open. */
  streaming: boolean;
}

/** Public snapshot of a stream's progress — used by useStreamStatus. */
export interface StreamStatus {
  rowCount: number;
  /** True while the stream is still open (not yet done or errored). */
  streaming: boolean;
  /** ms elapsed from fetch start until the first row was parsed. */
  firstRowMs: number | null;
  /** ms elapsed from fetch start until the stream closed. null if still open. */
  lastRowMs: number | null;
  error: Error | null;
}

// ---------------------------------------------------------------------------
// Module-level JSON cache (unchanged from original)
//
//   resolvedDataCache  URL → ChartDataRow[]
//     Written once a fetch + parse + transform completes.
//     Subsequent hook calls return the array synchronously.
//
//   fetchPromiseCache  URL → Promise<ChartDataRow[]>
//     Written when a JSON fetch begins, deleted when it settles.
//     Multiple charts that mount while a fetch is in-flight all attach to
//     the same Promise — exactly one network request.
// ---------------------------------------------------------------------------

const resolvedDataCache = new Map<string, ChartDataRow[]>();
const fetchPromiseCache = new Map<string, Promise<ChartDataRow[]>>();

// ---------------------------------------------------------------------------
// Module-level stream cache
//
//   streamCache  URL → StreamEntry
//     Created as soon as getOrStartStream() is called (synchronously).
//     All charts that share the same stream URL attach their update callback
//     to entry.listeners — one fetch, many live subscribers.
// ---------------------------------------------------------------------------

interface StreamEntry {
  rows: ChartDataRow[];
  done: boolean;
  error: Error | null;
  firstRowMs: number | null;
  lastRowMs: number | null;
  listeners: Set<() => void>;
}

const streamCache = new Map<string, StreamEntry>();

// ---------------------------------------------------------------------------
// Stream URL detection
//
// Matches any URL whose path segment (after the last '/') starts with or
// equals "stream", e.g. /api/v1/issues/stream or /data/stream?foo=1.
// Export so App.tsx can use the same predicate without duplicating logic.
// ---------------------------------------------------------------------------

export function isStreamUrl(url: string): boolean {
  // Use the pathname segment so query strings / fragments don't interfere.
  try {
    const pathname = new URL(url, "http://x").pathname;
    const last = pathname.split("/").filter(Boolean).pop() ?? "";
    return last.startsWith("stream");
  } catch {
    return url.includes("/stream");
  }
}

// ---------------------------------------------------------------------------
// Issue-row enrichment (unchanged from original)
// Adds derived fields useful for visualising issue / clinical data.
// ---------------------------------------------------------------------------

function transformRows(rows: ChartDataRow[]): ChartDataRow[] {
  return rows.map((row): ChartDataRow => {
    const result: ChartDataRow = { ...row, count: 1 };

    // status_label
    const status = String(row["status"] ?? "");
    if (status) {
      result["status_label"] =
        status === "ISSUE_STATE_OPEN"
          ? "Open"
          : status === "ISSUE_STATE_CLSD"
            ? "Closed"
            : status;
    }

    // priority_label
    const priority = String(row["priority"] ?? "");
    if (priority) {
      result["priority_label"] = priority.includes("High")
        ? "High"
        : priority.includes("Medium")
          ? "Medium"
          : priority.includes("Low")
            ? "Low"
            : priority;
    }

    // created_month  ──  YYYY-MM substring from ISO timestamp
    const createdTs = String(row["created_ts"] ?? "");
    if (createdTs.length >= 7) {
      result["created_month"] = createdTs.substring(0, 7);
    }

    // created_hour  ──  0-23 integer extracted from "YYYY-MM-DDTHH:…"
    if (createdTs.length >= 13) {
      const h = parseInt(createdTs.substring(11, 13), 10);
      if (!isNaN(h)) result["created_hour"] = h;
    }

    return result;
  });
}

// ---------------------------------------------------------------------------
// NDJSON stream reader
//
// Reads the response body chunk-by-chunk, splits on newlines, validates each
// line as a ChartDataRow, enriches it, and appends it to entry.rows.
// Every time new rows land all registered listeners are called so subscribed
// hooks can re-render their charts with the latest partial dataset.
// ---------------------------------------------------------------------------

async function readNdJsonStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  entry: StreamEntry,
  t0: number,
  url: string,
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";

  const flushLine = (line: string): void => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let raw: unknown;
    try {
      raw = JSON.parse(trimmed);
    } catch {
      // Skip malformed lines — don't abort the whole stream.
      return;
    }

    const parsed = ChartDataRowSchema.safeParse(raw);
    if (!parsed.success) return;

    const enriched = transformRows([parsed.data]);

    if (entry.firstRowMs === null) {
      entry.firstRowMs = performance.now() - t0;
    }

    // Produce a new array reference on every update so React can detect the
    // change via reference equality (avoids stale-closure issues in hooks).
    entry.rows = [...entry.rows, ...enriched];
    entry.listeners.forEach((fn) => fn());
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // The last element is a potentially incomplete line — keep it in the
      // buffer until the next chunk (or stream end) provides the newline.
      buffer = lines.pop()!;
      lines.forEach(flushLine);
    }

    // Process any trailing content that arrived without a final newline.
    if (buffer.trim()) {
      flushLine(buffer);
    }

    entry.lastRowMs = performance.now() - t0;
    entry.done = true;
    // Promote to the resolved cache so JSON-path callers (e.g. a fresh
    // mount after the stream has closed) get data synchronously.
    resolvedDataCache.set(url, entry.rows);
    entry.listeners.forEach((fn) => fn());
  } catch (err) {
    entry.error = err instanceof Error ? err : new Error(String(err));
    entry.done = true;
    entry.listeners.forEach((fn) => fn());
  }
}

// ---------------------------------------------------------------------------
// getOrStartStream
//
// Creates or returns an existing StreamEntry for the given URL and kicks off
// the fetch + reader pipeline if this is the first call.  All subsequent
// calls (from the other N charts) receive the same entry and attach their own
// listener — no second network request.
// ---------------------------------------------------------------------------

function getOrStartStream(url: string): StreamEntry {
  const existing = streamCache.get(url);
  if (existing) return existing;

  const entry: StreamEntry = {
    rows: [],
    done: false,
    error: null,
    firstRowMs: null,
    lastRowMs: null,
    listeners: new Set(),
  };
  streamCache.set(url, entry);

  const t0 = performance.now();

  fetch(url)
    .then((res) => {
      if (!res.ok) {
        throw new Error(
          `Failed to fetch stream: ${res.status} ${res.statusText}`,
        );
      }
      if (!res.body) {
        throw new Error(
          "Response body is null — server did not send a stream.",
        );
      }
      return readNdJsonStream(res.body.getReader(), entry, t0, url);
    })
    .catch((err: unknown) => {
      entry.error = err instanceof Error ? err : new Error(String(err));
      entry.done = true;
      entry.listeners.forEach((fn) => fn());
    });

  return entry;
}

// ---------------------------------------------------------------------------
// getOrFetchData  (JSON path — unchanged from original)
//
// Ensures only one network request is ever in-flight per URL.
// ---------------------------------------------------------------------------

function getOrFetchData(url: string): Promise<ChartDataRow[]> {
  // 1. Already resolved — return synchronously.
  const cached = resolvedDataCache.get(url);
  if (cached) return Promise.resolve(cached);

  // 2. A fetch is already in-flight — re-use it.
  const existing = fetchPromiseCache.get(url);
  if (existing) return existing;

  // 3. Start a new fetch.
  const promise = fetch(url)
    .then((res) => {
      if (!res.ok) {
        throw new Error(
          `Failed to fetch data: ${res.status} ${res.statusText}`,
        );
      }
      return res.json() as Promise<unknown>;
    })
    .then((json) => {
      const parsed = ChartDataSchema.safeParse(json);
      if (!parsed.success) {
        throw new Error(parsed.error.message);
      }
      const data = transformRows(parsed.data);
      resolvedDataCache.set(url, data);
      return data;
    })
    .finally(() => {
      fetchPromiseCache.delete(url);
    });

  fetchPromiseCache.set(url, promise);
  return promise;
}

// ---------------------------------------------------------------------------
// prefetchData
//
// Call as early as possible (e.g. when the user switches data source) to warm
// the cache before any DynamicChart mounts.  Routes to the appropriate path
// based on whether the URL is a stream or regular JSON endpoint.
// ---------------------------------------------------------------------------

export function prefetchData(url: string): void {
  if (isStreamUrl(url)) {
    getOrStartStream(url);
  } else {
    void getOrFetchData(url);
  }
}

// ---------------------------------------------------------------------------
// clearDataCache
//
// Removes one URL (or all URLs) from all caches — JSON and stream alike.
// ---------------------------------------------------------------------------

export function clearDataCache(url?: string): void {
  if (url) {
    resolvedDataCache.delete(url);
    fetchPromiseCache.delete(url);
    streamCache.delete(url);
  } else {
    resolvedDataCache.clear();
    fetchPromiseCache.clear();
    streamCache.clear();
  }
}

// ---------------------------------------------------------------------------
// useStreamStatus
//
// Returns a live snapshot of a stream's progress for a given URL.
// Re-renders the consumer whenever the stream delivers new rows.
//
// Safe to call with any URL — non-stream URLs return a zeroed-out status.
// Safe to call with an empty string — returns zeroed-out status immediately.
// ---------------------------------------------------------------------------

export function useStreamStatus(url: string): StreamStatus {
  // useReducer(x => x+1) is the idiomatic "force re-render" pattern that
  // doesn't allocate new state objects on every notification.
  const [, rerender] = useReducer((x: number) => x + 1, 0);

  useEffect(() => {
    if (!url) return;
    const entry = streamCache.get(url);
    if (!entry || entry.done) return;

    entry.listeners.add(rerender);
    return () => {
      entry.listeners.delete(rerender);
    };
  }, [url]);

  if (!url) {
    return {
      rowCount: 0,
      streaming: false,
      firstRowMs: null,
      lastRowMs: null,
      error: null,
    };
  }

  const entry = streamCache.get(url);
  if (!entry) {
    return {
      rowCount: 0,
      streaming: false,
      firstRowMs: null,
      lastRowMs: null,
      error: null,
    };
  }

  return {
    rowCount: entry.rows.length,
    streaming: !entry.done,
    firstRowMs: entry.firstRowMs,
    lastRowMs: entry.lastRowMs,
    error: entry.error,
  };
}

// ---------------------------------------------------------------------------
// useChartData
//
// Resolves chart data from either:
//   - an inline ChartDataRow[]  (validated + transformed via Zod, memoised)
//   - a URL / relative path     (fetched via the shared cache above)
//
// For regular JSON URLs:
//   Exactly one network request is made regardless of how many charts share
//   the same URL.  Charts that mount after the resolve are synchronous.
//
// For NDJSON stream URLs:
//   All chart instances share one ReadableStream reader via the stream cache.
//   Each chunk delivers new rows to every subscribed hook instance via the
//   pub/sub listener set.  `streaming: true` is set while the stream is open
//   so DynamicChart can decide to render partial data instead of a spinner.
// ---------------------------------------------------------------------------

export function useChartData(
  source: ChartDataRow[] | string,
): UseChartDataResult {
  const isString = typeof source === "string";

  // ── Inline path ──────────────────────────────────────────────────────────

  const inlineResult = useMemo<UseChartDataResult | null>(() => {
    if (isString) return null;
    const parsed = ChartDataSchema.safeParse(source);
    if (parsed.success) {
      return {
        data: transformRows(parsed.data),
        loading: false,
        error: null,
        streaming: false,
      };
    }
    return {
      data: [],
      loading: false,
      error: new Error(parsed.error.message),
      streaming: false,
    };
  }, [isString, source]);

  // ── URL path — lazy initial state ────────────────────────────────────────
  // Initialise synchronously from whichever cache already has data so that
  // charts mounting after a prefetch or a completed stream never flash a
  // loading state.

  const [fetchResult, setFetchResult] = useState<UseChartDataResult>(() => {
    if (!isString)
      return { data: [], loading: false, error: null, streaming: false };

    if (isStreamUrl(source)) {
      const entry = streamCache.get(source);
      if (entry) {
        return {
          data: entry.rows,
          loading: !entry.done,
          error: entry.error,
          streaming: !entry.done,
        };
      }
      // Stream hasn't started yet — show loading.
      return { data: [], loading: true, error: null, streaming: true };
    }

    // JSON path: check resolved cache for a synchronous initialisation.
    const cached = resolvedDataCache.get(source);
    return cached
      ? { data: cached, loading: false, error: null, streaming: false }
      : { data: [], loading: true, error: null, streaming: false };
  });

  // ── URL path — effect ────────────────────────────────────────────────────

  useEffect(() => {
    if (!isString) return;

    let cancelled = false;

    // ── Stream path ───────────────────────────────────────────────────────
    if (isStreamUrl(source)) {
      const entry = getOrStartStream(source);

      const update = () => {
        if (cancelled) return;
        setFetchResult({
          data: entry.rows,
          loading: !entry.done,
          error: entry.error,
          streaming: !entry.done,
        });
      };

      // Apply current state immediately (covers the case where rows arrived
      // between the lazy useState init and this effect running).
      update();

      if (!entry.done) {
        entry.listeners.add(update);
        return () => {
          cancelled = true;
          entry.listeners.delete(update);
        };
      }

      return () => {
        cancelled = true;
      };
    }

    // ── JSON path ─────────────────────────────────────────────────────────

    // Fast path: data arrived in the resolved cache (e.g. from a sibling
    // chart's effect) between the lazy init and this effect running.
    const alreadyCached = resolvedDataCache.get(source);
    if (alreadyCached) {
      setFetchResult({
        data: alreadyCached,
        loading: false,
        error: null,
        streaming: false,
      });
      return () => {
        cancelled = true;
      };
    }

    setFetchResult({ data: [], loading: true, error: null, streaming: false });

    getOrFetchData(source)
      .then((data) => {
        if (!cancelled) {
          setFetchResult({
            data,
            loading: false,
            error: null,
            streaming: false,
          });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setFetchResult({
            data: [],
            loading: false,
            error: err instanceof Error ? err : new Error(String(err)),
            streaming: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isString, source]);

  return isString ? fetchResult : (inlineResult as UseChartDataResult);
}
