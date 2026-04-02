import { useEffect, useMemo, useState } from 'react';
import { ChartConfigSchema, type ChartConfig } from '../types/ChartConfig';

interface UseChartConfigResult {
  config: ChartConfig | null;
  loading: boolean;
  error: Error | null;
}

/**
 * Resolves a `ChartConfig` from either:
 * - an inline object (validated via Zod), or
 * - a URL / relative path string (fetched as JSON then validated).
 *
 * When the source is a URL the hook re-fetches whenever the URL changes.
 */
export function useChartConfig(source: ChartConfig | string): UseChartConfigResult {
  const isString = typeof source === 'string';

  // For inline objects we run a one-shot parse, no async work needed.
  const inlineResult = useMemo<UseChartConfigResult | null>(() => {
    if (isString) return null;
    const parsed = ChartConfigSchema.safeParse(source);
    if (parsed.success) {
      return { config: parsed.data, loading: false, error: null };
    }
    return { config: null, loading: false, error: new Error(parsed.error.message) };
  }, [isString, source]);

  const [fetchResult, setFetchResult] = useState<UseChartConfigResult>({
    config: null,
    loading: isString,
    error: null,
  });

  useEffect(() => {
    if (!isString) return;
    let cancelled = false;

    setFetchResult({ config: null, loading: true, error: null });

    fetch(source as string)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch config: ${res.status} ${res.statusText}`);
        return res.json() as Promise<unknown>;
      })
      .then((json) => {
        if (cancelled) return;
        const parsed = ChartConfigSchema.safeParse(json);
        if (parsed.success) {
          setFetchResult({ config: parsed.data, loading: false, error: null });
        } else {
          setFetchResult({
            config: null,
            loading: false,
            error: new Error(parsed.error.message),
          });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setFetchResult({
            config: null,
            loading: false,
            error: err instanceof Error ? err : new Error(String(err)),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isString, source]);

  return isString ? fetchResult : (inlineResult as UseChartConfigResult);
}
