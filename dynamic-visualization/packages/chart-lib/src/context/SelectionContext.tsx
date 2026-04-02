import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { PlotSelectionEvent, PlotMouseEvent } from "plotly.js";
import type { ChartConfig } from "../types/ChartConfig";
import type { ChartDataRow } from "../types/ChartData";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SelectionState {
  /** ID of the chart that originated the selection */
  sourceChartId: string;
  /** Plotly point indices within the source trace (empty for non-Plotly renderers) */
  selectedIndices: number[];
  /** Distinct raw values of the xAxis field that are selected */
  selectedValues: (string | number | null)[];
  /** The data field name to cross-filter on */
  field: string;
}

interface SelectionContextValue {
  selection: SelectionState | null;

  /**
   * Plotly-specific: call when a plotly_selected or plotly_click event fires.
   * Extracts x-values and point indices from the Plotly event automatically.
   */
  setSelectionFromEvent: (
    event: PlotSelectionEvent | PlotMouseEvent,
    config: ChartConfig,
  ) => void;

  /**
   * Renderer-agnostic: call directly with the x-values you want to filter on.
   * Used by Chart.js, D3, and any future renderer that doesn't use Plotly events.
   *
   * @param sourceChartId  The `id` from the chart's ChartConfig
   * @param field          The xAxis field name to filter on across other charts
   * @param values         The selected x-values (pass empty array to clear)
   */
  setSelectionByValues: (
    sourceChartId: string,
    field: string,
    values: (string | number | null)[],
  ) => void;

  /** Clear the active selection — all charts return to showing full data. */
  clearSelection: () => void;

  /**
   * Returns a filtered subset of `data` where `row[field]` matches any value
   * in the current selection. When no selection is active, returns `data` unchanged.
   */
  filterData: (data: ChartDataRow[], field: string) => ChartDataRow[];
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const SelectionContext = createContext<SelectionContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export const SelectionProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selection, setSelection] = useState<SelectionState | null>(null);

  // ── Plotly path ────────────────────────────────────────────────────────────

  const setSelectionFromEvent = useCallback(
    (event: PlotSelectionEvent | PlotMouseEvent, config: ChartConfig) => {
      if (!event.points || event.points.length === 0) {
        setSelection(null);
        return;
      }

      const xField = config.xAxis.field;
      const indices = event.points.map(
        (p) => p.pointIndex ?? p.pointNumber ?? 0,
      );
      const values = Array.from(
        new Set(event.points.map((p) => p.x as string | number | null)),
      );

      setSelection({
        sourceChartId: config.id,
        selectedIndices: indices,
        selectedValues: values,
        field: xField,
      });
    },
    [],
  );

  // ── Renderer-agnostic path ─────────────────────────────────────────────────

  const setSelectionByValues = useCallback(
    (
      sourceChartId: string,
      field: string,
      values: (string | number | null)[],
    ) => {
      if (values.length === 0) {
        setSelection(null);
        return;
      }
      setSelection({
        sourceChartId,
        selectedIndices: [], // not applicable outside Plotly
        selectedValues: Array.from(new Set(values)),
        field,
      });
    },
    [],
  );

  // ── Clear ──────────────────────────────────────────────────────────────────

  const clearSelection = useCallback(() => setSelection(null), []);

  // ── Filter ─────────────────────────────────────────────────────────────────

  const filterData = useCallback(
    (data: ChartDataRow[], field: string): ChartDataRow[] => {
      if (!selection) return data;
      const valueSet = new Set(selection.selectedValues.map(String));
      return data.filter((row) => valueSet.has(String(row[field])));
    },
    [selection],
  );

  // ── Context value ──────────────────────────────────────────────────────────

  const value = useMemo<SelectionContextValue>(
    () => ({
      selection,
      setSelectionFromEvent,
      setSelectionByValues,
      clearSelection,
      filterData,
    }),
    [
      selection,
      setSelectionFromEvent,
      setSelectionByValues,
      clearSelection,
      filterData,
    ],
  );

  return (
    <SelectionContext.Provider value={value}>
      {children}
    </SelectionContext.Provider>
  );
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSelection(): SelectionContextValue {
  const ctx = useContext(SelectionContext);
  if (!ctx) {
    throw new Error("useSelection must be used inside a <ChartProvider>");
  }
  return ctx;
}
