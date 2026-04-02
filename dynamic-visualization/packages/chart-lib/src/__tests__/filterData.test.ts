import { describe, it, expect } from 'bun:test';
import type { ChartDataRow } from '../types/ChartData';

// ---------------------------------------------------------------------------
// filterData logic — extracted as a pure function so it can be unit-tested
// without mounting a React context.
// ---------------------------------------------------------------------------

function filterData(
  data: ChartDataRow[],
  field: string,
  selectedValues: (string | number | null)[],
): ChartDataRow[] {
  if (selectedValues.length === 0) return data;
  const valueSet = new Set(selectedValues.map(String));
  return data.filter((row) => valueSet.has(String(row[field])));
}

const ROWS: ChartDataRow[] = [
  { region: 'North', sales: 100 },
  { region: 'South', sales: 200 },
  { region: 'East',  sales: 150 },
  { region: 'West',  sales: 180 },
];

describe('filterData', () => {
  it('returns all rows when selectedValues is empty', () => {
    expect(filterData(ROWS, 'region', [])).toHaveLength(4);
  });

  it('returns matching rows only', () => {
    const result = filterData(ROWS, 'region', ['North', 'East']);
    expect(result).toHaveLength(2);
    expect(result.map((r) => r['region'])).toEqual(['North', 'East']);
  });

  it('returns empty array when no rows match', () => {
    expect(filterData(ROWS, 'region', ['Unknown'])).toHaveLength(0);
  });

  it('handles numeric field values via string coercion', () => {
    const result = filterData(ROWS, 'sales', [100, 150]);
    expect(result).toHaveLength(2);
  });

  it('null data values coerce to "null" and match null in selectedValues', () => {
    const rows: ChartDataRow[] = [{ region: null, sales: 50 }, { region: 'North', sales: 100 }];
    // String(null) === 'null' on both sides — should match
    const result = filterData(rows, 'region', [null]);
    expect(result).toHaveLength(1);
    expect(result[0]!['region']).toBeNull();
  });
});
