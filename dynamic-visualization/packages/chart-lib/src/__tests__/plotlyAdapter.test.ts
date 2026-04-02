import { describe, it, expect } from 'bun:test';
import { buildTraces, buildLayout } from '../adapters/plotlyAdapter';
import type { ChartConfig } from '../types/ChartConfig';
import type { ChartDataRow } from '../types/ChartData';

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const BASE_ROWS: ChartDataRow[] = [
  { date: '2025-01', region: 'North', sales: 15200, volume: 152, price: 100, category: 'Electronics' },
  { date: '2025-01', region: 'South', sales: 9800,  volume: 196, price: 50,  category: 'Clothing'    },
  { date: '2025-02', region: 'North', sales: 16800, volume: 336, price: 50,  category: 'Clothing'    },
  { date: '2025-02', region: 'South', sales: 11200, volume: 112, price: 100, category: 'Electronics' },
];

function baseConfig(overrides: Partial<ChartConfig>): ChartConfig {
  return {
    id: 'test',
    type: 'line',
    xAxis: { field: 'date' },
    yAxis: { field: 'sales' },
    enableSelection: true,
    selectionMode: 'box',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// buildTraces — one test per chart type
// ---------------------------------------------------------------------------

describe('buildTraces', () => {
  it('line → scatter trace with mode=lines', () => {
    const traces = buildTraces(baseConfig({ type: 'line' }), BASE_ROWS);
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ type: 'scatter', mode: 'lines' });
  });

  it('scatter → scatter trace with mode=markers', () => {
    const traces = buildTraces(baseConfig({ type: 'scatter' }), BASE_ROWS);
    expect(traces[0]).toMatchObject({ type: 'scatter', mode: 'markers' });
  });

  it('bar → bar trace', () => {
    const traces = buildTraces(baseConfig({ type: 'bar' }), BASE_ROWS);
    expect(traces[0]).toMatchObject({ type: 'bar' });
  });

  it('pie → pie trace with labels and values', () => {
    const traces = buildTraces(
      baseConfig({ type: 'pie', xAxis: { field: 'category' }, yAxis: { field: 'sales' } }),
      BASE_ROWS,
    );
    expect(traces[0]).toMatchObject({ type: 'pie' });
    const pie = traces[0] as Record<string, unknown>;
    expect(Array.isArray(pie['labels'])).toBe(true);
    expect(Array.isArray(pie['values'])).toBe(true);
  });

  it('area → scatter trace with fill=tozeroy', () => {
    const traces = buildTraces(baseConfig({ type: 'area' }), BASE_ROWS);
    const t = traces[0] as Record<string, unknown>;
    expect(t['type']).toBe('scatter');
    expect(t['fill']).toBe('tozeroy');
  });

  it('area with colorBy → multiple traces each with fill=tonexty', () => {
    const traces = buildTraces(
      baseConfig({ type: 'area', colorBy: 'region' }),
      BASE_ROWS,
    );
    expect(traces.length).toBeGreaterThan(1);
    traces.forEach((t) => {
      expect((t as Record<string, unknown>)['fill']).toBe('tonexty');
    });
  });

  it('heatmap → heatmap trace with z matrix', () => {
    const rows: ChartDataRow[] = [
      { date: '2025-01', region: 'North', sales: 100 },
      { date: '2025-01', region: 'South', sales: 200 },
      { date: '2025-02', region: 'North', sales: 150 },
      { date: '2025-02', region: 'South', sales: 250 },
    ];
    const traces = buildTraces(
      baseConfig({
        type: 'heatmap',
        xAxis: { field: 'date' },
        yAxis: { field: 'region' },
        zAxis: { field: 'sales' },
      }),
      rows,
    );
    expect(traces[0]).toMatchObject({ type: 'heatmap' });
    const t = traces[0] as Record<string, unknown>;
    expect(Array.isArray(t['z'])).toBe(true);
  });

  it('histogram → histogram trace', () => {
    const traces = buildTraces(
      baseConfig({ type: 'histogram', xAxis: { field: 'price' } }),
      BASE_ROWS,
    );
    expect(traces[0]).toMatchObject({ type: 'histogram' });
  });

  it('box → box trace with boxmean', () => {
    const traces = buildTraces(baseConfig({ type: 'box' }), BASE_ROWS);
    expect(traces[0]).toMatchObject({ type: 'box', boxmean: true });
  });

  it('colorBy splits data into separate traces', () => {
    const traces = buildTraces(
      baseConfig({ type: 'bar', colorBy: 'region' }),
      BASE_ROWS,
    );
    expect(traces.length).toBe(2); // North + South
    expect((traces[0] as Record<string, unknown>)['name']).toBeDefined();
  });

  it('aggregation=sum reduces rows correctly', () => {
    const rows: ChartDataRow[] = [
      { region: 'North', sales: 100 },
      { region: 'North', sales: 200 },
      { region: 'South', sales: 50  },
    ];
    const traces = buildTraces(
      baseConfig({
        type: 'bar',
        xAxis: { field: 'region' },
        yAxis: { field: 'sales' },
        aggregation: 'sum',
        groupBy: 'region',
      }),
      rows,
    );
    const t = traces[0] as Record<string, unknown>;
    const yVals = t['y'] as number[];
    const northIdx = (t['x'] as string[]).indexOf('North');
    expect(yVals[northIdx]).toBe(300);
  });
});

// ---------------------------------------------------------------------------
// buildLayout
// ---------------------------------------------------------------------------

describe('buildLayout', () => {
  it('sets title text when provided', () => {
    const layout = buildLayout(baseConfig({ title: 'My Chart' }));
    expect((layout.title as Record<string, unknown>)?.['text']).toBe('My Chart');
  });

  it('sets dragmode=select when selectionMode=box', () => {
    const layout = buildLayout(baseConfig({ selectionMode: 'box', enableSelection: true }));
    expect(layout.dragmode).toBe('select');
  });

  it('sets dragmode=lasso when selectionMode=lasso', () => {
    const layout = buildLayout(baseConfig({ selectionMode: 'lasso', enableSelection: true }));
    expect(layout.dragmode).toBe('lasso');
  });

  it('sets dragmode=zoom when enableSelection=false', () => {
    const layout = buildLayout(baseConfig({ enableSelection: false }));
    expect(layout.dragmode).toBe('zoom');
  });

  it('clickmode is always event+select', () => {
    const layout = buildLayout(baseConfig({}));
    expect(layout.clickmode).toBe('event+select');
  });

  it('merges layout overrides with highest priority', () => {
    const layout = buildLayout(
      baseConfig({ layout: { hovermode: 'x unified' } }),
    );
    expect(layout.hovermode).toBe('x unified');
  });

  it('xaxis title uses AxisConfig.title', () => {
    const layout = buildLayout(
      baseConfig({ xAxis: { field: 'date', title: 'Month' } }),
    );
    expect((layout.xaxis?.title as Record<string, unknown> | undefined)?.['text']).toBe('Month');
  });

  it('sets barmode=group for bar charts with groupBy', () => {
    const layout = buildLayout(
      baseConfig({ type: 'bar', groupBy: 'region' }),
    );
    expect(layout.barmode).toBe('group');
  });
});
