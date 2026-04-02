import React, { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { sankey as d3Sankey, sankeyLinkHorizontal } from "d3-sankey";
import type { SankeyNode, SankeyLink } from "d3-sankey";
import type { ChartConfig } from "../types/ChartConfig";
import type { ChartDataRow } from "../types/ChartData";
import { useSelection } from "../context/SelectionContext";
import {
  PALETTE,
  groupByField,
  applyAggregation,
  buildSankeyData,
} from "../utils/dataUtils";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface D3RendererProps {
  config: ChartConfig;
  data: ChartDataRow[];
  style?: React.CSSProperties;
  className?: string;
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type SVGSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;
type GSelection = d3.Selection<SVGGElement, unknown, null, undefined>;

interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

type SelectFn = (values: (string | number | null)[]) => void;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AXIS_COLOUR = "#666";
const GRID_COLOUR = "#e8e8e8";
const TITLE_COLOUR = "#1a1a2e";
const TEXT_FONT = "'Inter', system-ui, sans-serif";

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

function addTitle(
  svg: SVGSelection,
  title: string | undefined,
  w: number,
): void {
  if (!title) return;
  svg
    .append("text")
    .attr("x", w / 2)
    .attr("y", 18)
    .attr("text-anchor", "middle")
    .style("font-family", TEXT_FONT)
    .style("font-size", "12px")
    .style("font-weight", "600")
    .style("fill", TITLE_COLOUR)
    .text(title);
}

function addAxisLabel(
  g: GSelection,
  text: string,
  axis: "x" | "y",
  iw: number,
  ih: number,
): void {
  if (axis === "x") {
    g.append("text")
      .attr("x", iw / 2)
      .attr("y", ih + 42)
      .attr("text-anchor", "middle")
      .style("font-family", TEXT_FONT)
      .style("font-size", "10px")
      .style("fill", AXIS_COLOUR)
      .text(text);
  } else {
    g.append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -ih / 2)
      .attr("y", -46)
      .attr("text-anchor", "middle")
      .style("font-family", TEXT_FONT)
      .style("font-size", "10px")
      .style("fill", AXIS_COLOUR)
      .text(text);
  }
}

function addHorizontalGrid(
  g: GSelection,
  yScale: d3.ScaleLinear<number, number>,
  iw: number,
): void {
  const grid = g
    .append("g")
    .attr("class", "grid")
    .call(
      d3
        .axisLeft(yScale)
        .ticks(5)
        .tickSize(-iw)
        .tickFormat(() => ""),
    );
  grid.select(".domain").remove();
  grid
    .selectAll("line")
    .style("stroke", GRID_COLOUR)
    .style("stroke-dasharray", "3,3");
}

function styleAxis(
  sel: d3.Selection<SVGGElement, unknown, d3.BaseType, unknown>,
): void {
  sel.select(".domain").style("stroke", AXIS_COLOUR);
  sel.selectAll(".tick line").style("stroke", AXIS_COLOUR);
  sel
    .selectAll(".tick text")
    .style("font-family", TEXT_FONT)
    .style("font-size", "9px")
    .style("fill", AXIS_COLOUR);
}

function rotateXLabels(g: GSelection): void {
  g.selectAll(".x-axis .tick text")
    .attr("transform", "rotate(-35)")
    .attr("text-anchor", "end")
    .attr("dx", "-0.4em")
    .attr("dy", "0.6em");
}

// ---------------------------------------------------------------------------
// Per-chart renderers
// ---------------------------------------------------------------------------

// ── Line / Area ──────────────────────────────────────────────────────────────

function renderLineArea(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, colorBy, title, type } = config;
  if (!yAxis) return;
  const isArea = type === "area";

  const margin: Margin = { top: 36, right: 80, bottom: 58, left: 58 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const groups = colorBy ? groupByField(data, colorBy) : new Map([["", data]]);
  const xValues = [...new Set(data.map((d) => String(d[xAxis.field] ?? "")))];

  let yMax = 0;
  for (const row of data) {
    const v = Number(row[yAxis.field]);
    if (!isNaN(v) && v > yMax) yMax = v;
  }

  const xScale = d3
    .scalePoint<string>()
    .domain(xValues)
    .range([0, iw])
    .padding(0.1);

  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.12])
    .range([ih, 0])
    .nice();

  addHorizontalGrid(g, yScale, iw);

  // Axes
  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale));
  styleAxis(xAxis_);
  rotateXLabels(g);

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(5));
  styleAxis(yAxis_);

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  if (yAxis.title) addAxisLabel(g, yAxis.title, "y", iw, ih);

  let colIdx = 0;
  const legend: { name: string; colour: string }[] = [];

  for (const [name, rows] of groups) {
    const colour = PALETTE[colIdx++ % PALETTE.length] ?? "#636EFA";
    legend.push({ name, colour });

    if (isArea) {
      const areaGen = d3
        .area<ChartDataRow>()
        .x((d) => xScale(String(d[xAxis.field] ?? "")) ?? 0)
        .y0(ih)
        .y1((d) => yScale(Number(d[yAxis!.field]) || 0))
        .defined((d) => !isNaN(Number(d[yAxis!.field])));

      g.append("path")
        .datum(rows)
        .attr("d", areaGen)
        .attr("fill", colour)
        .attr("fill-opacity", 0.18)
        .attr("stroke", "none");
    }

    const lineGen = d3
      .line<ChartDataRow>()
      .x((d) => xScale(String(d[xAxis.field] ?? "")) ?? 0)
      .y((d) => yScale(Number(d[yAxis!.field]) || 0))
      .defined((d) => !isNaN(Number(d[yAxis!.field])));

    g.append("path")
      .datum(rows)
      .attr("d", lineGen)
      .attr("fill", "none")
      .attr("stroke", colour)
      .attr("stroke-width", 2);

    const dotG = g.append("g");
    dotG
      .selectAll<SVGCircleElement, ChartDataRow>("circle")
      .data(rows)
      .join("circle")
      .attr(
        "cx",
        (d: ChartDataRow) => xScale(String(d[xAxis.field] ?? "")) ?? 0,
      )
      .attr("cy", (d: ChartDataRow) => yScale(Number(d[yAxis!.field]) || 0))
      .attr("r", 4)
      .attr("fill", colour)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .on("click", (_: MouseEvent, d: ChartDataRow) =>
        onSelect([d[xAxis.field] ?? null]),
      );
  }

  // Inline legend
  if (colorBy && legend.length > 1) {
    const lx = iw + 8;
    legend.forEach(({ name, colour }, li) => {
      const ly = li * 18;
      g.append("rect")
        .attr("x", lx)
        .attr("y", ly)
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", colour);
      g.append("text")
        .attr("x", lx + 14)
        .attr("y", ly + 9)
        .style("font-size", "10px")
        .style("font-family", TEXT_FONT)
        .style("fill", "#444")
        .text(name.length > 12 ? name.slice(0, 11) + "…" : name);
    });
  }
}

// ── Bar ───────────────────────────────────────────────────────────────────────

function renderBar(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, colorBy, title } = config;
  if (!yAxis) return;

  const margin: Margin = { top: 36, right: 24, bottom: 64, left: 58 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const groups = colorBy ? groupByField(data, colorBy) : new Map([["", data]]);
  const xValues = [...new Set(data.map((d) => String(d[xAxis.field] ?? "")))];
  const yMax = d3.max(data, (d) => Number(d[yAxis.field])) ?? 0;

  const xScale = d3
    .scaleBand<string>()
    .domain(xValues)
    .range([0, iw])
    .padding(0.2);

  const yScale = d3
    .scaleLinear()
    .domain([0, yMax * 1.12])
    .range([ih, 0])
    .nice();

  addHorizontalGrid(g, yScale, iw);

  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale));
  styleAxis(xAxis_);
  rotateXLabels(g);

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(5));
  styleAxis(yAxis_);

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  if (yAxis.title) addAxisLabel(g, yAxis.title, "y", iw, ih);

  const numGroups = groups.size;
  const groupWidth = xScale.bandwidth() / numGroups;

  let gIdx = 0;
  for (const [, rows] of groups) {
    const colour = PALETTE[gIdx % PALETTE.length] ?? "#636EFA";
    const offset = gIdx * groupWidth;

    const barG = g.append("g");
    barG
      .selectAll<SVGRectElement, ChartDataRow>("rect")
      .data(rows)
      .join("rect")
      .attr(
        "x",
        (d: ChartDataRow) =>
          (xScale(String(d[xAxis.field] ?? "")) ?? 0) + offset,
      )
      .attr("y", (d: ChartDataRow) => yScale(Number(d[yAxis!.field]) || 0))
      .attr("width", groupWidth - 2)
      .attr(
        "height",
        (d: ChartDataRow) => ih - yScale(Number(d[yAxis!.field]) || 0),
      )
      .attr("fill", colour)
      .attr("rx", 2)
      .attr("cursor", "pointer")
      .on("click", (_: MouseEvent, d: ChartDataRow) =>
        onSelect([d[xAxis.field] ?? null]),
      );

    gIdx++;
  }
}

// ── Scatter ───────────────────────────────────────────────────────────────────

function renderScatter(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, colorBy, marker, title } = config;
  if (!yAxis) return;

  const margin: Margin = { top: 36, right: 80, bottom: 52, left: 58 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const xVals = data
    .map((d) => Number(d[xAxis.field]))
    .filter((v) => !isNaN(v));
  const yVals = data
    .map((d) => Number(d[yAxis.field]))
    .filter((v) => !isNaN(v));

  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(xVals) as [number, number])
    .range([0, iw])
    .nice();
  const yScale = d3
    .scaleLinear()
    .domain(d3.extent(yVals) as [number, number])
    .range([ih, 0])
    .nice();

  addHorizontalGrid(g, yScale, iw);

  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale).ticks(6));
  styleAxis(xAxis_);

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(5));
  styleAxis(yAxis_);

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  if (yAxis.title) addAxisLabel(g, yAxis.title, "y", iw, ih);

  const groups = colorBy ? groupByField(data, colorBy) : new Map([["", data]]);
  const legend: { name: string; colour: string }[] = [];
  let colIdx = 0;

  for (const [name, rows] of groups) {
    const colour = PALETTE[colIdx++ % PALETTE.length] ?? "#636EFA";
    legend.push({ name, colour });

    const scatterG = g.append("g");
    scatterG
      .selectAll<SVGCircleElement, ChartDataRow>("circle")
      .data(rows)
      .join("circle")
      .attr("cx", (d: ChartDataRow) => xScale(Number(d[xAxis.field])))
      .attr("cy", (d: ChartDataRow) => yScale(Number(d[yAxis!.field])))
      .attr("r", marker?.size ?? 6)
      .attr("fill", colour)
      .attr("fill-opacity", marker?.opacity ?? 0.7)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .attr("cursor", "pointer")
      .on("click", (_: MouseEvent, d: ChartDataRow) =>
        onSelect([d[xAxis.field] ?? null]),
      );
  }

  if (colorBy && legend.length > 1) {
    const lx = iw + 8;
    legend.forEach(({ name, colour }, li) => {
      g.append("circle")
        .attr("cx", lx + 5)
        .attr("cy", li * 18 + 5)
        .attr("r", 5)
        .attr("fill", colour);
      g.append("text")
        .attr("x", lx + 14)
        .attr("y", li * 18 + 9)
        .style("font-size", "10px")
        .style("font-family", TEXT_FONT)
        .style("fill", "#444")
        .text(name.length > 12 ? name.slice(0, 11) + "…" : name);
    });
  }
}

// ── Pie ──────────────────────────────────────────────────────────────────────

function renderPie(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, title } = config;
  if (!yAxis) return;

  const margin = 30;
  const cx = w / 2;
  const cy = h / 2 + 6;
  const outerR = Math.min(w, h) / 2 - margin - 20;
  const innerR = outerR * 0.4;

  addTitle(svg, title, w);

  const labels = data.map((d) => String(d[xAxis.field] ?? ""));
  const values = data.map((d) => Number(d[yAxis.field]) || 0);

  const pie = d3
    .pie<number>()
    .value((d) => d)
    .sort(null);
  const arcs = pie(values);

  const arcGen = d3
    .arc<d3.PieArcDatum<number>>()
    .innerRadius(innerR)
    .outerRadius(outerR);

  const arcHover = d3
    .arc<d3.PieArcDatum<number>>()
    .innerRadius(innerR)
    .outerRadius(outerR + 6);

  const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);

  arcs.forEach((arc: d3.PieArcDatum<number>, i: number) => {
    const colour = PALETTE[i % PALETTE.length] ?? "#636EFA";
    const slice = g
      .append("path")
      .datum(arc)
      .attr("d", arcGen)
      .attr("fill", colour)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("cursor", "pointer");

    slice
      .on("mouseenter", function (this: SVGPathElement) {
        d3.select<SVGPathElement, d3.PieArcDatum<number>>(this).attr(
          "d",
          arcHover,
        );
      })
      .on("mouseleave", function (this: SVGPathElement) {
        d3.select<SVGPathElement, d3.PieArcDatum<number>>(this).attr(
          "d",
          arcGen,
        );
      })
      .on("click", () => {
        const lbl = labels[i];
        if (lbl !== undefined) onSelect([lbl]);
      });

    // Label on larger slices
    const pct = (arc.endAngle - arc.startAngle) / (2 * Math.PI);
    if (pct > 0.06) {
      const [lx, ly] = arcGen.centroid(arc);
      g.append("text")
        .attr("x", lx)
        .attr("y", ly)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .style("font-size", "9px")
        .style("font-family", TEXT_FONT)
        .style("fill", "#fff")
        .style("pointer-events", "none")
        .text(`${(pct * 100).toFixed(0)}%`);
    }
  });

  // Legend at bottom
  const legendY = cy + outerR + 14;
  const itemW = 90;
  const perRow = Math.max(1, Math.floor(w / itemW));
  labels.forEach((lbl, i) => {
    const row = Math.floor(i / perRow);
    const col = i % perRow;
    const lx =
      w / 2 - (Math.min(labels.length, perRow) * itemW) / 2 + col * itemW;
    const ly = legendY + row * 16;
    svg
      .append("rect")
      .attr("x", lx)
      .attr("y", ly)
      .attr("width", 10)
      .attr("height", 10)
      .attr("fill", PALETTE[i % PALETTE.length] ?? "#636EFA");
    svg
      .append("text")
      .attr("x", lx + 13)
      .attr("y", ly + 9)
      .style("font-size", "10px")
      .style("font-family", TEXT_FONT)
      .style("fill", "#444")
      .text(lbl.length > 10 ? lbl.slice(0, 9) + "…" : lbl);
  });
}

// ── Histogram ────────────────────────────────────────────────────────────────

function renderHistogram(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, colorBy, title } = config;

  const margin: Margin = { top: 36, right: 80, bottom: 52, left: 58 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const allVals = data
    .map((d) => Number(d[xAxis.field]))
    .filter((v) => !isNaN(v));
  const xExtent = d3.extent(allVals) as [number, number];

  const xScale = d3.scaleLinear().domain(xExtent).range([0, iw]).nice();
  const binGen = d3
    .bin<ChartDataRow, number>()
    .value((d) => Number(d[xAxis.field]))
    .domain(xScale.domain() as [number, number])
    .thresholds(xScale.ticks(20));

  const groups = colorBy ? groupByField(data, colorBy) : new Map([["", data]]);
  const legend: { name: string; colour: string }[] = [];
  let maxCount = 0;

  const allBinSets: {
    bins: d3.Bin<ChartDataRow, number>[];
    colour: string;
    name: string;
  }[] = [];
  let colIdx = 0;
  for (const [name, rows] of groups) {
    const colour = PALETTE[colIdx++ % PALETTE.length] ?? "#636EFA";
    const bins = binGen(rows);
    bins.forEach((b) => {
      if (b.length > maxCount) maxCount = b.length;
    });
    allBinSets.push({ bins, colour, name });
    legend.push({ name, colour });
  }

  const yScale = d3
    .scaleLinear()
    .domain([0, maxCount * 1.12])
    .range([ih, 0])
    .nice();

  addHorizontalGrid(g, yScale, iw);

  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale).ticks(8));
  styleAxis(xAxis_);

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(5));
  styleAxis(yAxis_);

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  addAxisLabel(g, "Count", "y", iw, ih);

  allBinSets.forEach(({ bins, colour }) => {
    type BinDatum = d3.Bin<ChartDataRow, number>;
    const histG = g.append("g");
    histG
      .selectAll<SVGRectElement, BinDatum>("rect")
      .data(bins)
      .join("rect")
      .attr("x", (b: BinDatum) => xScale(b.x0 ?? 0) + 1)
      .attr("y", (b: BinDatum) => yScale(b.length))
      .attr("width", (b: BinDatum) =>
        Math.max(0, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 2),
      )
      .attr("height", (b: BinDatum) => ih - yScale(b.length))
      .attr("fill", colour)
      .attr("fill-opacity", 0.7)
      .attr("rx", 2)
      .attr("cursor", "pointer")
      .on("click", (_: MouseEvent, b: BinDatum) => {
        const mid = ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2;
        onSelect([mid]);
      });
  });

  if (colorBy && legend.length > 1) {
    const lx = iw + 8;
    legend.forEach(({ name, colour }, li) => {
      g.append("rect")
        .attr("x", lx)
        .attr("y", li * 18)
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", colour);
      g.append("text")
        .attr("x", lx + 14)
        .attr("y", li * 18 + 9)
        .style("font-size", "10px")
        .style("font-family", TEXT_FONT)
        .style("fill", "#444")
        .text(name.length > 12 ? name.slice(0, 11) + "…" : name);
    });
  }
}

// ── Box Plot ──────────────────────────────────────────────────────────────────

function renderBox(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, colorBy, title } = config;
  if (!yAxis) return;

  const margin: Margin = { top: 36, right: 24, bottom: 64, left: 58 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const splitField = colorBy ?? xAxis.field;
  const groups = groupByField(data, splitField);
  const groupNames = Array.from(groups.keys());

  interface BoxStats {
    name: string;
    q1: number;
    q2: number;
    q3: number;
    whiskerLow: number;
    whiskerHigh: number;
    outliers: number[];
  }

  const stats: BoxStats[] = [];
  const allVals: number[] = [];

  for (const [name, rows] of groups) {
    const vals = rows
      .map((r) => Number(r[yAxis.field]))
      .filter((v) => !isNaN(v))
      .sort((a, b) => a - b);
    if (vals.length === 0) continue;

    allVals.push(...vals);

    const q1 = d3.quantileSorted(vals, 0.25) ?? 0;
    const q2 = d3.quantileSorted(vals, 0.5) ?? 0;
    const q3 = d3.quantileSorted(vals, 0.75) ?? 0;
    const iqr = q3 - q1;
    const wLow = vals.find((v) => v >= q1 - 1.5 * iqr) ?? q1;
    const wHigh = [...vals].reverse().find((v) => v <= q3 + 1.5 * iqr) ?? q3;

    stats.push({
      name,
      q1,
      q2,
      q3,
      whiskerLow: wLow,
      whiskerHigh: wHigh,
      outliers: vals.filter((v) => v < wLow || v > wHigh),
    });
  }

  if (stats.length === 0) return;

  const yMin = d3.min(allVals) ?? 0;
  const yMax = d3.max(allVals) ?? 1;
  const yPad = (yMax - yMin) * 0.12;

  const xScale = d3
    .scaleBand<string>()
    .domain(groupNames)
    .range([0, iw])
    .padding(0.35);
  const yScale = d3
    .scaleLinear()
    .domain([yMin - yPad, yMax + yPad])
    .range([ih, 0])
    .nice();

  addHorizontalGrid(g, yScale, iw);

  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale));
  styleAxis(xAxis_);
  rotateXLabels(g);

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(5));
  styleAxis(yAxis_);

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  if (yAxis.title) addAxisLabel(g, yAxis.title, "y", iw, ih);

  stats.forEach((s, i) => {
    const colour = PALETTE[i % PALETTE.length] ?? "#636EFA";
    const bx = xScale(s.name) ?? 0;
    const bw = xScale.bandwidth();
    const cx = bx + bw / 2;

    // Whisker vertical lines
    const whiskerLine = (y1: number, y2: number) =>
      g
        .append("line")
        .attr("x1", cx)
        .attr("x2", cx)
        .attr("y1", yScale(y1))
        .attr("y2", yScale(y2))
        .attr("stroke", colour)
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "3,2");

    whiskerLine(s.q3, s.whiskerHigh);
    whiskerLine(s.q1, s.whiskerLow);

    // Whisker caps
    const cap = (yVal: number) =>
      g
        .append("line")
        .attr("x1", bx + bw * 0.2)
        .attr("x2", bx + bw * 0.8)
        .attr("y1", yScale(yVal))
        .attr("y2", yScale(yVal))
        .attr("stroke", colour)
        .attr("stroke-width", 1.5);

    cap(s.whiskerHigh);
    cap(s.whiskerLow);

    // IQR box
    g.append("rect")
      .attr("x", bx)
      .attr("y", yScale(s.q3))
      .attr("width", bw)
      .attr("height", Math.max(1, yScale(s.q1) - yScale(s.q3)))
      .attr("fill", colour)
      .attr("fill-opacity", 0.25)
      .attr("stroke", colour)
      .attr("stroke-width", 2)
      .attr("rx", 2)
      .attr("cursor", "pointer")
      .on("click", () => onSelect([s.name]));

    // Median line
    g.append("line")
      .attr("x1", bx)
      .attr("x2", bx + bw)
      .attr("y1", yScale(s.q2))
      .attr("y2", yScale(s.q2))
      .attr("stroke", colour)
      .attr("stroke-width", 2.5);

    // Outlier dots
    s.outliers.forEach((v) => {
      g.append("circle")
        .attr("cx", cx)
        .attr("cy", yScale(v))
        .attr("r", 3)
        .attr("fill", colour)
        .attr("fill-opacity", 0.6)
        .attr("stroke", colour)
        .attr("stroke-width", 0.5);
    });
  });
}

// ── Heatmap ──────────────────────────────────────────────────────────────────

function renderHeatmap(
  svg: SVGSelection,
  config: ChartConfig,
  data: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, zAxis, marker, title } = config;
  if (!yAxis || !zAxis) return;

  const margin: Margin = { top: 36, right: 24, bottom: 84, left: 72 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  addTitle(svg, title, w);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const xVals = [...new Set(data.map((d) => String(d[xAxis.field] ?? "")))];
  const yVals = [...new Set(data.map((d) => String(d[yAxis.field] ?? "")))];

  const valueMap = new Map<string, number>();
  for (const row of data) {
    const key = `${String(row[xAxis.field] ?? "")}|||${String(row[yAxis.field] ?? "")}`;
    valueMap.set(
      key,
      (valueMap.get(key) ?? 0) + (Number(row[zAxis.field]) || 0),
    );
  }

  const maxVal = d3.max(Array.from(valueMap.values())) ?? 1;

  const colorScale =
    marker?.colorScale === "Reds"
      ? d3.interpolateReds
      : marker?.colorScale === "Greens"
        ? d3.interpolateGreens
        : marker?.colorScale === "YlOrRd"
          ? d3.interpolateYlOrRd
          : marker?.colorScale === "RdBu"
            ? d3.interpolateRdBu
            : d3.interpolateBlues;

  const xScale = d3
    .scaleBand<string>()
    .domain(xVals)
    .range([0, iw])
    .padding(0.04);
  const yScale = d3
    .scaleBand<string>()
    .domain(yVals)
    .range([0, ih])
    .padding(0.04);

  const xAxis_ = g
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(xScale));
  styleAxis(xAxis_);
  xAxis_
    .selectAll(".tick text")
    .attr("transform", "rotate(-40)")
    .attr("text-anchor", "end")
    .attr("dx", "-0.4em")
    .attr("dy", "0.6em");

  const yAxis_ = g
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale));
  styleAxis(yAxis_);

  for (const xv of xVals) {
    for (const yv of yVals) {
      const val = valueMap.get(`${xv}|||${yv}`) ?? 0;
      const norm = maxVal > 0 ? val / maxVal : 0;

      g.append("rect")
        .attr("x", xScale(xv) ?? 0)
        .attr("y", yScale(yv) ?? 0)
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", colorScale(norm))
        .attr("rx", 2)
        .attr("cursor", "pointer")
        .on("click", () => onSelect([xv]));

      // Value label if cells are large enough
      if (xScale.bandwidth() > 32 && yScale.bandwidth() > 14 && val > 0) {
        g.append("text")
          .attr("x", (xScale(xv) ?? 0) + xScale.bandwidth() / 2)
          .attr("y", (yScale(yv) ?? 0) + yScale.bandwidth() / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .style("font-size", "9px")
          .style("font-family", TEXT_FONT)
          .style("fill", norm > 0.55 ? "#fff" : "#333")
          .style("pointer-events", "none")
          .text(val.toLocaleString());
      }
    }
  }

  if (xAxis.title) addAxisLabel(g, xAxis.title, "x", iw, ih);
  if (yAxis.title) addAxisLabel(g, yAxis.title, "y", iw, ih);
}

// ---------------------------------------------------------------------------
// Sankey diagram
// ---------------------------------------------------------------------------

function renderSankey(
  svg: SVGSelection,
  config: ChartConfig,
  rawData: ChartDataRow[],
  w: number,
  h: number,
  onSelect: SelectFn,
): void {
  const { xAxis, yAxis, sankeyTarget, aggregation, title } = config;
  if (!sankeyTarget || !yAxis) return;

  const margin = { top: title ? 48 : 24, right: 110, bottom: 24, left: 110 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;
  if (iw <= 0 || ih <= 0) return;

  if (title) addTitle(svg, title, w);

  const sankeyData = buildSankeyData(
    rawData,
    xAxis.field,
    sankeyTarget,
    yAxis.field,
    aggregation ?? "sum",
  );

  if (!sankeyData.nodes.length || !sankeyData.links.length) return;

  // ── Local types (keep module scope clean) ─────────────────────────────
  interface SNExtra {
    name: string;
  }
  interface SLExtra {
    value: number;
  }
  type SNType = SankeyNode<SNExtra, SLExtra>;
  type SLType = SankeyLink<SNExtra, SLExtra>;

  // ── Per-node colour ────────────────────────────────────────────────────
  const colour = (idx: number) => PALETTE[idx % PALETTE.length] ?? "#636EFA";

  // ── Compute layout ─────────────────────────────────────────────────────
  const layout = d3Sankey<
    { nodes: SNType[]; links: SLType[] },
    SNExtra,
    SLExtra
  >()
    .nodeWidth(18)
    .nodePadding(14)
    .extent([
      [margin.left, margin.top],
      [w - margin.right, h - margin.bottom],
    ]);

  const graph = layout({
    nodes: sankeyData.nodes.map((d) => ({ ...d })) as SNType[],
    links: sankeyData.links.map((l) => ({ ...l })) as SLType[],
  });

  // ── Links ──────────────────────────────────────────────────────────────
  const linkPath = sankeyLinkHorizontal<SNExtra, SLExtra>();

  const linkSel = svg
    .append("g")
    .attr("fill", "none")
    .selectAll<SVGPathElement, SLType>("path")
    .data(graph.links)
    .join("path")
    .attr("d", linkPath)
    .attr("stroke", (l) => {
      const src = l.source as SNType;
      const idx = graph.nodes.findIndex((n) => n.name === src.name);
      return colour(idx < 0 ? 0 : idx);
    })
    .attr("stroke-width", (l) => Math.max(1, l.width ?? 1))
    .attr("stroke-opacity", 0.38);

  linkSel
    .on("mouseover", function () {
      d3.select(this).attr("stroke-opacity", 0.65);
    })
    .on("mouseout", function () {
      d3.select(this).attr("stroke-opacity", 0.38);
    });

  linkSel.append("title").text((l) => {
    const src = (l.source as SNType).name;
    const tgt = (l.target as SNType).name;
    return `${src} → ${tgt}\n${l.value.toLocaleString()}`;
  });

  // ── Nodes ──────────────────────────────────────────────────────────────
  const nodeG = svg
    .append("g")
    .selectAll<SVGGElement, SNType>("g")
    .data(graph.nodes)
    .join("g")
    .style("cursor", "pointer")
    .on("click", (_event, d) => {
      onSelect([d.name]);
    });

  const rects = nodeG
    .append("rect")
    .attr("x", (d) => d.x0 ?? 0)
    .attr("y", (d) => d.y0 ?? 0)
    .attr("height", (d) => Math.max(1, (d.y1 ?? 0) - (d.y0 ?? 0)))
    .attr("width", (d) => (d.x1 ?? 0) - (d.x0 ?? 0))
    .attr("fill", (_d, i) => colour(i))
    .attr("rx", 3)
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  rects
    .append("title")
    .text((d) => `${d.name}\n${(d.value ?? 0).toLocaleString()}`);

  // ── Node labels ────────────────────────────────────────────────────────
  const midX = w / 2;

  nodeG
    .append("text")
    .attr("x", (d) => ((d.x0 ?? 0) < midX ? (d.x1 ?? 0) + 6 : (d.x0 ?? 0) - 6))
    .attr("y", (d) => ((d.y0 ?? 0) + (d.y1 ?? 0)) / 2)
    .attr("dy", "0.35em")
    .attr("text-anchor", (d) => ((d.x0 ?? 0) < midX ? "start" : "end"))
    .attr("font-size", 11)
    .attr("font-family", TEXT_FONT)
    .attr("fill", TITLE_COLOUR)
    .text((d) => {
      const val = d.value ?? 0;
      const fmtVal = val >= 1000 ? `${(val / 1000).toFixed(1)}k` : String(val);
      return `${d.name} (${fmtVal})`;
    });

  // ── Source / target axis labels ────────────────────────────────────────
  const labelY = h - 6;
  if (xAxis.title) {
    svg
      .append("text")
      .attr("x", margin.left)
      .attr("y", labelY)
      .attr("font-size", 10)
      .attr("font-family", TEXT_FONT)
      .attr("fill", AXIS_COLOUR)
      .text(xAxis.title);
  }
  if (yAxis.title) {
    svg
      .append("text")
      .attr("x", w - margin.right)
      .attr("y", labelY)
      .attr("font-size", 10)
      .attr("font-family", TEXT_FONT)
      .attr("text-anchor", "end")
      .attr("fill", AXIS_COLOUR)
      .text(yAxis.title);
  }
}

// ---------------------------------------------------------------------------
// D3Renderer component
// ---------------------------------------------------------------------------

export const D3Renderer: React.FC<D3RendererProps> = ({
  config,
  data,
  style,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { setSelectionByValues } = useSelection();

  const onSelect = useCallback(
    (values: (string | number | null)[]) => {
      setSelectionByValues(config.id, config.xAxis.field, values);
    },
    [config.id, config.xAxis.field, setSelectionByValues],
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Use actual rendered dimensions; fall back to sensible defaults
    const w = el.clientWidth || 460;
    const h = el.clientHeight || 280;

    // Clear previous render
    el.innerHTML = "";

    // Pre-aggregate if the config requires it
    const rows = applyAggregation(data, config);

    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", w)
      .attr("height", h)
      .style("overflow", "visible");

    switch (config.type) {
      case "line":
      case "area":
        renderLineArea(svg, config, rows, w, h, onSelect);
        break;
      case "bar":
        renderBar(svg, config, rows, w, h, onSelect);
        break;
      case "scatter":
        renderScatter(svg, config, rows, w, h, onSelect);
        break;
      case "pie":
        renderPie(svg, config, rows, w, h, onSelect);
        break;
      case "histogram":
        renderHistogram(svg, config, rows, w, h, onSelect);
        break;
      case "box":
        renderBox(svg, config, rows, w, h, onSelect);
        break;
      case "heatmap":
        renderHeatmap(svg, config, rows, w, h, onSelect);
        break;
      case "sankey":
        // Sankey performs its own 2-field aggregation — pass raw data
        renderSankey(svg, config, data, w, h, onSelect);
        break;
    }
  }, [config, data, onSelect]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden", ...style }}
      className={className}
    />
  );
};

D3Renderer.displayName = "D3Renderer";
