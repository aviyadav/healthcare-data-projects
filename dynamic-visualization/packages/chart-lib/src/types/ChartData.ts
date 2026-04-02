import { z } from "zod";

/**
 * A single row of chart data — a flat record where each value is a primitive.
 * Null is permitted so that sparse datasets (e.g. missing measurements) are not
 * rejected at the validation boundary.
 */
export const ChartDataRowSchema = z.record(
  z.string(),
  z.union([z.string(), z.number(), z.null()]),
);

export type ChartDataRow = z.infer<typeof ChartDataRowSchema>;

/**
 * Full dataset: a bare array of rows, or an API wrapper `{ data: [...] }`.
 *
 * After parsing, every row is enriched with two computed fields:
 *   - `_count: 1`          — enables `aggregation: "count"` on any groupBy field
 *   - `_timestamp: number | null` — ms from epoch derived from `created_ts`
 */
export const ChartDataSchema = z
  .union([
    z.array(ChartDataRowSchema),
    z
      .object({ data: z.array(ChartDataRowSchema) })
      .transform(({ data }) => data),
  ])
  .transform((rows) =>
    rows.map((row) => {
      const rawTs = row["created_ts"];
      const msEpoch = rawTs ? new Date(String(rawTs)).getTime() : NaN;
      return {
        ...row,
        _count: 1,
        _timestamp: Number.isNaN(msEpoch) ? null : msEpoch,
      } as ChartDataRow;
    }),
  );

export type ChartData = ChartDataRow[];
