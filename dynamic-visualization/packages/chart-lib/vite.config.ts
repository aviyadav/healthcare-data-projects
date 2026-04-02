import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import dts from "vite-plugin-dts";
import { resolve } from "path";

export default defineConfig({
  plugins: [
    react(),
    dts({ outDir: "dist/types", tsconfigPath: "./tsconfig.json" }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "ChartLib",
      formats: ["es", "cjs"],
      fileName: (format) => `index.${format === "es" ? "mjs" : "cjs"}`,
    },
    rollupOptions: {
      external: [
        "react",
        "react-dom",
        "react/jsx-runtime",
        "plotly.js",
        "react-plotly.js",
        "chart.js",
        "react-chartjs-2",
        "d3",
        /^d3-.*/,
      ],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          "react/jsx-runtime": "ReactJsxRuntime",
          "plotly.js": "Plotly",
          "react-plotly.js": "ReactPlotly",
          "chart.js": "ChartJS",
          "react-chartjs-2": "ReactChartJS2",
          d3: "d3",
        },
      },
    },
  },
});
