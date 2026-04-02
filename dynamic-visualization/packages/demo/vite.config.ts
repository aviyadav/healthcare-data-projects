import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Point 'chart-lib' directly to its TypeScript source so the demo
      // always picks up the latest changes without requiring a lib rebuild.
      "chart-lib": resolve(__dirname, "../chart-lib/src/index.ts"),
    },
  },
  server: {
    proxy: {
      // Clinical Data API — port 8090
      "/api": {
        target: "http://127.0.0.1:8090",
        changeOrigin: true,
      },
    },
  },
});
