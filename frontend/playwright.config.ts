import { defineConfig, devices } from "@playwright/test";

// Runs against its own stack so nothing you use for development is touched: a dedicated
// database (outlier_e2e, or E2E_DATABASE_URL), the API on :8100, Next on :3100 (override with
// E2E_API_PORT / E2E_WEB_PORT when those are busy). `pnpm e2e`
// creates the tables, seeds them, and sets the test users' passwords first (e2e/global-setup.ts).
const API_PORT = Number(process.env.E2E_API_PORT ?? 8100);
const WEB_PORT = Number(process.env.E2E_WEB_PORT ?? 3100);
export const E2E_DATABASE_URL =
  process.env.E2E_DATABASE_URL ?? "postgresql+asyncpg://outlier:outlier@127.0.0.1:5433/outlier_e2e";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  globalSetup: "./e2e/global-setup.ts",
  use: { baseURL: `http://localhost:${WEB_PORT}`, trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  webServer: [
    {
      command: `cd .. && UV_PROJECT_ENVIRONMENT=venv uv run --no-sync oai serve --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      env: { DATABASE_URL: E2E_DATABASE_URL },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `pnpm dev --port ${WEB_PORT}`,
      url: `http://localhost:${WEB_PORT}/login`,
      env: { API_URL: `http://127.0.0.1:${API_PORT}`, NEXT_DIST_DIR: ".next-e2e" },
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
});
