import { defineConfig, devices } from "@playwright/test";

// Runs against the local dev stack: Postgres from docker compose, the API on :8000, Next on :3000.
// `pnpm e2e` seeds the database and creates the test users first (see e2e/global-setup.ts).
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  globalSetup: "./e2e/global-setup.ts",
  use: { baseURL: "http://localhost:3000", trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  webServer: [
    {
      command: "cd .. && UV_PROJECT_ENVIRONMENT=venv uv run --no-sync oai serve --port 8000",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "pnpm dev --port 3000",
      url: "http://localhost:3000/login",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
});
