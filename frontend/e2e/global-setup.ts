import { execSync } from "node:child_process";
import path from "node:path";
import { E2E_DATABASE_URL } from "../playwright.config";

// Against the dedicated e2e database only: create tables, seed a small synthetic dataset, derive
// outcomes, and give the seeded users passwords. The development database is never touched.
export default function globalSetup() {
  const root = path.resolve(__dirname, "..", "..");
  const env = { ...process.env, UV_PROJECT_ENVIRONMENT: "venv", DATABASE_URL: E2E_DATABASE_URL };
  const run = (cmd: string) => execSync(`uv run --no-sync ${cmd}`, { cwd: root, env, stdio: "inherit" });
  run("oai db create-all");
  run("oai synth seed --briefs 4 --trajectories 60 --seed 21 --days-back 90");
  run("oai outlier recompute");
  for (const role of ["operator", "researcher", "expert"]) {
    run(`oai users set-password --email ${role}@example.com --password e2e-password-1`);
  }
}
