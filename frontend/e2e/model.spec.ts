import { expect, test, type Page } from "@playwright/test";

async function login(page: Page, role = "researcher") {
  await page.goto("/login");
  await page.getByLabel("Email").fill(`${role}@example.com`);
  await page.getByLabel("Password").fill("e2e-password-1");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL("**/data**");
}

test("model screen shows the architecture, stats, and applies a config change", async ({ page }) => {
  await login(page);
  await page.goto("/model");
  await expect(page.getByTestId("architecture")).toContainText("Loop A: search");
  await expect(page.getByTestId("loop-a-stats")).toContainText("per backend");
  await expect(page.getByTestId("accounts-panel")).toContainText("act_fake_1");

  const before = await page.getByTestId("config-hash").textContent();
  const yaml = page.getByTestId("config-yaml");
  await expect(yaml).toHaveValue(/tier2: 3\.0/);
  const text = await yaml.inputValue();
  expect(text).toContain("tier2: 3.0");
  await yaml.fill(text.replace("tier2: 3.0", "tier2: 3.5"));
  await page.getByTestId("config-validate").click();
  await expect(page.getByTestId("config-diff")).toContainText("outlier.tier_multiples.tier2");
  await expect(page.getByTestId("config-diff")).toContainText("recomputes every tier");
  await page.getByTestId("config-apply").click();
  await expect(page.getByTestId("config-hash")).not.toHaveText(before ?? "", { timeout: 60_000 });
  await expect(page.getByTestId("config-yaml")).toHaveValue(/tier2: 3\.5/);
});

test("operator cannot apply config", async ({ page }) => {
  await login(page, "operator");
  await page.goto("/model");
  const yaml = page.getByTestId("config-yaml");
  await expect(yaml).toHaveValue(/n_elite: 8/);
  const text = await yaml.inputValue();
  await yaml.fill(text.replace("n_elite: 8", "n_elite: 9"));
  await page.getByTestId("config-validate").click();
  await page.getByTestId("config-apply").click();
  await expect(page.getByText(/may not config:write/)).toBeVisible();
});

test("researcher launches a dry training run and a fake-arm eval from the Model screen", async ({ page }) => {
  await login(page);
  await page.goto("/model");
  await expect(page.getByTestId("rm-panel")).toContainText("tier 2+ positives");
  await expect(page.getByTestId("gold-gap")).toBeVisible();

  // Loop B: a dry run snapshots the archive and queues the trainer job
  await page.getByTestId("train-stage").selectOption("rft");
  await page.getByTestId("train-mode").selectOption("dry");
  await page.getByTestId("launch-training").click();
  await expect(page.getByTestId("training-runs-table")).toContainText("rft", { timeout: 30_000 });
  await expect(page.getByTestId("training-runs-table")).toContainText("queued");

  // online eval on the simulator arms: blind labels, per-arm rates, intervals
  await page.getByTestId("eval-briefs").fill("1");
  await page.getByTestId("launch-eval").click();
  const run = page.getByTestId("eval-run").first();
  await expect(run).toContainText("completed", { timeout: 120_000 });
  await expect(run).toContainText("1 briefs");
  await expect(run).toContainText("loop_a_fake");
  await expect(run).toContainText("random_fake");

  // reward model: seeded labels are too one-sided for the activation guards, so the panel says why
  await page.getByTestId("train-rm").click();
  await expect(page.getByTestId("train-rm-result")).toContainText(/trained rm_|cold reward model stays in use/, { timeout: 60_000 });
});

test("operator sees no launch controls", async ({ page }) => {
  await login(page, "operator");
  await page.goto("/model");
  await expect(page.getByTestId("loop-b-runs")).toBeVisible();
  await expect(page.getByTestId("launch-training")).toHaveCount(0);
  await expect(page.getByTestId("launch-eval")).toHaveCount(0);
  await expect(page.getByTestId("train-rm")).toHaveCount(0);
});
