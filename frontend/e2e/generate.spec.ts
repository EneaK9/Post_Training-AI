import { expect, test, type Page } from "@playwright/test";

async function login(page: Page, role = "operator") {
  await page.goto("/login");
  await page.getByLabel("Email").fill(`${role}@example.com`);
  await page.getByLabel("Password").fill("e2e-password-1");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL("**/data**");
}

test("generate into a new episode, approve, simulate, and see the strip update", async ({ page }) => {
  await login(page);
  await page.goto("/generate");
  const select = page.getByTestId("brief-select");
  await select.selectOption({ index: 1 });
  await page.getByTestId("k-input").fill("4");
  await page.getByTestId("episode-select").selectOption("new");
  await page.getByTestId("cap-input").fill("3000");
  await page.getByTestId("generate-button").click();
  const ideas = page.getByTestId("idea-card");
  await expect(ideas.first()).toBeVisible({ timeout: 60_000 });
  expect(await ideas.count()).toBeGreaterThan(0);
  await expect(page.getByTestId("what-model-saw")).toBeVisible();
  await expect(page.getByText("why this combination").first()).toBeVisible();

  const strip = page.getByTestId("episode-strip");
  await expect(strip).toContainText("searching");
  await ideas.first().getByTestId("action-run").click();
  await expect(ideas.first()).toContainText("run");
  await strip.getByTestId("ship-approved").click();
  await expect(strip).toContainText(/[1-9]\d* live ads/, { timeout: 30_000 });
  await expect(strip.getByTestId("simulate-7")).toBeEnabled();
  await strip.getByTestId("simulate-7").click();
  await expect(strip.getByTestId("episode-spent")).not.toContainText("$0 /", { timeout: 60_000 });
});
