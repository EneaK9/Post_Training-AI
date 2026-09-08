import { expect, test, type Page } from "@playwright/test";

async function login(page: Page, role: "expert" | "operator" | "researcher" = "expert") {
  await page.goto("/login");
  await page.getByLabel("Email").fill(`${role}@example.com`);
  await page.getByLabel("Password").fill("e2e-password-1");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL("**/data**");
}

test.describe("Data screen", () => {
  test("expert authors a card, edits it into a new version, and activates it", async ({ page }) => {
    await login(page);
    const playbook = page.getByTestId("playbook-column");
    await expect(playbook).toContainText("Playbook");
    await expect(playbook.getByTestId("card-objection-flip")).toBeVisible();

    const slug = `ogilvy-long-copy-${Date.now().toString(36)}`;
    await page.getByTestId("add-card").click();
    await page.getByTestId("card-slug").fill(slug);
    await page.getByTestId("card-name").fill("Ogilvy Long Copy");
    await page.getByTestId("card-definition").fill("Long copy sells when the reader is already interested.");
    await page.getByTestId("save-card").click();
    const card = playbook.getByTestId(`card-${slug}`);
    await expect(card).toBeVisible();
    await expect(card).toContainText("draft");
    await expect(card).toContainText("v1");

    await card.getByTestId(`edit-${slug}`).click();
    await page.getByTestId("card-definition").fill("Long copy sells when the reader is already interested. Ogilvy proved it.");
    await page.getByTestId("save-card").click();
    await expect(card).toContainText("v2");

    await card.getByTestId(`activate-${slug}`).click();
    await expect(card).toContainText("active");
    await expect(card).toContainText("v3");

    // versions and diff in the drawer
    await card.getByRole("link", { name: "Ogilvy Long Copy" }).click();
    const drawer = page.getByTestId("drawer");
    await expect(drawer).toContainText("Versions");
    await expect(drawer).toContainText("v3");
  });

  test("history shows trajectories with chips, and a signal can be confirmed", async ({ page }) => {
    await login(page);
    const history = page.getByTestId("history-column");
    await expect(history.getByTestId("trajectory-card").first()).toBeVisible();
    await expect(history.getByText("written").first()).toBeVisible();

    await page.getByTestId("tab-signals").click();
    const queue = page.getByTestId("signals-queue");
    const proposed = queue.getByTestId("signal-proposed");
    const before = await proposed.count();
    expect(before).toBeGreaterThan(0);
    await proposed.first().getByLabel("confirm signal").click();
    await expect.poll(() => proposed.count()).toBeLessThan(before);
  });

  test("combinations tab lists usage and can name a combination as a card", async ({ page }) => {
    await login(page);
    await page.getByTestId("tab-combinations").click();
    const rows = page.getByTestId("combination-row");
    await expect(rows.first()).toBeVisible();
    await page.getByTestId("name-as-card").first().click();
    const slug = `named-${Date.now().toString(36)}`;
    await page.getByTestId("name-slug").fill(slug);
    await page.getByTestId("name-definition").fill("A pairing that keeps working on this account.");
    await page.getByTestId("name-save").click();
    await expect(rows.first()).toContainText("named");
    await page.getByTestId("tab-board").click();
    await expect(page.getByTestId("playbook-column").getByTestId(`card-${slug}`)).toBeVisible();
  });

  test("operator can label a trajectory wrong cards from the history column", async ({ page }) => {
    await login(page, "operator");
    const first = page.getByTestId("history-column").getByTestId("trajectory-card").first();
    await first.getByTestId("action-wrong-cards").click();
    await page.getByRole("checkbox").first().check();
    await page.getByTestId("confirm-wrong-cards").click();
    await expect(first).toContainText("wrong cards");
    await page.getByTestId("tab-audit").click();
    await expect(page.getByTestId("audit-tab")).toContainText("trajectory.review");
  });

  test("operator cannot create cards", async ({ page }) => {
    await login(page, "operator");
    await page.getByTestId("add-card").click();
    await page.getByTestId("card-slug").fill("forbidden-card");
    await page.getByTestId("card-name").fill("Forbidden");
    await page.getByTestId("card-definition").fill("Should be rejected.");
    await page.getByTestId("save-card").click();
    await expect(page.getByText(/may not cards:write/)).toBeVisible();
  });
});
