import { expect, test } from "@playwright/test";

test("a new expert signs up from the login page and lands on the Data screen", async ({ page }) => {
  await page.goto("/login");
  await page.getByTestId("signup-link").click();
  await page.waitForURL("**/signup");
  const email = `new-expert-${Date.now()}@example.com`;
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Display name").fill("New Expert");
  await page.getByLabel("Password (8+ characters)").fill("longenough1");
  await page.getByTestId("signup-role").selectOption("expert");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.waitForURL("**/data**");
  await expect(page.getByText("New Expert")).toBeVisible();

  // the same email cannot sign up twice
  await page.goto("/signup");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password (8+ characters)").fill("longenough1");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page.getByText(/already exists/)).toBeVisible();
});
