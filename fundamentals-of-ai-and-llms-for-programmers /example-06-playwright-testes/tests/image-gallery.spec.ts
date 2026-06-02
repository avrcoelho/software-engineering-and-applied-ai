import { test, expect } from '@playwright/test';

test.describe('Image Gallery App', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the base URL configured in playwright.config.ts
    await page.goto('/vanilla-js-web-app-example/');
  });

  test('should display the correct page title', async ({ page }) => {
    await expect(page).toHaveTitle('TDD Frontend Example');
  });

  test('should submit the form successfully and update the image list', async ({ page }) => {
    const titleInput = page.getByRole('textbox', { name: 'Image Title' });
    const urlInput = page.getByRole('textbox', { name: 'Image URL' });
    const submitBtn = page.getByRole('button', { name: 'Submit Form' });

    // Fill in the form fields with test data
    const testTitle = 'Beautiful Mountain';
    const testUrl = 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b';

    await titleInput.fill(testTitle);
    await urlInput.fill(testUrl);

    // Submit the form
    await submitBtn.click();

    // Verify the card list now contains the new image card
    const cardList = page.locator('#card-list');
    const newCard = cardList.locator('article').last();

    // Verify the card's title is correct
    await expect(newCard.locator('.card-title')).toHaveText(testTitle);

    // Verify the card's image src and alt attributes
    const img = newCard.locator('img');
    await expect(img).toHaveAttribute('src', testUrl);
    await expect(img).toHaveAttribute('alt', `Image of an ${testTitle}`);
  });

  test.describe('Form Validation', () => {
    test('should show error messages when trying to submit empty fields', async ({ page }) => {
      const submitBtn = page.getByRole('button', { name: 'Submit Form' });

      // Feedback locators
      const titleFeedback = page.locator('#titleFeedback');
      const urlFeedback = page.locator('#urlFeedback');

      // Before submitting, validation errors should not be visible
      await expect(titleFeedback).not.toBeVisible();
      await expect(urlFeedback).not.toBeVisible();

      // Attempt submission without filling the form
      await submitBtn.click();

      // Both validation errors should be visible now
      await expect(titleFeedback).toBeVisible();
      await expect(titleFeedback).toHaveText('Please type a title for the image.');
      
      await expect(urlFeedback).toBeVisible();
      await expect(urlFeedback).toHaveText('Please type a valid URL');
    });

    test('should show error when title is filled but URL is empty', async ({ page }) => {
      const titleInput = page.getByRole('textbox', { name: 'Image Title' });
      const submitBtn = page.getByRole('button', { name: 'Submit Form' });
      
      const titleFeedback = page.locator('#titleFeedback');
      const urlFeedback = page.locator('#urlFeedback');

      await titleInput.fill('Valid Title');
      await submitBtn.click();

      // Title feedback should be hidden, URL feedback should be visible
      await expect(titleFeedback).not.toBeVisible();
      await expect(urlFeedback).toBeVisible();
    });

    test('should show error when URL is filled but title is empty', async ({ page }) => {
      const urlInput = page.getByRole('textbox', { name: 'Image URL' });
      const submitBtn = page.getByRole('button', { name: 'Submit Form' });
      
      const titleFeedback = page.locator('#titleFeedback');
      const urlFeedback = page.locator('#urlFeedback');

      await urlInput.fill('https://example.com/image.png');
      await submitBtn.click();

      // Title feedback should be visible, URL feedback should be hidden
      await expect(titleFeedback).toBeVisible();
      await expect(urlFeedback).not.toBeVisible();
    });

    test('should show error when URL format is invalid', async ({ page }) => {
      const titleInput = page.getByRole('textbox', { name: 'Image Title' });
      const urlInput = page.getByRole('textbox', { name: 'Image URL' });
      const submitBtn = page.getByRole('button', { name: 'Submit Form' });
      
      const titleFeedback = page.locator('#titleFeedback');
      const urlFeedback = page.locator('#urlFeedback');

      await titleInput.fill('Valid Title');
      await urlInput.fill('invalid-url-format');
      await submitBtn.click();

      // Title feedback should be hidden, URL feedback should be visible due to invalid format
      await expect(titleFeedback).not.toBeVisible();
      await expect(urlFeedback).toBeVisible();
    });
  });
});
