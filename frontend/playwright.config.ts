import { defineConfig } from '@playwright/test'
import path from 'path'

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:5173'

export default defineConfig({
  testDir: path.join(__dirname, 'tests'),
  timeout: 180_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL,
    headless: true,
    trace: 'retain-on-failure'
  }
})
