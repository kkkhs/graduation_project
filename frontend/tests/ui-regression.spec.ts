import { test, expect } from '@playwright/test'
import path from 'path'

const modelKeys = ['drenet', 'yolo', 'mmdet_fcos'] as const

const imagePath = path.resolve(
  __dirname,
  '..',
  '..',
  'experiment_assets',
  'qualitative',
  'fcos_main_fixedcfg_20260315_160824_stable',
  'success',
  'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_0_6144_success.jpg'
)

type ModelItem = {
  key: string
  name: string
  is_enabled: boolean
}

async function fetchModels(request: any): Promise<ModelItem[]> {
  const resp = await request.get('/api/v1/models')
  if (!resp.ok()) {
    throw new Error(`fetch models failed: ${resp.status()} ${resp.statusText()}`)
  }
  const data = await resp.json()
  return data.items ?? []
}

test.beforeEach(async ({ request }) => {
  const models = await fetchModels(request)
  for (const key of modelKeys) {
    const row = models.find((item) => item.key === key)
    if (!row) {
      throw new Error(`model not found: ${key}`)
    }
    if (!row.is_enabled) {
      const resp = await request.patch(`/api/v1/models/${key}`, {
        data: { is_enabled: true },
        headers: { 'Content-Type': 'application/json' }
      })
      if (!resp.ok()) {
        throw new Error(`enable model failed: ${key}`)
      }
    }
  }
})

test.describe.serial('FCOS regression flow', () => {
  for (const key of modelKeys) {
    test(`single-model inference: ${key}`, async ({ page, request }) => {
      const models = await fetchModels(request)
      const row = models.find((item) => item.key === key)
      if (!row) {
        throw new Error(`model not found: ${key}`)
      }

      await page.goto('/')
      await expect(page.getByRole('heading', { name: '任务提交' })).toBeVisible()

      await page.getByRole('radio', { name: '单模型' }).click()

      const combobox = page.getByRole('combobox')
      await combobox.click()
      await page.getByRole('option', { name: row.name }).click()

      const fileInput = page.locator('input[type=\"file\"]')
      await fileInput.setInputFiles(imagePath)

      await page.getByRole('button', { name: '提交任务' }).click()

      await expect(page).toHaveURL(/\\/tasks\\//, { timeout: 30_000 })

      const statusValue = page
        .locator('.metric-card', { hasText: '任务状态' })
        .locator('.metric-value')
      await expect(statusValue).toHaveText('已完成', { timeout: 120_000 })

      await page.waitForSelector('.vis-image-frame, text=暂无结果', { timeout: 30_000 })
      await expect(page.getByText('结构化检测结果')).toBeVisible()
    })
  }
})
