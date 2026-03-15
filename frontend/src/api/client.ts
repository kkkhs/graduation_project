import type { ModelItem, TaskListResponse, TaskResultsResponse, TaskSummary } from './types'

async function parseJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      detail = body.detail ?? detail
    } catch {
      // ignored
    }
    throw new Error(detail)
  }
  return (await resp.json()) as T
}

export async function fetchHealth() {
  return parseJson(await fetch('/api/v1/health'))
}

export async function fetchModels(): Promise<ModelItem[]> {
  const data = await parseJson<{ items: ModelItem[] }>(await fetch('/api/v1/models'))
  return data.items
}

export async function toggleModel(modelKey: string, isEnabled: boolean): Promise<ModelItem> {
  return parseJson(
    await fetch(`/api/v1/models/${modelKey}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_enabled: isEnabled })
    })
  )
}

export async function submitTask(payload: {
  type: 'single' | 'batch'
  mode: 'single' | 'ensemble'
  modelKey?: string
  scoreThr: number
  files: File[]
}): Promise<{ task_id: number; status: string }> {
  const formData = new FormData()
  formData.append('type', payload.type)
  formData.append('mode', payload.mode)
  if (payload.modelKey) {
    formData.append('model_key', payload.modelKey)
  }
  formData.append('score_thr', String(payload.scoreThr))
  for (const file of payload.files) {
    formData.append('images', file)
  }

  return parseJson(
    await fetch('/api/v1/tasks/infer', {
      method: 'POST',
      body: formData
    })
  )
}

export async function fetchTasks(page = 1, pageSize = 20): Promise<TaskListResponse> {
  return parseJson(await fetch(`/api/v1/tasks?page=${page}&page_size=${pageSize}`))
}

export async function fetchTask(taskId: number): Promise<TaskSummary> {
  return parseJson(await fetch(`/api/v1/tasks/${taskId}`))
}

export async function fetchTaskResults(taskId: number): Promise<TaskResultsResponse> {
  return parseJson(await fetch(`/api/v1/tasks/${taskId}/results`))
}
