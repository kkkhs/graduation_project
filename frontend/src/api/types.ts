export type TaskStatus = 'queued' | 'running' | 'done' | 'failed'
export type TaskMode = 'single' | 'ensemble'
export type TaskType = 'single' | 'batch'

export interface TaskSummary {
  id: number
  type: TaskType
  status: TaskStatus
  mode: TaskMode
  model_key: string | null
  score_thr: number
  input_count: number
  done_count: number
  error_code: string | null
  error_message: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface TaskListResponse {
  items: TaskSummary[]
  total: number
  page: number
  page_size: number
}

export interface ModelItem {
  id: number
  name: string
  key: string
  weight_path: string
  is_enabled: boolean
  created_at: string
}

export interface ResultRecord {
  id: number
  image_name: string
  source_model: string
  is_fused: boolean
  bbox: number[]
  score: number
  category_id: number
}

export interface TaskResultImage {
  image_name: string
  input_url: string | null
  vis_urls: string[]
  output_urls: string[]
  records: ResultRecord[]
}

export interface ModelStats {
  source_model: string
  count: number
  average_score: number
}

export interface TaskResultsResponse {
  task_id: number
  images: TaskResultImage[]
  total_objects: number
  average_score: number
  by_model: ModelStats[]
}
