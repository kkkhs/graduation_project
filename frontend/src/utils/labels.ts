import type { TaskMode, TaskStatus, TaskType } from '../api/types'

const TASK_STATUS_LABELS: Record<TaskStatus, string> = {
  queued: '排队中',
  running: '运行中',
  done: '已完成',
  failed: '失败'
}

const TASK_MODE_LABELS: Record<TaskMode, string> = {
  ensemble: '融合模式',
  single: '单模型'
}

const TASK_TYPE_LABELS: Record<TaskType, string> = {
  single: '单图',
  batch: '批量'
}

export function formatTaskStatusLabel(status: TaskStatus): string {
  return TASK_STATUS_LABELS[status] ?? status
}

export function formatTaskModeLabel(mode: TaskMode): string {
  return TASK_MODE_LABELS[mode] ?? mode
}

export function formatTaskTypeLabel(type: TaskType): string {
  return TASK_TYPE_LABELS[type] ?? type
}

export function formatModelSourceLabel(source: string | null | undefined): string {
  if (!source) return '-'
  if (source === 'ensemble' || source === 'fused') return '融合结果'
  return source
}

export function formatFusionLabel(isFused: boolean): string {
  return isFused ? '融合' : '原始'
}

export function formatModelKeyLabel(modelKey: string | null | undefined): string {
  if (!modelKey) return '-'
  return modelKey
    .split(',')
    .map((item) => formatModelSourceLabel(item.trim()))
    .join('，')
}
