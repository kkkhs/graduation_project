import { Progress } from 'antd'
import type { TaskStatus } from '../api/types'

interface TaskProgressBarProps {
  doneCount: number
  totalCount: number
  status?: TaskStatus
  size?: 'small' | 'default'
  className?: string
}

export function TaskProgressBar({
  doneCount,
  totalCount,
  status,
  size = 'small',
  className
}: TaskProgressBarProps) {
  const safeDone = Math.max(0, doneCount)
  const safeTotal = Math.max(0, totalCount)
  const percent = safeTotal > 0 ? Math.min(100, Math.round((safeDone / safeTotal) * 100)) : 0
  const ratioTone = status === 'done' ? 'done' : status === 'failed' ? 'failed' : ''

  return (
    <div className={`progress-with-ratio ${className ?? ''}`.trim()}>
      <Progress
        percent={percent}
        size={size}
        showInfo={false}
        status={status === 'failed' ? 'exception' : status === 'done' ? 'success' : undefined}
      />
      <span className={`progress-ratio ${ratioTone}`}>{safeDone}/{safeTotal}</span>
    </div>
  )
}
