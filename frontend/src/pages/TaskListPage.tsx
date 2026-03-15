import { Button, Card, Space, Table, Tooltip, message } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchTasks } from '../api/client'
import type { TaskSummary } from '../api/types'
import { TaskProgressBar } from '../components/TaskProgressBar'
import { formatModelKeyLabel, formatTaskModeLabel, formatTaskStatusLabel, formatTaskTypeLabel } from '../utils/labels'
import { formatShanghaiTime } from '../utils/time'

function statusClass(status: TaskSummary['status']) {
  if (status === 'done') return 'done'
  if (status === 'running') return 'running'
  if (status === 'failed') return 'failed'
  return 'queued'
}

function typeChipClass(type: TaskSummary['type']) {
  return type === 'single' ? 'type-single' : 'type-batch'
}

function modeChipClass(mode: TaskSummary['mode']) {
  return mode === 'ensemble' ? 'mode-ensemble' : 'mode-single'
}

function renderEllipsis(text: string, maxWidth = 220) {
  return (
    <Tooltip title={text}>
      <span className="truncate-cell" style={{ maxWidth }}>{text}</span>
    </Tooltip>
  )
}

export function TaskListPage() {
  const [rows, setRows] = useState<TaskSummary[]>([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await fetchTasks(1, 50)
      setRows(data.items)
    } catch (err) {
      message.error(`加载任务失败: ${(err as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const timer = setInterval(load, 4000)
    return () => clearInterval(timer)
  }, [])

  const stats = useMemo(() => {
    const total = rows.length
    const done = rows.filter((r) => r.status === 'done').length
    const running = rows.filter((r) => r.status === 'running').length
    const failed = rows.filter((r) => r.status === 'failed').length
    return { total, done, running, failed }
  }, [rows])

  const columns: ColumnsType<TaskSummary> = [
    {
      title: '任务ID',
      dataIndex: 'id',
      width: 96,
      fixed: 'left',
      render: (id: number) => <Link to={`/tasks/${id}`}>#{id}</Link>
    },
    {
      title: '类型',
      dataIndex: 'type',
      width: 100,
      render: (v: TaskSummary['type']) => <span className={`task-chip ${typeChipClass(v)}`}>{formatTaskTypeLabel(v)}</span>
    },
    {
      title: '模式',
      dataIndex: 'mode',
      width: 116,
      render: (v: TaskSummary['mode']) => <span className={`task-chip ${modeChipClass(v)}`}>{formatTaskModeLabel(v)}</span>
    },
    {
      title: '模型',
      dataIndex: 'model_key',
      width: 180,
      render: (v: string | null) => renderEllipsis(formatModelKeyLabel(v))
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 96,
      render: (status: TaskSummary['status']) => (
        <span className={`pill-status ${statusClass(status)}`}>{formatTaskStatusLabel(status)}</span>
      )
    },
    {
      title: '进度',
      width: 210,
      render: (_, row) => <TaskProgressBar doneCount={row.done_count} totalCount={row.input_count} status={row.status} />
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 188,
      render: (v: string) => formatShanghaiTime(v)
    },
    {
      title: '错误信息',
      dataIndex: 'error_message',
      width: 420,
      render: (v: string | null) => renderEllipsis(v ?? '-', 400)
    }
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <div>
        <h2 className="section-title">任务列表</h2>
        <div className="section-desc">每 4 秒自动刷新，支持快速定位失败任务与查看执行进度。</div>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">总任务</div>
          <div className="metric-value">{stats.total}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">运行中</div>
          <div className="metric-value">{stats.running}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">已完成</div>
          <div className="metric-value status-text done">{stats.done}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">失败</div>
          <div className="metric-value status-text failed">{stats.failed}</div>
        </div>
      </div>

      <Card className="panel-card" extra={<Button onClick={load}>刷新</Button>}>
        <Table
          className="table-quiet"
          rowKey="id"
          loading={loading}
          columns={columns}
          dataSource={rows}
          pagination={false}
          size="small"
          tableLayout="fixed"
          scroll={{ x: 1360 }}
        />
      </Card>
    </Space>
  )
}
