import { Button, Card, Space, Switch, Table, Tooltip, Typography, message } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useEffect, useMemo, useState } from 'react'
import { fetchModels, toggleModel } from '../api/client'
import type { ModelItem } from '../api/types'
import { formatShanghaiTime } from '../utils/time'

function shortPath(path: string) {
  if (path.length <= 58) return path
  return `${path.slice(0, 22)}...${path.slice(-30)}`
}

export function ModelPage() {
  const [rows, setRows] = useState<ModelItem[]>([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await fetchModels()
      setRows(data)
    } catch (err) {
      message.error(`加载模型失败: ${(err as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const stats = useMemo(() => {
    const total = rows.length
    const enabled = rows.filter((item) => item.is_enabled).length
    return { total, enabled }
  }, [rows])

  const columns: ColumnsType<ModelItem> = [
    { title: '模型名', dataIndex: 'name', width: 140 },
    { title: 'key', dataIndex: 'key', width: 140, responsive: ['sm'] },
    {
      title: '权重路径',
      dataIndex: 'weight_path',
      responsive: ['md'],
      render: (value: string) => (
        <Tooltip title={value}>
          <span>{shortPath(value)}</span>
        </Tooltip>
      )
    },
    {
      title: '启用',
      width: 110,
      render: (_, row) => (
        <Switch
          checked={row.is_enabled}
          onChange={async (checked) => {
            try {
              await toggleModel(row.key, checked)
              message.success(`模型 ${row.key} 已${checked ? '启用' : '禁用'}`)
              load()
            } catch (err) {
              message.error((err as Error).message)
            }
          }}
        />
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 180,
      responsive: ['lg'],
      render: (value: string) => formatShanghaiTime(value)
    }
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <div>
        <h2 className="section-title">模型管理</h2>
        <div className="section-desc">维护当前推理可用模型和启用状态，支持实验阶段快速切换。</div>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">总模型数</div>
          <div className="metric-value">{stats.total}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">启用模型数</div>
          <div className="metric-value">{stats.enabled}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">禁用模型数</div>
          <div className="metric-value">{Math.max(0, stats.total - stats.enabled)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">管理模式</div>
          <div className="metric-value" style={{ fontSize: '1.05rem' }}>在线切换</div>
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
          scroll={{ x: 960 }}
        />
      </Card>
    </Space>
  )
}
