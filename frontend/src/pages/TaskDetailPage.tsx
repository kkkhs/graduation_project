import {
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Image,
  Row,
  Space,
  Table,
  Tag,
  message
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import ReactECharts from 'echarts-for-react'
import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { fetchTask, fetchTaskResults } from '../api/client'
import type { ModelStats, ResultRecord, TaskResultsResponse, TaskSummary } from '../api/types'
import { TaskProgressBar } from '../components/TaskProgressBar'
import {
  formatFusionLabel,
  formatModelKeyLabel,
  formatModelSourceLabel,
  formatTaskTypeLabel,
  formatTaskModeLabel,
  formatTaskStatusLabel
} from '../utils/labels'
import { formatShanghaiTime } from '../utils/time'

function isFusedSource(sourceModel: string): boolean {
  const normalized = sourceModel.trim().toLowerCase()
  return normalized === 'ensemble' || normalized === 'fused' || normalized === '融合结果'
}

function compareModelSourceFusedLast(a: string, b: string): number {
  const aFused = isFusedSource(a)
  const bFused = isFusedSource(b)
  if (aFused !== bFused) return aFused ? 1 : -1
  return a.localeCompare(b)
}

function buildModelCountOption(byModel: ModelStats[]) {
  const names = byModel.map((item) => formatModelSourceLabel(item.source_model))
  const counts = byModel.map((item) => item.count)
  return {
    grid: { left: 36, right: 16, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: names,
      axisLine: { lineStyle: { color: '#94a3b8' } },
      axisLabel: { color: '#334155' }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      axisLabel: { color: '#475569' }
    },
    series: [
      {
        type: 'bar',
        data: counts,
        barWidth: 30,
        itemStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: '#0f766e' },
              { offset: 1, color: '#14b8a6' }
            ]
          },
          borderRadius: [8, 8, 0, 0]
        }
      }
    ]
  }
}

function buildAvgScoreOption(byModel: ModelStats[]) {
  const names = byModel.map((item) => formatModelSourceLabel(item.source_model))
  const scores = byModel.map((item) => Number(item.average_score.toFixed(3)))
  return {
    grid: { left: 36, right: 16, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: names,
      axisLine: { lineStyle: { color: '#94a3b8' } },
      axisLabel: { color: '#334155' }
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      axisLabel: { color: '#475569' }
    },
    series: [
      {
        type: 'line',
        data: scores,
        smooth: true,
        lineStyle: { width: 3, color: '#0ea5a5' },
        symbolSize: 8,
        itemStyle: { color: '#0f766e' },
        areaStyle: {
          color: 'rgba(15, 118, 110, 0.14)'
        }
      }
    ]
  }
}

function parseVisLabel(url: string): string {
  const fileName = url.split('/').pop() ?? ''
  const stem = fileName.replace(/\.[^.]+$/, '')
  if (stem.endsWith('_vis')) {
    return '结果可视化'
  }

  const marker = '_vis_'
  const markerIndex = stem.indexOf(marker)
  if (markerIndex === -1) {
    return '结果可视化'
  }

  const token = stem.slice(markerIndex + marker.length)
  if (!token) {
    return '结果可视化'
  }

  let clean = token
  try {
    clean = decodeURIComponent(token)
  } catch {
    clean = token
  }
  clean = clean.replace(/[_-]+/g, ' ')
  if (clean === 'fused' || clean === 'ensemble') {
    return '融合结果'
  }
  return `模型: ${clean}`
}

function typeChipClass(type: TaskSummary['type']) {
  return type === 'single' ? 'type-single' : 'type-batch'
}

function modeChipClass(mode: TaskSummary['mode']) {
  return mode === 'ensemble' ? 'mode-ensemble' : 'mode-single'
}

function isNoResultRow(record: ResultRecord): boolean {
  return record.source_model === 'no_result'
}

function normalizeModelSource(sourceModel: string): string {
  const normalized = sourceModel.trim().toLowerCase()
  if (normalized === 'fused' || normalized === 'ensemble' || normalized === '融合结果') {
    return 'ensemble'
  }
  return sourceModel
}

type ModelGroupRow = {
  key: string
  indexLabel: string
  bboxText: string
  scoreText: string
  categoryText: string
  isPlaceholder?: boolean
}

type ModelGroupCard = {
  modelKey: string
  title: string
  count: number
  isFused: boolean
  rows: ModelGroupRow[]
}

type ImageGroupCard = {
  imageName: string
  groups: ModelGroupCard[]
}

export function TaskDetailPage() {
  const params = useParams()
  const taskId = Number(params.taskId)

  const [task, setTask] = useState<TaskSummary | null>(null)
  const [result, setResult] = useState<TaskResultsResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    if (!taskId) return
    setLoading(true)
    try {
      const [taskData, resultData] = await Promise.all([fetchTask(taskId), fetchTaskResults(taskId)])
      setTask(taskData)
      setResult(resultData)
    } catch (err) {
      message.error(`加载任务详情失败: ${(err as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const timer = setInterval(load, 4000)
    return () => clearInterval(timer)
  }, [taskId])

  const tableData = useMemo(() => {
    if (!result) return []
    return result.images
      .flatMap((image) => image.records)
      .slice()
      .sort((a, b) => {
        const imageDiff = a.image_name.localeCompare(b.image_name)
        if (imageDiff !== 0) return imageDiff
        const aNoResult = isNoResultRow(a)
        const bNoResult = isNoResultRow(b)
        if (aNoResult !== bNoResult) return aNoResult ? 1 : -1
        if (a.is_fused !== b.is_fused) return a.is_fused ? 1 : -1
        const modelDiff = compareModelSourceFusedLast(a.source_model, b.source_model)
        if (modelDiff !== 0) return modelDiff
        return a.id - b.id
      })
  }, [result])

  const orderedByModel = useMemo(() => {
    if (!result) return []
    return result.by_model.slice().sort((a, b) => compareModelSourceFusedLast(a.source_model, b.source_model))
  }, [result])

  const groupedImageData = useMemo<ImageGroupCard[]>(() => {
    if (!result || !task) return []

    const selectedModelKeys = (task.model_key ?? '')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
      .sort(compareModelSourceFusedLast)

    return result.images.map((image) => {
      const validRecords = image.records.filter((record) => !isNoResultRow(record))
      const grouped = new Map<string, ResultRecord[]>()

      for (const record of validRecords) {
        const groupKey = normalizeModelSource(record.source_model)
        const existing = grouped.get(groupKey) ?? []
        existing.push(record)
        grouped.set(groupKey, existing)
      }

      const groupKeys = [...selectedModelKeys]
      if (task.mode === 'ensemble') {
        groupKeys.push('ensemble')
      }

      const uniqueGroupKeys = Array.from(new Set(groupKeys)).sort(compareModelSourceFusedLast)
      const groups: ModelGroupCard[] = uniqueGroupKeys.map((groupKey) => {
        const rows = (grouped.get(groupKey) ?? []).slice().sort((a, b) => b.score - a.score)
        const rowItems: ModelGroupRow[] =
          rows.length > 0
            ? rows.map((record, index) => ({
                key: `${groupKey}-${record.id}`,
                indexLabel: String(index + 1),
                bboxText: record.bbox.map((item) => item.toFixed(1)).join(', '),
                scoreText: record.score.toFixed(3),
                categoryText: String(record.category_id)
              }))
            : [
                {
                  key: `${groupKey}-empty`,
                  indexLabel: '-',
                  bboxText: '-',
                  scoreText: '-',
                  categoryText: '-',
                  isPlaceholder: true
                }
              ]

        return {
          modelKey: groupKey,
          title: formatModelSourceLabel(groupKey),
          count: rows.length,
          isFused: groupKey === 'ensemble',
          rows: rowItems
        }
      })

      return {
        imageName: image.image_name,
        groups
      }
    })
  }, [result, task])

  const statusToneClass =
    task?.status === 'done' ? 'status-text done' : task?.status === 'failed' ? 'status-text failed' : ''

  const columns: ColumnsType<ResultRecord> = [
    { title: 'ID', dataIndex: 'id', width: 80 },
    { title: '图片', dataIndex: 'image_name', width: 240 },
    {
      title: '来源模型',
      dataIndex: 'source_model',
      width: 140,
      render: (v: string, record: ResultRecord) => (isNoResultRow(record) ? '无检测结果' : formatModelSourceLabel(v))
    },
    {
      title: '融合',
      dataIndex: 'is_fused',
      width: 90,
      render: (v: boolean, record: ResultRecord) =>
        isNoResultRow(record) ? <span className="pill-status queued">-</span> : <span className={`pill-status ${v ? 'done' : 'queued'}`}>{formatFusionLabel(v)}</span>
    },
    {
      title: '框(x1,y1,x2,y2)',
      dataIndex: 'bbox',
      width: 290,
      render: (bbox: number[], record: ResultRecord) => (isNoResultRow(record) ? '-' : bbox.map((n) => n.toFixed(1)).join(', '))
    },
    {
      title: 'score',
      dataIndex: 'score',
      width: 90,
      render: (v: number, record: ResultRecord) => (isNoResultRow(record) ? '-' : v.toFixed(3))
    }
  ]

  const groupedColumns: ColumnsType<ModelGroupRow> = [
    { title: '序号', dataIndex: 'indexLabel', width: 72 },
    { title: '框(x1,y1,x2,y2)', dataIndex: 'bboxText' },
    { title: 'score', dataIndex: 'scoreText', width: 96 },
    { title: '类别', dataIndex: 'categoryText', width: 88 }
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <div>
        <h2 className="section-title">任务详情 #{taskId}</h2>
        <div className="section-desc">实时查看任务状态、检测产物和模型对比统计。</div>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">任务状态</div>
          <div className={`metric-value ${statusToneClass}`}>{task ? formatTaskStatusLabel(task.status) : '-'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">处理进度</div>
          {task ? (
            <TaskProgressBar
              doneCount={task.done_count}
              totalCount={task.input_count}
              status={task.status}
              className="metric-progress-view"
            />
          ) : (
            <div className="metric-value">-</div>
          )}
        </div>
        <div className="metric-card">
          <div className="metric-label">检测总数</div>
          <div className="metric-value">{result?.total_objects ?? 0}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">平均置信度</div>
          <div className="metric-value">{result ? result.average_score.toFixed(3) : '0.000'}</div>
        </div>
      </div>

      <Button onClick={load} loading={loading}>
        刷新
      </Button>

      {task ? (
        <Card className="panel-card" title="任务配置与状态">
          <Descriptions bordered column={2}>
            <Descriptions.Item label="类型">
              <span className={`task-chip ${typeChipClass(task.type)}`}>{formatTaskTypeLabel(task.type)}</span>
            </Descriptions.Item>
            <Descriptions.Item label="模式">
              <span className={`task-chip ${modeChipClass(task.mode)}`}>{formatTaskModeLabel(task.mode)}</span>
            </Descriptions.Item>
            <Descriptions.Item label="模型">{formatModelKeyLabel(task.model_key)}</Descriptions.Item>
            <Descriptions.Item label="阈值">{task.score_thr}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{formatShanghaiTime(task.created_at)}</Descriptions.Item>
            <Descriptions.Item label="开始时间">{formatShanghaiTime(task.started_at)}</Descriptions.Item>
            <Descriptions.Item label="结束时间">{formatShanghaiTime(task.finished_at)}</Descriptions.Item>
            <Descriptions.Item label="错误信息" span={2}>
              {task.error_message ?? '-'}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      ) : null}

      {result ? (
        <>
          <Row gutter={[12, 12]}>
            <Col xs={24} lg={12}>
              <Card className="panel-card" title="各模型目标数">
                <ReactECharts option={buildModelCountOption(orderedByModel)} style={{ height: 280 }} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card className="panel-card" title="各模型平均置信度">
                <ReactECharts option={buildAvgScoreOption(orderedByModel)} style={{ height: 280 }} />
              </Card>
            </Col>
          </Row>

          <Card className="panel-card" title="可视化结果">
            {result.images.length === 0 ? (
              <Empty description="暂无结果" />
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {result.images.map((image) => {
                  const modelNames = Array.from(new Set(image.records.map((record) => record.source_model)))
                    .sort(compareModelSourceFusedLast)
                    .map((name) => formatModelSourceLabel(name))
                    .filter(Boolean)
                  return (
                    <Card
                      key={image.image_name}
                      type="inner"
                      className="result-image-card"
                      title={
                        <div className="result-image-head">
                          <span>{image.image_name}</span>
                          <div className="model-tag-row">
                            {modelNames.length === 0 ? <Tag>-</Tag> : null}
                            {modelNames.map((name) => (
                              <Tag key={name} color="cyan">
                                {name}
                              </Tag>
                            ))}
                          </div>
                        </div>
                      }
                    >
                      <div className="media-grid">
                        {image.input_url ? (
                          <div className="vis-image-frame">
                            <div className="vis-label">输入图</div>
                            <Image src={image.input_url} />
                          </div>
                        ) : null}
                        {image.vis_urls.map((url) => (
                          <div className="vis-image-frame" key={url}>
                            <div className="vis-label">{parseVisLabel(url)}</div>
                            <Image src={url} />
                          </div>
                        ))}
                      </div>
                    </Card>
                  )
                })}
              </Space>
            )}
          </Card>

          <Card className="panel-card" title="结构化结果概览">
            {groupedImageData.length === 0 ? (
              <Empty description="暂无结果" />
            ) : (
              <Space direction="vertical" style={{ width: '100%' }} size={12}>
                {groupedImageData.map((image) => (
                  <Card
                    key={`grouped-${image.imageName}`}
                    type="inner"
                    className="result-image-card grouped-result-card"
                    title={<div className="result-image-head"><span>{image.imageName}</span></div>}
                  >
                    <div className="grouped-model-grid">
                      {image.groups.map((group) => (
                        <div className="grouped-model-card" key={`${image.imageName}-${group.modelKey}`}>
                          <div className="grouped-model-head">
                            <div className="grouped-model-title-row">
                              <span className="grouped-model-title">{group.title}</span>
                              <span className={`pill-status ${group.isFused ? 'done' : 'queued'}`}>
                                {group.isFused ? '融合结果' : '原始模型结果'}
                              </span>
                            </div>
                            <div className="grouped-model-meta">
                              {group.count > 0 ? `检测到 ${group.count} 个结果` : '无检测结果'}
                            </div>
                          </div>
                          <Table
                            className="table-quiet grouped-result-table"
                            rowKey="key"
                            columns={groupedColumns}
                            dataSource={group.rows}
                            pagination={false}
                            size="small"
                          />
                        </div>
                      ))}
                    </div>
                  </Card>
                ))}
              </Space>
            )}
          </Card>

          <Card className="panel-card" title="原始明细表">
            <Table
              className="table-quiet"
              rowKey="id"
              columns={columns}
              dataSource={tableData}
              pagination={{ pageSize: 10 }}
              scroll={{ x: 980 }}
            />
          </Card>
        </>
      ) : null}
    </Space>
  )
}
