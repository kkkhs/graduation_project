import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Radio,
  Row,
  Select,
  Slider,
  Space,
  Tag,
  Typography,
  Upload,
  message
} from 'antd'
import type { RcFile } from 'antd/es/upload'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchModels, submitTask } from '../api/client'
import type { ModelItem } from '../api/types'
import { formatTaskModeLabel, formatTaskTypeLabel } from '../utils/labels'

const { Dragger } = Upload

export function TaskSubmitPage() {
  const [models, setModels] = useState<ModelItem[]>([])
  const [loading, setLoading] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [form] = Form.useForm()
  const navigate = useNavigate()

  useEffect(() => {
    fetchModels()
      .then((rows) => setModels(rows))
      .catch((err: Error) => message.error(`加载模型失败: ${err.message}`))
  }, [])

  const enabledModels = useMemo(() => models.filter((item) => item.is_enabled), [models])
  const taskType = Form.useWatch('type', form) ?? 'single'
  const mode = Form.useWatch('mode', form) ?? 'ensemble'
  const scoreThr = Form.useWatch('score_thr', form) ?? 0.25
  const modelKey = Form.useWatch('model_key', form) as string | undefined

  const selectedModelName = useMemo(() => {
    if (!modelKey) return '-'
    return models.find((item) => item.key === modelKey)?.name ?? modelKey
  }, [modelKey, models])

  const totalUploadSizeMb = useMemo(() => {
    const bytes = files.reduce((acc, file) => acc + file.size, 0)
    return (bytes / 1024 / 1024).toFixed(2)
  }, [files])

  useEffect(() => {
    if (taskType === 'single' && files.length > 1) {
      setFiles((prev) => prev.slice(0, 1))
      message.info('单图模式仅保留第一张图片，可切换到批量模式提交多图。')
    }
  }, [taskType, files.length])

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <div>
        <h2 className="section-title">任务提交</h2>
        <div className="section-desc">上传单图或批量图像，按模式自动调度模型并生成可视化与结构化结果。</div>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">可用模型数</div>
          <div className="metric-value">{enabledModels.length}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">总模型数</div>
          <div className="metric-value">{models.length}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">当前上传</div>
          <div className="metric-value">{files.length}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">默认阈值</div>
          <div className="metric-value">0.25</div>
        </div>
      </div>

      <Alert
        type="info"
        showIcon
        message="融合模式会使用已启用模型，单模型模式适合做定量对比与消融演示。"
      />

      <div className="submit-grid">
        <Card className="panel-card" title="创建推理任务">
          <Form
            form={form}
            layout="vertical"
            initialValues={{ type: 'single', mode: 'ensemble', score_thr: 0.25 }}
            onFinish={async (values) => {
              if (files.length === 0) {
                message.warning('请先上传图片')
                return
              }
              if (values.type === 'single' && files.length !== 1) {
                message.warning('单图任务必须且只能上传 1 张图片')
                return
              }

              setLoading(true)
              try {
                const res = await submitTask({
                  type: values.type,
                  mode: values.mode,
                  modelKey: values.model_key,
                  scoreThr: values.score_thr,
                  files
                })
                message.success(`任务已创建: ${res.task_id}`)
                navigate(`/tasks/${res.task_id}`)
              } catch (err) {
                message.error((err as Error).message)
              } finally {
                setLoading(false)
              }
            }}
          >
            <Row gutter={12}>
              <Col xs={24} md={12}>
                <Form.Item
                  label="任务类型"
                  name="type"
                  extra={<span className="form-tip">单图用于快速验证，批量用于评估稳定性。</span>}
                >
                  <Radio.Group
                    options={[
                      { label: '单图', value: 'single' },
                      { label: '批量', value: 'batch' }
                    ]}
                    optionType="button"
                    buttonStyle="solid"
                  />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item
                  label="推理模式"
                  name="mode"
                  extra={<span className="form-tip">融合模式自动调度全部启用模型。</span>}
                >
                  <Radio.Group
                    options={[
                      { label: '融合模式', value: 'ensemble' },
                      { label: '单模型', value: 'single' }
                    ]}
                    optionType="button"
                    buttonStyle="solid"
                  />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={12}>
              <Col xs={24} md={12}>
                <Form.Item shouldUpdate={(prev, cur) => prev.mode !== cur.mode} noStyle>
                  {({ getFieldValue }) => {
                    const modeValue = getFieldValue('mode')
                    if (modeValue !== 'single') return null
                    return (
                      <Form.Item
                        label="单模型选择"
                        name="model_key"
                        rules={[{ required: true, message: '单模型模式必须选择模型' }]}
                        extra={<span className="form-tip">建议用于不同权重之间的效果对比。</span>}
                      >
                        <Select
                          showSearch
                          placeholder="请选择模型"
                          options={enabledModels.map((item) => ({ label: item.name, value: item.key }))}
                        />
                      </Form.Item>
                    )
                  }}
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="置信度阈值" required>
                  <div className="score-control">
                    <Slider
                      min={0}
                      max={1}
                      step={0.01}
                      value={scoreThr}
                      onChange={(value) => form.setFieldValue('score_thr', value)}
                    />
                    <Form.Item name="score_thr" noStyle>
                      <InputNumber min={0} max={1} step={0.01} precision={2} style={{ width: '100%' }} />
                    </Form.Item>
                  </div>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item label="图片上传">
              <Dragger
                multiple={taskType === 'batch'}
                beforeUpload={(file: RcFile) => {
                  setFiles((prev) => {
                    const duplicated = prev.some(
                      (item) =>
                        item.name === file.name &&
                        item.size === file.size &&
                        item.lastModified === file.lastModified
                    )
                    if (duplicated) return prev
                    if (taskType === 'single') return [file]
                    return [...prev, file]
                  })
                  return false
                }}
                onRemove={(file) => {
                  setFiles((prev) =>
                    prev.filter(
                      (f) =>
                        !(
                          f.name === file.name &&
                          f.size === file.size
                        )
                    )
                  )
                }}
                fileList={files.map((file) => ({
                  uid: `${file.name}-${file.size}-${file.lastModified}`,
                  name: file.name,
                  status: 'done'
                }))}
                accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
              >
                <p style={{ margin: 0, fontWeight: 600 }}>
                  {taskType === 'single' ? '选择 1 张图片进行快速推理' : '拖拽多张图片到此处，或点击选择'}
                </p>
                <p className="dropzone-note">支持 JPG / PNG / BMP / TIFF，系统会自动创建异步任务并写入结果数据库</p>
              </Dragger>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <div className="task-cta-row">
                <Button type="primary" htmlType="submit" loading={loading} size="large">
                  提交任务
                </Button>
                <Typography.Text type="secondary">任务提交后会自动跳转详情页并开始轮询</Typography.Text>
              </div>
            </Form.Item>
          </Form>
        </Card>

        <Card className="panel-card" title="任务快照">
          <div className="queue-preview">
            <div className="queue-item">
              <b>任务类型</b>
              <code>{formatTaskTypeLabel(taskType)}</code>
            </div>
            <div className="queue-item">
              <b>推理模式</b>
              <code>{formatTaskModeLabel(mode)}</code>
            </div>
            <div className="queue-item">
              <b>当前阈值</b>
              <code>{Number(scoreThr).toFixed(2)}</code>
            </div>
            <div className="queue-item">
              <b>生效模型</b>
              <code>{mode === 'single' ? selectedModelName : `${enabledModels.length} 个模型`}</code>
            </div>
            <div className="queue-item">
              <b>上传文件</b>
              <code>{files.length}</code>
            </div>
            <div className="queue-item">
              <b>总大小</b>
              <code>{totalUploadSizeMb} MB</code>
            </div>
          </div>

          <Space direction="vertical" size={10} style={{ width: '100%', marginTop: 12 }}>
            <Typography.Text strong>本次将使用的模型</Typography.Text>
            <div>
              {mode === 'single' ? (
                <Tag>{selectedModelName}</Tag>
              ) : (
                enabledModels.map((item) => <Tag key={item.key}>{item.name}</Tag>)
              )}
            </div>
            <Typography.Text strong>已选文件</Typography.Text>
            <div className="file-chip-list">
              {files.length === 0 ? <Typography.Text type="secondary">暂无上传文件</Typography.Text> : null}
              {files.slice(0, 10).map((file) => (
                <span className="file-chip" key={`${file.name}-${file.size}-${file.lastModified}`}>
                  {file.name}
                </span>
              ))}
              {files.length > 10 ? <span className="file-chip">+{files.length - 10} 个文件</span> : null}
            </div>
          </Space>
        </Card>
      </div>
    </Space>
  )
}
