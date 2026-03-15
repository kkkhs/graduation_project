import { Layout, Menu, Tag, Typography } from 'antd'
import { useMemo } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { ModelPage } from './pages/ModelPage'
import { TaskDetailPage } from './pages/TaskDetailPage'
import { TaskListPage } from './pages/TaskListPage'
import { TaskSubmitPage } from './pages/TaskSubmitPage'

const { Content } = Layout

export default function App() {
  const location = useLocation()

  const selectedKey = useMemo(() => {
    if (location.pathname.startsWith('/tasks/')) return 'tasks'
    if (location.pathname.startsWith('/tasks')) return 'tasks'
    if (location.pathname.startsWith('/models')) return 'models'
    return 'submit'
  }, [location.pathname])

  return (
    <div className="page-shell">
      <div className="main-panel">
        <header className="top-dock">
          <div className="header-row">
            <div className="brand-block">
              <Typography.Title className="brand-title" level={4}>
                遥感微小船舶检测系统
              </Typography.Title>
              <div className="brand-subtitle">Web 控制台 · 任务可追踪 · 多模型统一推理</div>
            </div>
            <div className="dock-metrics">
              <div className="dock-metric">
                <span>执行器</span>
                <strong>单线程</strong>
              </div>
              <div className="dock-metric dock-metric-status">
                <span>状态</span>
                <strong className="status-online">在线</strong>
              </div>
              <Tag className="runtime-tag">Local Runtime</Tag>
            </div>
          </div>
        </header>

        <section className="nav-dock">
          <Menu
            className="nav-menu"
            mode="horizontal"
            overflowedIndicator={null}
            selectedKeys={[selectedKey]}
            items={[
              { key: 'submit', label: <Link to="/">任务提交</Link> },
              { key: 'tasks', label: <Link to="/tasks">任务列表</Link> },
              { key: 'models', label: <Link to="/models">模型管理</Link> }
            ]}
          />
        </section>

        <Layout className="content-wrap">
          <Content>
            <Routes>
              <Route path="/" element={<TaskSubmitPage />} />
              <Route path="/tasks" element={<TaskListPage />} />
              <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
              <Route path="/models" element={<ModelPage />} />
            </Routes>
          </Content>
        </Layout>
      </div>
    </div>
  )
}
