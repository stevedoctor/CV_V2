import { useState, useEffect, useCallback } from 'react'
import { listTasks, getTaskStatus, getTaskResult, getVlmConfig, setVlmConfig, getVlmProviders, uploadVideo } from './services/api'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = ['#22c55e', '#eab308', '#f97316', '#ef4444']

function App() {
  const [selectedRoute, setSelectedRoute] = useState('route1')
  const [tasks, setTasks] = useState([])
  const [activeTaskId, setActiveTaskId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [progressMsg, setProgressMsg] = useState('')
  const [result, setResult] = useState(null)
  const [uploadFile, setUploadFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [vlmConfig, setVlmConfigState] = useState({})
  const [providers, setProviders] = useState([])
  const [ws, setWs] = useState(null)

  const loadTasks = useCallback(async () => {
    try {
      const data = await listTasks({ limit: 20 })
      setTasks(data.tasks || [])
    } catch (e) {
      console.error('加载任务失败:', e)
    }
  }, [])

  useEffect(() => {
    loadTasks()
    getVlmConfig().then(setVlmConfigState).catch(() => {})
    getVlmProviders().then(r => setProviders(r.providers || [])).catch(() => {})

    const interval = setInterval(loadTasks, 5000)
    return () => clearInterval(interval)
  }, [loadTasks])

  useEffect(() => {
    if (!activeTaskId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = import.meta.env.VITE_API_BASE ? new URL(import.meta.env.VITE_API_BASE).host : 'localhost:8000'
    const wsUrl = `${protocol}//${host}/ws/ws/${activeTaskId}`

    let socket = new WebSocket(wsUrl)

    socket.onopen = () => console.log('[WS] Connected to', wsUrl)
    socket.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'ack') {
        setProgress(msg.progress)
        setProgressMsg(msg.message)
      }
    }
    socket.onclose = () => setWs(null)
    socket.onerror = () => setWs(null)

    setWs(socket)
    return () => socket.close()
  }, [activeTaskId])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) setUploadFile(file)
  }

  const handleUpload = async () => {
    const filePath = uploadFile ? uploadFile.name : ''
    if (!filePath) return
    setUploading(true)
    try {
      const data = await uploadVideo({
        video_path: filePath,
        route: selectedRoute,
        vlm_provider: vlmConfig.provider || 'none',
        vlm_trigger: vlmConfig.trigger || 'MODERATE',
        vlm_api_key: vlmConfig.api_key || '',
        vlm_model: vlmConfig.model || '',
        workers: vlmConfig.max_workers || 4,
      })
      setActiveTaskId(data.task_id)
      setProgress(0)
      setResult(null)
      await loadTasks()
    } catch (e) {
      console.error('上传失败:', e)
      alert('上传失败: ' + e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleTaskClick = async (taskId) => {
    setActiveTaskId(taskId)
    try {
      const status = await getTaskStatus(taskId)
      setProgress(status.progress || 0)
      setProgressMsg(status.progress_message || '')
      if (status.status === 'completed') {
        const res = await getTaskResult(taskId)
        setResult(res.result)
      }
    } catch (e) {
      console.error('获取任务失败:', e)
    }
  }

  const handleVlmConfigSave = async () => {
    try {
      await setVlmConfig(vlmConfig)
      alert('配置已保存')
    } catch (e) {
      alert('保存失败')
    }
  }

  const routes = [
    { id: 'route1', name: '路线1: 规则引擎', desc: 'ByteTrack + RFAC + 规则引擎 + VLM审计' },
    { id: 'route2', name: '路线2: VLM端到端', desc: '帧采样 + ByteTrack + 多线程VLM分析' },
    { id: 'route3', name: '路线3: 规则+CNN-LSTM', desc: '规则引擎 + 数据集 + CNN-LSTM集成' },
  ]

  const pieData = result?.level_distribution ? Object.entries(result.level_distribution).map(([k, v]) => ({
    name: k, value: v
  })) : []

  return (
    <div className="app">
      <header className="header">
        <h1>会议注意力检测系统</h1>
        <div className="status">
          <span className="status-dot" />
          <span>本地Client运行中</span>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <div className="card">
            <div className="card-title">上传视频</div>
            <div className="upload-zone">
              <div className="icon">📹</div>
              <p>{uploadFile ? uploadFile.name : '点击或拖拽视频文件'}</p>
              <input type="file" accept=".mp4,.avi,.mov" onChange={handleFileChange} style={{ marginTop: 8 }} />
            </div>
            <button className="btn btn-primary" onClick={handleUpload} disabled={!uploadFile || uploading} style={{ marginTop: 12 }}>
              {uploading ? '上传中...' : '开始分析'}
            </button>
          </div>

          <div className="card">
            <div className="card-title">选择路线</div>
            <div className="route-buttons">
              {routes.map(r => (
                <button
                  key={r.id}
                  className={`route-btn ${selectedRoute === r.id ? 'active' : ''}`}
                  onClick={() => setSelectedRoute(r.id)}
                >
                  <div className="route-name">{r.name}</div>
                  <div className="route-desc">{r.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-title">VLM 配置</div>
            <div className="input-group">
              <label>提供者</label>
              <select value={vlmConfig.provider || 'none'} onChange={e => setVlmConfigState({ ...vlmConfig, provider: e.target.value })}>
                {providers.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label>触发等级</label>
              <select value={vlmConfig.trigger || 'MODERATE'} onChange={e => setVlmConfigState({ ...vlmConfig, trigger: e.target.value })}>
                {['MILD', 'MODERATE', 'SEVERE'].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label>API Key</label>
              <input
                type="password"
                value={vlmConfig.api_key || ''}
                onChange={e => setVlmConfigState({ ...vlmConfig, api_key: e.target.value })}
                placeholder="输入API Key"
              />
            </div>
            <div className="input-group">
              <label>并发线程数</label>
              <input
                type="number"
                value={vlmConfig.max_workers || 4}
                onChange={e => setVlmConfigState({ ...vlmConfig, max_workers: parseInt(e.target.value) })}
              />
            </div>
            <button className="btn btn-primary" onClick={handleVlmConfigSave} style={{ marginTop: 8 }}>
              保存配置
            </button>
          </div>
        </aside>

        <section className="content">
          <div className="card">
            <div className="card-title">任务列表</div>
            {activeTaskId && (
              <div style={{ marginBottom: 12, padding: 12, background: '#0f172a', borderRadius: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>当前: {activeTaskId}</span>
                  <span>{progressMsg}</span>
                </div>
                <div className="progress-bar">
                  <div className="fill" style={{ width: `${progress}%` }} />
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{Math.round(progress)}%</div>
              </div>
            )}
            <div className="task-list">
              {tasks.length === 0 && <div className="empty-state">暂无任务</div>}
              {tasks.map(t => (
                <div
                  key={t.task_id}
                  className="task-item"
                  onClick={() => handleTaskClick(t.task_id)}
                  style={{ cursor: 'pointer', borderColor: activeTaskId === t.task_id ? '#3b82f6' : undefined }}
                >
                  <div className="task-info">
                    <div className="task-name">{t.video_name}</div>
                    <div className="task-meta">{t.route} • {new Date(t.created_at).toLocaleString()}</div>
                  </div>
                  <span className={`task-badge ${t.status}`}>{t.status}</span>
                </div>
              ))}
            </div>
          </div>

          {result && (
            <div className="result-section">
              <div className="result-header">
                <h3>分析结果</h3>
                <span style={{ fontSize: 12, color: '#64748b' }}>{result.route}</span>
              </div>
              <div className="stats-grid">
                {Object.entries(result.level_distribution || {}).map(([k, v], i) => (
                  <div key={k} className="stat-card">
                    <div className="value" style={{ color: COLORS[i] }}>{v}</div>
                    <div className="label">{k}</div>
                  </div>
                ))}
              </div>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ marginTop: 16, fontSize: 12, color: '#64748b' }}>
                总帧数: {result.total_frames} | 检测人次: {result.total_detections} | FPS: {result.fps}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App