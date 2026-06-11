import { useState, useEffect, useCallback } from 'react'
import { listTasks, getTaskStatus, getTaskResult, getVlmConfig, setVlmConfig, getVlmProviders, uploadVideo } from './services/api'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = ['#16a34a', '#ca8a04', '#ea580c', '#dc2626']

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

  const defaultModelForProvider = (provider, config = vlmConfig) => {
    if (provider === 'ollama') return config.ollama_model || 'qwen3-vl:8b'
    if (provider === 'siliconflow') return 'Qwen/Qwen3-VL-32B-Instruct'
    if (provider === 'mock') return 'mock'
    return ''
  }

  const normalizeVlmConfig = (config) => {
    const provider = config.provider || 'none'
    const configuredModel = config.model || ''
    const configuredOllamaModel = config.ollama_model || ''
    const ollamaModel = configuredOllamaModel || (!configuredModel.includes('/') && configuredModel !== 'mock' ? configuredModel : '') || 'qwen3-vl:8b'
    const model = provider === 'ollama'
      ? ollamaModel
      : provider === 'siliconflow'
        ? (configuredModel && configuredModel !== 'qwen3-vl:8b' && configuredModel !== 'mock' ? configuredModel : 'Qwen/Qwen3-VL-32B-Instruct')
        : defaultModelForProvider(provider, config)

    return {
      ...config,
      model,
      ollama_host: config.ollama_host || 'http://localhost:11434',
      ollama_model: ollamaModel,
      max_workers: config.max_workers || 4,
    }
  }

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
    getVlmConfig().then(config => setVlmConfigState(normalizeVlmConfig(config))).catch(() => {})
    getVlmProviders().then(r => setProviders(r.providers || [])).catch(() => {})

    const interval = setInterval(loadTasks, 5000)
    return () => clearInterval(interval)
  }, [loadTasks])

  useEffect(() => {
    if (!activeTaskId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = import.meta.env.VITE_API_BASE ? new URL(import.meta.env.VITE_API_BASE).host : window.location.host
    const wsUrl = `${protocol}//${host}/ws/${activeTaskId}`

    let socket = new WebSocket(wsUrl)

    socket.onopen = () => console.log('[WS] Connected to', wsUrl)
    socket.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (['ack', 'progress', 'complete', 'error'].includes(msg.type)) {
        if (typeof msg.progress === 'number') setProgress(msg.progress)
        if (msg.message) setProgressMsg(msg.message)
        if (msg.type === 'complete' && msg.result) setResult(msg.result)
        if (msg.type === 'error') setProgressMsg(msg.message || '任务失败')
      }
    }
    socket.onclose = () => setWs(null)
    socket.onerror = () => setWs(null)

    setWs(socket)
    return () => socket.close()
  }, [activeTaskId])

  const handleFileChange = (e) => {
    const filePath = e.target.value
    if (filePath) setUploadFile({ name: filePath })
    else setUploadFile(null)
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
        vlm_model: vlmConfig.model || defaultModelForProvider(vlmConfig.provider),
        ollama_host: vlmConfig.ollama_host || 'http://localhost:11434',
        ollama_model: vlmConfig.provider === 'ollama' ? (vlmConfig.model || vlmConfig.ollama_model || 'qwen3-vl:8b') : (vlmConfig.ollama_model || 'qwen3-vl:8b'),
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
      const normalized = normalizeVlmConfig(vlmConfig)
      await setVlmConfig(normalized)
      setVlmConfigState(normalized)
      alert('配置已保存')
    } catch (e) {
      alert('保存失败')
    }
  }

  const routes = [
    { id: 'route1', name: '路线1: 规则引擎', desc: 'ByteTrack + RFAC + 规则引擎 + VLM审计' },
    { id: 'route2', name: '路线2: VLM端到端', desc: '帧采样 + ByteTrack + 多线程VLM分析' },
    { id: 'route3', name: '路线3: 规则+CNN-LSTM', desc: '当前先跑规则+VLM审计，CNN-LSTM待模型确认后启用' },
  ]

  const pieData = result?.level_distribution ? Object.entries(result.level_distribution).map(([k, v]) => ({
    name: k, value: v
  })) : []

  const resultMeta = result ? [
    { label: '总帧数', value: result.total_frames ?? '-' },
    { label: '检测人次', value: result.total_detections ?? '-' },
    { label: 'FPS', value: result.fps ?? '-' },
  ] : []

  return (
    <div className="app">
      <header className="header">
        <div className="brand-block">
          <div className="brand-kicker">RFAC 视频分析平台</div>
          <h1>会议注意力检测系统</h1>
          <p>提交本地视频路径，选择分析路线，并实时查看任务进度与注意力分布。</p>
        </div>
        <div className="status" aria-live="polite">
          <span className="status-dot" />
          <span>{ws ? '任务通道已连接' : '本地 Client 待命'}</span>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <div className="card accent-blue">
            <div className="section-heading">
              <div>
                <div className="card-title">本地视频路径</div>
                <p>请输入服务器或本地 Client 可访问的视频文件路径。</p>
              </div>
            </div>
            <div className="path-panel">
              <label htmlFor="video-path">视频文件路径</label>
              <input
                id="video-path"
                type="text"
                value={uploadFile ? uploadFile.name : ''}
                onChange={handleFileChange}
                placeholder="输入本地视频文件路径，如 /home/user/video.mp4"
              />
              <span>浏览器不会直接读取文件，任务会交给后端和本地 Client 处理。</span>
            </div>
            <button className="btn btn-primary form-action" onClick={handleUpload} disabled={!uploadFile || uploading}>
              {uploading ? '提交中...' : '开始分析'}
            </button>
          </div>

          <div className="card accent-orange">
            <div className="section-heading compact">
              <div className="card-title">选择路线</div>
            </div>
            <div className="route-buttons">
              {routes.map(r => (
                <button
                  key={r.id}
                  className={`route-btn ${selectedRoute === r.id ? 'active' : ''}`}
                  onClick={() => setSelectedRoute(r.id)}
                  aria-pressed={selectedRoute === r.id}
                >
                  <div className="route-name">{r.name}</div>
                  <div className="route-desc">{r.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="section-heading">
              <div>
                <div className="card-title">VLM 配置</div>
                <p>按需启用多模态复核，默认设置适合本地演示。</p>
              </div>
            </div>
            <div className="input-group">
              <label htmlFor="vlm-provider">提供者</label>
              <select
                id="vlm-provider"
                value={vlmConfig.provider || 'none'}
                onChange={e => {
                  const provider = e.target.value
                  setVlmConfigState({ ...vlmConfig, provider, model: defaultModelForProvider(provider) })
                }}
              >
                {providers.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label htmlFor="vlm-trigger">触发等级</label>
              <select id="vlm-trigger" value={vlmConfig.trigger || 'MODERATE'} onChange={e => setVlmConfigState({ ...vlmConfig, trigger: e.target.value })}>
                {['MILD', 'MODERATE', 'SEVERE'].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label htmlFor="vlm-model">模型</label>
              <input
                id="vlm-model"
                value={vlmConfig.model || ''}
                onChange={e => setVlmConfigState({ ...vlmConfig, model: e.target.value })}
                placeholder={vlmConfig.provider === 'ollama' ? 'qwen3-vl:8b' : 'Qwen/Qwen3-VL-32B-Instruct'}
              />
            </div>
            <div className="input-group">
              <label htmlFor="ollama-host">Ollama Host</label>
              <input
                id="ollama-host"
                value={vlmConfig.ollama_host || 'http://localhost:11434'}
                onChange={e => setVlmConfigState({ ...vlmConfig, ollama_host: e.target.value })}
                placeholder="http://localhost:11434"
              />
            </div>
            <div className="input-group">
              <label htmlFor="vlm-api-key">API Key</label>
              <input
                id="vlm-api-key"
                type="password"
                value={vlmConfig.api_key || ''}
                onChange={e => setVlmConfigState({ ...vlmConfig, api_key: e.target.value })}
                placeholder="输入API Key"
              />
            </div>
            <div className="input-group">
              <label htmlFor="vlm-workers">并发线程数</label>
              <input
                id="vlm-workers"
                type="number"
                value={vlmConfig.max_workers || 4}
                onChange={e => setVlmConfigState({ ...vlmConfig, max_workers: parseInt(e.target.value) })}
              />
            </div>
            <button className="btn btn-secondary form-action" onClick={handleVlmConfigSave}>
              保存配置
            </button>
          </div>
        </aside>

        <section className="content">
          <div className="card content-card accent-blue">
            <div className="content-header">
              <div>
                <div className="card-title">实时任务</div>
                <h2>分析队列与进度</h2>
              </div>
              <span className="task-count">{tasks.length} 个任务</span>
            </div>
            {activeTaskId && (
              <div className="active-progress" style={{ '--progress': `${progress}%` }}>
                <div className="progress-copy">
                  <span>当前任务</span>
                  <strong>{activeTaskId}</strong>
                </div>
                <div className="progress-status">
                  <span>{progressMsg || '等待进度更新'}</span>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div className="progress-bar">
                  <div className="fill" />
                </div>
              </div>
            )}
            <div className="task-list">
              {tasks.length === 0 && <div className="empty-state">暂无任务</div>}
              {tasks.map(t => (
                <button
                  key={t.task_id}
                  type="button"
                  className={`task-item ${activeTaskId === t.task_id ? 'active' : ''}`}
                  onClick={() => handleTaskClick(t.task_id)}
                >
                  <div className="task-info">
                    <div className="task-name">{t.video_name}</div>
                    <div className="task-meta">
                      <span>{t.route}</span>
                      <span>{new Date(t.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                  <span className={`task-badge ${t.status}`}>{t.status}</span>
                </button>
              ))}
            </div>
          </div>

          {result && (
            <div className="result-section accent-orange">
              <div className="result-header">
                <div>
                  <div className="card-title">分析结果</div>
                  <h2>RFAC 注意力分布</h2>
                </div>
                <span className="result-route">{result.route}</span>
              </div>
              <div className="stats-grid">
                {Object.entries(result.level_distribution || {}).map(([k, v], i) => (
                  <div key={k} className="stat-card">
                    <div className="value" style={{ color: COLORS[i] }}>{v}</div>
                    <div className="label">{k}</div>
                  </div>
                ))}
              </div>
              <div className="analytics-grid">
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={62}
                        outerRadius={104}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="result-meta-grid">
                  {resultMeta.map(item => (
                    <div key={item.label} className="meta-card">
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
