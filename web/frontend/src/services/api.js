import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://121.41.192.58'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

export const uploadVideo = async (data) => {
  const response = await api.post('/api/analysis/upload', data)
  return response.data
}

export const getTaskStatus = async (taskId) => {
  const response = await api.get(`/api/analysis/tasks/${taskId}/status`)
  return response.data
}

export const getTaskResult = async (taskId) => {
  const response = await api.get(`/api/analysis/tasks/${taskId}/result`)
  return response.data
}

export const listTasks = async (params = {}) => {
  const response = await api.get('/api/tasks', { params })
  return response.data
}

export const deleteTask = async (taskId) => {
  const response = await api.delete(`/api/tasks/${taskId}`)
  return response.data
}

export const getVlmConfig = async () => {
  const response = await api.get('/api/config/vlm')
  return response.data
}

export const setVlmConfig = async (config) => {
  const response = await api.post('/api/config/vlm', config)
  return response.data
}

export const getVlmProviders = async () => {
  const response = await api.get('/api/config/vlm/providers')
  return response.data
}

export default api