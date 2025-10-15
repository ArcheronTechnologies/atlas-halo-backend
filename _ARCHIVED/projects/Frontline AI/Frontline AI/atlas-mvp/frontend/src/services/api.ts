import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Video API
export const videoAPI = {
  startStream: () => api.post('/api/video/start'),
  stopStream: () => api.post('/api/video/stop'),
  getStatus: () => api.get('/api/video/status'),
  getDetections: (limit = 100) => api.get(`/api/video/detections?limit=${limit}`),
  getStats: (hours = 24) => api.get(`/api/video/stats?hours=${hours}`),
};

// Configuration API
export const configAPI = {
  getConfig: () => api.get('/api/config/'),
  updateConfig: (key: string, value: any) => 
    api.post('/api/config/', { key, value }),
  updateConfidenceThresholds: (thresholds: Record<string, number>) =>
    api.put('/api/config/confidence-thresholds', thresholds),
  enableAlerts: () => api.post('/api/config/alerts/enable'),
  disableAlerts: () => api.post('/api/config/alerts/disable'),
  enableAudioAlerts: () => api.post('/api/config/audio-alerts/enable'),
  disableAudioAlerts: () => api.post('/api/config/audio-alerts/disable'),
};

// Training API
export const trainingAPI = {
  getModels: () => api.get('/api/training/models'),
  getActiveModel: () => api.get('/api/training/models/active'),
  activateModel: (modelId: number) => 
    api.post(`/api/training/models/${modelId}/activate`),
  uploadImages: (formData: FormData) =>
    api.post('/api/training/upload-images', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }),
  saveAnnotations: (projectName: string, annotations: any) =>
    api.post(`/api/training/projects/${projectName}/annotate`, annotations),
  startTraining: (projectName: string, epochs = 10, batchSize = 16) =>
    api.post(`/api/training/projects/${projectName}/train?epochs=${epochs}&batch_size=${batchSize}`),
  getProjects: () => api.get('/api/training/projects'),
  deleteProject: (projectName: string) =>
    api.delete(`/api/training/projects/${projectName}`),
  getProjectImages: (projectName: string) =>
    api.get(`/api/training/projects/${projectName}/images`),
};

// Health check
export const healthAPI = {
  check: () => api.get('/health'),
  root: () => api.get('/'),
};

export default api;