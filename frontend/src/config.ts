// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const API_ENDPOINTS = {
  ANALYZE: `${API_BASE_URL}/api/analyze`,
  HEALTH: `${API_BASE_URL}/health`,
  MODELS: `${API_BASE_URL}/models`,
} as const;

