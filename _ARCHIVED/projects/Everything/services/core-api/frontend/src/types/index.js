// Type definitions for AI Capabilities Framework

export const CapabilityStatus = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  UNAVAILABLE: 'unavailable',
  LOADING: 'loading',
  ERROR: 'error'
};

export const AlertSeverity = {
  CRITICAL: 'critical',
  WARNING: 'warning',
  INFO: 'info'
};

export const TestStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  PASSED: 'passed',
  FAILED: 'failed'
};

// Default structures for API responses
export const defaultCapabilityResult = {
  success: false,
  data: null,
  confidence: null,
  execution_time_ms: null,
  error: null,
  warnings: [],
  metadata: {},
  capability_name: null,
  capability_version: null,
  fallback_used: false
};

export const defaultSystemHealth = {
  health_status: 'unknown',
  health_score: 0,
  total_capabilities: 0,
  active_alerts: 0,
  critical_alerts: 0,
  capabilities_status: {},
  timestamp: new Date().toISOString()
};

export const defaultPerformanceMetrics = {
  capability_name: '',
  execution_count: 0,
  success_count: 0,
  failure_count: 0,
  avg_execution_time_ms: 0,
  min_execution_time_ms: 0,
  max_execution_time_ms: 0,
  p50_execution_time_ms: 0,
  p95_execution_time_ms: 0,
  p99_execution_time_ms: 0,
  cache_hit_rate: 0,
  error_rate: 0,
  throughput_per_second: 0,
  concurrent_executions_avg: 0,
  last_updated: new Date().toISOString()
};