import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// AI Capabilities API
export const capabilitiesAPI = {
  // Get system health
  getSystemHealth: () => api.get('/api/ai/health'),
  
  // Get detailed metrics
  getMetrics: () => api.get('/api/ai/metrics'),
  
  // Extract components from text
  extractComponents: (text, options = {}) => 
    api.post('/api/ai/extract-components', { text, ...options }),
  
  // Classify intent
  classifyIntent: (text, subject = '') => 
    api.post('/api/ai/classify-intent', { text, subject }),
  
  // Get recommendations
  getRecommendations: (component, options = {}) => 
    api.post('/api/ai/recommendations', { component, ...options }),
  
  // Analyze scenario
  analyzeScenario: (scenarioType, parameters, affectedComponents, options = {}) =>
    api.post('/api/ai/analyze-scenario', {
      scenario_type: scenarioType,
      parameters,
      affected_components: affectedComponents,
      ...options
    }),
  
  // Analyze supplier
  analyzeSupplier: (supplierId, options = {}) =>
    api.post('/api/ai/analyze-supplier', {
      supplier_id: supplierId,
      ...options
    }),
  
  // Process email
  processEmail: (emailBody, emailSubject = '') =>
    api.post('/api/ai/process-email', {
      email_body: emailBody,
      email_subject: emailSubject
    }),
  
  // Run evaluation
  runEvaluation: (suiteName) =>
    api.post('/api/ai/evaluate', { suite_name: suiteName }),
  
  // Get capabilities list
  getCapabilities: () => api.get('/api/ai/capabilities'),
  
  // Get capability details
  getCapabilityDetails: (capabilityName) => 
    api.get(`/api/ai/capabilities/${capabilityName}`),
};

// Mock API for development/demo purposes
export const mockAPI = {
  // System health with realistic data
  getSystemHealth: () => Promise.resolve({
    data: {
      health_status: 'healthy',
      health_score: 0.94,
      total_capabilities: 9,
      active_alerts: 1,
      critical_alerts: 0,
      capabilities_status: {
        'component_ner': {
          error_rate: 0.02,
          avg_execution_time_ms: 245,
          throughput_per_second: 12.3,
          cache_hit_rate: 0.76
        },
        'intent_classifier': {
          error_rate: 0.01,
          avg_execution_time_ms: 189,
          throughput_per_second: 15.7,
          cache_hit_rate: 0.82
        },
        'component_recommender': {
          error_rate: 0.03,
          avg_execution_time_ms: 1840,
          throughput_per_second: 3.2,
          cache_hit_rate: 0.65
        },
        'scenario_analysis': {
          error_rate: 0.01,
          avg_execution_time_ms: 3200,
          throughput_per_second: 0.8,
          cache_hit_rate: 0.45
        },
        'advanced_supplier_analysis': {
          error_rate: 0.02,
          avg_execution_time_ms: 2100,
          throughput_per_second: 1.5,
          cache_hit_rate: 0.71
        }
      },
      timestamp: new Date().toISOString()
    }
  }),

  // Extract components with mock results
  extractComponents: (text) => Promise.resolve({
    data: {
      success: true,
      data: {
        entities: [
          { text: 'STM32F429ZIT6', label: 'MICROCONTROLLER', confidence: 0.95, start: 0, end: 13 },
          { text: 'LM358', label: 'OPAMP', confidence: 0.88, start: 25, end: 30 }
        ],
        component_count: 2,
        processing_time_ms: 245
      },
      confidence: 0.91,
      execution_time_ms: 245,
      capability_name: 'component_ner'
    }
  }),

  // Classify intent with mock results
  classifyIntent: (text, subject) => Promise.resolve({
    data: {
      success: true,
      data: {
        primary_intent: 'rfq',
        confidence: 0.85,
        secondary_intents: [
          { intent: 'technical_inquiry', confidence: 0.12 },
          { intent: 'general_inquiry', confidence: 0.03 }
        ]
      },
      confidence: 0.85,
      execution_time_ms: 189
    }
  }),

  // Get recommendations with mock results
  getRecommendations: (component) => Promise.resolve({
    data: {
      success: true,
      data: {
        recommendations: [
          {
            component: 'STM32F729ZIT6',
            reason: 'Higher performance alternative in same family',
            confidence: 0.88,
            category: 'microcontroller',
            recommendation_type: 'upgrade',
            estimated_price: 12.50
          },
          {
            component: 'STM32F439ZIT6',
            reason: 'Similar specifications with crypto acceleration',
            confidence: 0.82,
            category: 'microcontroller',
            recommendation_type: 'functional_equivalent',
            estimated_price: 8.75
          },
          {
            component: 'STM32H743ZIT6',
            reason: 'Next generation with improved performance',
            confidence: 0.75,
            category: 'microcontroller',
            recommendation_type: 'upgrade',
            estimated_price: 18.90
          }
        ],
        total_found: 3,
        input_component: component
      },
      confidence: 0.78,
      execution_time_ms: 1840
    }
  }),

  // Scenario analysis with mock results
  analyzeScenario: (scenarioType, parameters, components) => Promise.resolve({
    data: {
      success: true,
      data: {
        scenario_id: `scenario_${Date.now()}`,
        scenario_type: scenarioType,
        parameters: parameters,
        risk_assessment: {
          overall_risk_level: 'medium',
          risk_score: 0.45,
          confidence: 0.78,
          components_analyzed: components.length,
          critical_components: 0,
          high_risk_components: 1
        },
        component_impacts: components.map((comp, idx) => ({
          component_id: comp,
          severity: idx === 0 ? 'medium' : 'low',
          probability: 0.3 + (idx * 0.1),
          impact_areas: ['availability', 'pricing'],
          estimated_cost_impact: 150 + (idx * 50),
          time_to_recovery: 60 + (idx * 30)
        })),
        mitigation_recommendations: [
          {
            priority: 'high',
            action: 'diversify_suppliers',
            timeline: '3-6 months',
            effectiveness: 0.8,
            estimated_cost: 75000
          }
        ]
      },
      confidence: 0.78,
      execution_time_ms: 3200
    }
  }),

  // Supplier analysis with mock results
  analyzeSupplier: (supplierId) => Promise.resolve({
    data: {
      success: true,
      data: {
        supplier_profile: {
          supplier_id: supplierId,
          name: 'Advanced Electronics Ltd',
          country: 'Germany',
          certifications: ['ISO9001', 'ISO14001', 'TS16949']
        },
        overall_score: 7.8,
        module_scores: {
          financial_health: 8.2,
          delivery_performance: 7.9,
          quality: 8.1,
          geopolitical_risk: 7.0,
          sustainability: 6.8,
          innovation: 7.5
        },
        risk_assessment: {
          overall_risk_level: 'low',
          critical_risk_areas: [],
          high_risk_areas: [],
          mitigation_priority: 'low'
        },
        recommendations: [
          {
            category: 'operational',
            priority: 'medium',
            action: 'expand_capacity',
            description: 'Consider expanding production capacity to meet growing demand'
          }
        ]
      },
      confidence: 0.82,
      execution_time_ms: 2100
    }
  }),

  // Get capabilities list
  getCapabilities: () => Promise.resolve({
    data: [
      {
        name: 'component_ner',
        version: '2.0.0',
        status: 'healthy',
        description: 'Named Entity Recognition for electronic components',
        tags: ['ner', 'extraction', 'advanced']
      },
      {
        name: 'intent_classifier',
        version: '2.0.0',
        status: 'healthy',
        description: 'Intent classification for supply chain communications',
        tags: ['classification', 'intent']
      },
      {
        name: 'component_recommender',
        version: '1.0.0',
        status: 'healthy',
        description: 'AI-powered component recommendations',
        tags: ['recommendation', 'ml']
      },
      {
        name: 'scenario_analysis',
        version: '2.0.0',
        status: 'healthy',
        description: 'Advanced scenario analysis for risk assessment',
        tags: ['scenario', 'risk', 'analysis']
      },
      {
        name: 'advanced_supplier_analysis',
        version: '2.0.0',
        status: 'healthy',
        description: 'Comprehensive supplier analysis with modular scoring',
        tags: ['supplier', 'analysis', 'scoring']
      }
    ]
  })
};

// Use mock API in development, real API in production
const isDevelopment = process.env.NODE_ENV === 'development';
export default isDevelopment ? mockAPI : capabilitiesAPI;