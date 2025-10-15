import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { videoAPI } from '../services/api';
import { useWebSocket, Alert } from '../hooks/useWebSocket';

interface Detection {
  id: number;
  timestamp: string;
  class_name: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
}

const AlertsPanel: React.FC = () => {
  const [timeRange, setTimeRange] = useState<number>(24);
  const [filterClass, setFilterClass] = useState<string>('all');
  
  const { alerts, clearAlerts } = useWebSocket({ autoConnect: true });

  // Query for recent detections
  const { data: detectionsData } = useQuery({
    queryKey: ['detections', 500],
    queryFn: () => videoAPI.getDetections(500),
    refetchInterval: 10000,
    select: (response) => response.data.detections as Detection[]
  });

  // Query for detection statistics
  const { data: statsData } = useQuery({
    queryKey: ['detection-stats', timeRange],
    queryFn: () => videoAPI.getStats(timeRange),
    refetchInterval: 30000,
    select: (response) => response.data.stats as Record<string, { count: number; avg_confidence: number }>
  });

  const detections = detectionsData || [];
  const stats = statsData || {};

  // Filter detections by class
  const filteredDetections = filterClass === 'all' 
    ? detections 
    : detections.filter(d => d.class_name === filterClass);

  // Get high-priority detections (weapons and low confidence)
  const highPriorityDetections = detections.filter(d => 
    d.class_name === 'weapon' || d.confidence < 0.4
  );

  const detectionClasses = detections.reduce<string[]>(
    (acc, detection) => {
      if (!acc.includes(detection.class_name)) {
        acc.push(detection.class_name);
      }
      return acc;
    },
    ['all']
  );

  const getClassIcon = (className: string) => {
    const icons: Record<string, string> = {
      person: '👤',
      car: '🚗',
      truck: '🚛',
      motorcycle: '🏍️',
      weapon: '🔫',
    };
    return icons[className] || '📦';
  };

  const getAlertPriority = (alert: Alert) => {
    if (alert.type === 'weapon_detected') return 'critical';
    return 'medium';
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
          <h2 className="text-xl font-semibold text-white">Alerts & Detection History</h2>
          <div className="flex items-center space-x-4 mt-4 sm:mt-0">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              className="bg-gray-700 text-white rounded-lg px-3 py-2 text-sm"
            >
              <option value={1}>Last Hour</option>
              <option value={6}>Last 6 Hours</option>
              <option value={24}>Last 24 Hours</option>
              <option value={168}>Last Week</option>
            </select>
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              className="bg-gray-700 text-white rounded-lg px-3 py-2 text-sm"
            >
              {detectionClasses.map(className => (
                <option key={className} value={className}>
                  {className === 'all' ? 'All Classes' : className.charAt(0).toUpperCase() + className.slice(1)}
                </option>
              ))}
            </select>
            <button
              onClick={clearAlerts}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Clear Alerts
            </button>
          </div>
        </div>

        {/* Statistics Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">📊</span>
              </div>
              <div>
                <p className="text-gray-300 text-sm">Total Detections</p>
                <p className="text-white text-xl font-semibold">{detections.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-red-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">🚨</span>
              </div>
              <div>
                <p className="text-gray-300 text-sm">Active Alerts</p>
                <p className="text-white text-xl font-semibold">{alerts.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-yellow-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">⚠️</span>
              </div>
              <div>
                <p className="text-gray-300 text-sm">High Priority</p>
                <p className="text-white text-xl font-semibold">{highPriorityDetections.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">✅</span>
              </div>
              <div>
                <p className="text-gray-300 text-sm">Avg Confidence</p>
                <p className="text-white text-xl font-semibold">
                  {detections.length > 0 
                    ? `${((detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length) * 100).toFixed(0)}%`
                    : '0%'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Class Breakdown */}
        {Object.keys(stats).length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-medium text-white mb-4">Detection Breakdown</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {Object.entries(stats).map(([className, classStats]) => (
                <div key={className} className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-2xl mb-2">{getClassIcon(className)}</div>
                  <p className="text-gray-300 text-sm capitalize">{className}</p>
                  <p className="text-white font-semibold">{classStats.count}</p>
                  <p className="text-gray-400 text-xs">
                    {(classStats.avg_confidence * 100).toFixed(0)}% avg
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Alerts */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-medium text-white mb-4">Live Alerts</h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {alerts.map((alert, index) => (
              <div
                key={`${alert.timestamp}-${index}`}
                className={`border rounded-lg p-4 ${
                  getAlertPriority(alert) === 'critical'
                    ? 'bg-red-900 bg-opacity-30 border-red-700'
                    : 'bg-yellow-900 bg-opacity-30 border-yellow-700'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    getAlertPriority(alert) === 'critical'
                      ? 'bg-red-600'
                      : 'bg-yellow-600'
                  }`}>
                    <span className="text-white text-sm">
                      {getAlertPriority(alert) === 'critical' ? '🚨' : '⚠️'}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <p className={`font-medium ${
                        getAlertPriority(alert) === 'critical'
                          ? 'text-red-300'
                          : 'text-yellow-300'
                      }`}>
                        {alert.type.replace('_', ' ').toUpperCase()}
                      </p>
                      <span className="text-xs text-gray-400">
                        {formatTimestamp(alert.timestamp)}
                      </span>
                    </div>
                    <p className={`text-sm ${
                      getAlertPriority(alert) === 'critical'
                        ? 'text-red-200'
                        : 'text-yellow-200'
                    }`}>
                      {alert.message}
                    </p>
                    {alert.detection && (
                      <div className="mt-2 text-xs text-gray-400">
                        Confidence: {(alert.detection.confidence * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {alerts.length === 0 && (
              <div className="text-center py-8">
                <div className="text-gray-400 mb-2">🔕</div>
                <p className="text-gray-400">No active alerts</p>
              </div>
            )}
          </div>
        </div>

        {/* Detection History */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-medium text-white mb-4">Recent Detections</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {filteredDetections.slice(0, 50).map((detection) => (
              <div
                key={detection.id}
                className="bg-gray-700 rounded-lg p-3 flex items-center justify-between"
              >
                <div className="flex items-center space-x-3">
                  <span className="text-xl">{getClassIcon(detection.class_name)}</span>
                  <div>
                    <p className="text-white font-medium capitalize">{detection.class_name}</p>
                    <p className="text-gray-400 text-xs">{formatTimestamp(detection.timestamp)}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    detection.confidence > 0.8 
                      ? 'bg-green-600 text-white'
                      : detection.confidence > 0.6
                      ? 'bg-yellow-600 text-white'
                      : 'bg-red-600 text-white'
                  }`}>
                    {(detection.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
            {filteredDetections.length === 0 && (
              <div className="text-center py-8">
                <div className="text-gray-400 mb-2">📭</div>
                <p className="text-gray-400">
                  {filterClass === 'all' ? 'No detections yet' : `No ${filterClass} detections`}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertsPanel;
