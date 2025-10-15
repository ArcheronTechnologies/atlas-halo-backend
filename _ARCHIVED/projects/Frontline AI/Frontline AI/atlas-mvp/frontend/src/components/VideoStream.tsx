import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { videoAPI } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

interface StreamStats {
  fps: number;
  avg_processing_time_ms: number;
  is_running: boolean;
  streaming: boolean;
  model_loaded: boolean;
  camera_available: boolean;
  device: string;
}

const VideoStream: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const videoRef = useRef<HTMLImageElement>(null);
  const queryClient = useQueryClient();

  const { 
    isConnected, 
    error: wsError,
    detections,
    alerts,
    startStream: wsStartStream,
    stopStream: wsStopStream,
    clearDetections,
    clearAlerts 
  } = useWebSocket({
    onFrame: (frameData) => {
      setCurrentFrame(frameData);
    },
    onAlert: (alert) => {
      console.log('Alert received:', alert);
      // You can add audio notification here
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Atlas Alert', {
          body: alert.message,
          icon: '/favicon.ico'
        });
      }
    }
  });

  // Query for stream status
  const { data: statusData } = useQuery({
    queryKey: ['stream-status'],
    queryFn: () => videoAPI.getStatus(),
    refetchInterval: 2000,
    select: (response) => response.data as { status: string; stats: StreamStats }
  });

  // Query for detection statistics
  const { data: statsData } = useQuery({
    queryKey: ['detection-stats'],
    queryFn: () => videoAPI.getStats(),
    refetchInterval: 5000,
    select: (response) => response.data as { stats: Record<string, { count: number; avg_confidence: number }> }
  });

  // Mutation for starting stream
  const startStreamMutation = useMutation({
    mutationFn: videoAPI.startStream,
    onSuccess: () => {
      setIsStreaming(true);
      wsStartStream();
      queryClient.invalidateQueries({ queryKey: ['stream-status'] });
    },
    onError: (error) => {
      console.error('Failed to start stream:', error);
    }
  });

  // Mutation for stopping stream
  const stopStreamMutation = useMutation({
    mutationFn: videoAPI.stopStream,
    onSuccess: () => {
      setIsStreaming(false);
      wsStopStream();
      setCurrentFrame(null);
      queryClient.invalidateQueries({ queryKey: ['stream-status'] });
    },
    onError: (error) => {
      console.error('Failed to stop stream:', error);
    }
  });

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const handleToggleStream = () => {
    if (isStreaming) {
      stopStreamMutation.mutate();
    } else {
      startStreamMutation.mutate();
    }
  };

  const stats = statusData?.stats;

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div>
            <h2 className="text-xl font-semibold text-white mb-2">Live Video Stream</h2>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-gray-300">WebSocket: {isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${stats?.camera_available ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-gray-300">Camera: {stats?.camera_available ? 'Available' : 'Unavailable'}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${stats?.model_loaded ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                <span className="text-gray-300">Model: {stats?.model_loaded ? 'Loaded' : 'Loading'}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleToggleStream}
              disabled={startStreamMutation.isPending || stopStreamMutation.isPending || !isConnected}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                isStreaming
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white disabled:bg-gray-600'
              }`}
            >
              {startStreamMutation.isPending || stopStreamMutation.isPending
                ? 'Processing...'
                : isStreaming
                ? 'Stop Stream'
                : 'Start Stream'}
            </button>
            <button
              onClick={() => {
                clearDetections();
                clearAlerts();
              }}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            >
              Clear History
            </button>
          </div>
        </div>

        {wsError && (
          <div className="mt-4 p-3 bg-red-900 bg-opacity-50 border border-red-700 rounded-lg">
            <p className="text-red-300 text-sm">{wsError}</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Display */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
              {currentFrame ? (
                <img
                  ref={videoRef}
                  src={currentFrame}
                  alt="Live stream"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gray-700 rounded-full flex items-center justify-center">
                    <span className="text-2xl">📹</span>
                  </div>
                  <p className="text-gray-400">
                    {!isConnected 
                      ? 'Connecting to server...' 
                      : !isStreaming 
                      ? 'Click "Start Stream" to begin live video'
                      : 'Starting video stream...'}
                  </p>
                </div>
              )}
              
              {/* FPS Overlay */}
              {stats && isStreaming && (
                <div className="absolute top-4 left-4 bg-black bg-opacity-60 px-3 py-1 rounded text-white text-sm">
                  FPS: {stats.fps?.toFixed(1) || '0.0'} | 
                  Latency: {stats.avg_processing_time_ms?.toFixed(0) || '0'}ms |
                  Device: {stats.device}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Detection Panel */}
        <div className="space-y-6">
          {/* Real-time Detections */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium text-white mb-4">Live Detections</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {detections.slice(0, 10).map((detection, index) => (
                <div
                  key={`${detection.timestamp}-${index}`}
                  className="bg-gray-700 rounded p-3 text-sm"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-medium text-white capitalize">
                      {detection.class_name}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      detection.confidence > 0.8 
                        ? 'bg-green-600 text-white'
                        : detection.confidence > 0.6
                        ? 'bg-yellow-600 text-white'
                        : 'bg-red-600 text-white'
                    }`}>
                      {(detection.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-gray-300 text-xs">
                    {new Date(detection.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
              {detections.length === 0 && (
                <p className="text-gray-400 text-sm text-center py-8">
                  No detections yet
                </p>
              )}
            </div>
          </div>

          {/* Recent Alerts */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium text-white mb-4">Recent Alerts</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {alerts.slice(0, 5).map((alert, index) => (
                <div
                  key={`${alert.timestamp}-${index}`}
                  className="bg-red-900 bg-opacity-50 border border-red-700 rounded p-3 text-sm"
                >
                  <div className="flex items-start space-x-2">
                    <span className="text-red-400 mt-0.5">🚨</span>
                    <div className="flex-1">
                      <p className="text-red-300 font-medium">{alert.message}</p>
                      <p className="text-red-400 text-xs mt-1">
                        {new Date(alert.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              {alerts.length === 0 && (
                <p className="text-gray-400 text-sm text-center py-8">
                  No alerts yet
                </p>
              )}
            </div>
          </div>

          {/* Detection Statistics */}
          {statsData && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-medium text-white mb-4">Detection Stats (24h)</h3>
              <div className="space-y-3">
                {Object.entries(statsData.stats).map(([className, stats]) => (
                  <div key={className} className="flex justify-between items-center">
                    <span className="text-gray-300 capitalize">{className}</span>
                    <div className="text-right">
                      <div className="text-white font-medium">{stats.count}</div>
                      <div className="text-gray-400 text-xs">
                        {(stats.avg_confidence * 100).toFixed(1)}% avg
                      </div>
                    </div>
                  </div>
                ))}
                {Object.keys(statsData.stats).length === 0 && (
                  <p className="text-gray-400 text-sm text-center py-4">
                    No detection data yet
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoStream;
