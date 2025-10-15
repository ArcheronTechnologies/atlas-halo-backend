import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { configAPI } from '../services/api';

interface ConfidenceThresholds {
  person: number;
  car: number;
  truck: number;
  motorcycle: number;
  weapon: number;
  [key: string]: number;
}

interface Config {
  confidence_thresholds: ConfidenceThresholds;
  alerts_enabled: boolean;
  audio_alerts: boolean;
}

const ConfigPanel: React.FC = () => {
  const queryClient = useQueryClient();
  const [localThresholds, setLocalThresholds] = useState<ConfidenceThresholds>({
    person: 0.5,
    car: 0.5,
    truck: 0.5,
    motorcycle: 0.5,
    weapon: 0.3,
  });

  // Query for configuration
  const { data: configData, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: () => configAPI.getConfig(),
    select: (response) => response.data as Config,
    onSuccess: (data) => {
      if (data.confidence_thresholds) {
        setLocalThresholds(data.confidence_thresholds);
      }
    }
  });

  // Mutation for updating confidence thresholds
  const updateThresholdsMutation = useMutation({
    mutationFn: (thresholds: ConfidenceThresholds) => 
      configAPI.updateConfidenceThresholds({ ...thresholds }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    }
  });

  // Mutation for toggling alerts
  const toggleAlertsMutation = useMutation({
    mutationFn: (enabled: boolean) => 
      enabled ? configAPI.enableAlerts() : configAPI.disableAlerts(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    }
  });

  // Mutation for toggling audio alerts
  const toggleAudioAlertsMutation = useMutation({
    mutationFn: (enabled: boolean) => 
      enabled ? configAPI.enableAudioAlerts() : configAPI.disableAudioAlerts(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    }
  });

  const handleThresholdChange = (className: keyof ConfidenceThresholds, value: number) => {
    setLocalThresholds(prev => ({
      ...prev,
      [className]: value
    }));
  };

  const handleSaveThresholds = () => {
    updateThresholdsMutation.mutate(localThresholds);
  };

  const handleResetThresholds = () => {
    const defaultThresholds = {
      person: 0.5,
      car: 0.5,
      truck: 0.5,
      motorcycle: 0.5,
      weapon: 0.3,
    };
    setLocalThresholds(defaultThresholds);
    updateThresholdsMutation.mutate(defaultThresholds);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-white">Loading configuration...</div>
      </div>
    );
  }

  const config = configData || { confidence_thresholds: localThresholds, alerts_enabled: true, audio_alerts: true };
  const hasChanges = JSON.stringify(localThresholds) !== JSON.stringify(config.confidence_thresholds);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-6">Configuration</h2>
        
        {/* Confidence Thresholds */}
        <div className="mb-8">
          <h3 className="text-lg font-medium text-white mb-4">Detection Confidence Thresholds</h3>
          <p className="text-gray-400 text-sm mb-6">
            Set the minimum confidence level required for each object type to trigger a detection.
            Higher values reduce false positives but may miss some detections.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(localThresholds).map(([className, threshold]) => (
              <div key={className} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-white font-medium capitalize">
                    {className}
                  </label>
                  <span className="text-blue-400 font-mono text-sm">
                    {(threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => 
                    handleThresholdChange(className as keyof ConfidenceThresholds, parseFloat(e.target.value))
                  }
                  className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>10%</span>
                  <span>90%</span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center space-x-3 mt-6">
            <button
              onClick={handleSaveThresholds}
              disabled={!hasChanges || updateThresholdsMutation.isPending}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
            >
              {updateThresholdsMutation.isPending ? 'Saving...' : 'Save Changes'}
            </button>
            <button
              onClick={handleResetThresholds}
              disabled={updateThresholdsMutation.isPending}
              className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            >
              Reset to Defaults
            </button>
            {hasChanges && (
              <span className="text-yellow-400 text-sm">
                You have unsaved changes
              </span>
            )}
          </div>
        </div>

        {/* Alert Settings */}
        <div className="border-t border-gray-700 pt-8">
          <h3 className="text-lg font-medium text-white mb-4">Alert Settings</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
              <div>
                <h4 className="text-white font-medium">Visual Alerts</h4>
                <p className="text-gray-400 text-sm">
                  Show visual notifications when high-priority objects are detected
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.alerts_enabled}
                  onChange={(e) => toggleAlertsMutation.mutate(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
              <div>
                <h4 className="text-white font-medium">Audio Alerts</h4>
                <p className="text-gray-400 text-sm">
                  Play sound notifications for critical detections like weapons
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.audio_alerts}
                  onChange={(e) => toggleAudioAlertsMutation.mutate(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>
        </div>

        {/* Performance Info */}
        <div className="border-t border-gray-700 pt-8">
          <h3 className="text-lg font-medium text-white mb-4">Performance Tips</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-4">
              <h4 className="text-blue-300 font-medium mb-2">🚀 Optimization</h4>
              <p className="text-blue-200 text-sm">
                Lower confidence thresholds increase detection sensitivity but may impact performance.
                Use higher values for faster processing.
              </p>
            </div>
            <div className="bg-green-900 bg-opacity-30 border border-green-700 rounded-lg p-4">
              <h4 className="text-green-300 font-medium mb-2">✅ Best Practice</h4>
              <p className="text-green-200 text-sm">
                Keep weapon detection threshold lower (30%) for security while setting vehicle 
                detection higher (50%) to reduce noise.
              </p>
            </div>
            <div className="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-4">
              <h4 className="text-yellow-300 font-medium mb-2">⚡ Performance</h4>
              <p className="text-yellow-200 text-sm">
                The system is optimized for Apple Silicon M1. Expected performance: 20+ FPS 
                with sub-100ms latency.
              </p>
            </div>
            <div className="bg-purple-900 bg-opacity-30 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-300 font-medium mb-2">🔧 Hardware</h4>
              <p className="text-purple-200 text-sm">
                Uses Core ML optimization for Apple M1 Neural Engine acceleration.
                CPU fallback available for compatibility.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfigPanel;
