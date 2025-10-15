import React, { useState } from 'react';
import VideoStream from './components/VideoStream';
import TrainingPanel from './components/TrainingPanel';
import AlertsPanel from './components/AlertsPanel';
import ConfigPanel from './components/ConfigPanel';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

type Tab = 'live' | 'training' | 'alerts' | 'config';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('live');

  const tabs = [
    { id: 'live', label: 'Live Video', icon: '📹' },
    { id: 'training', label: 'Training', icon: '🧠' },
    { id: 'alerts', label: 'Alerts', icon: '🚨' },
    { id: 'config', label: 'Config', icon: '⚙️' },
  ];

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-900 text-white">
        <header className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold text-sm">A</span>
                  </div>
                  <h1 className="text-xl font-bold text-white">Atlas MVP</h1>
                </div>
                <div className="hidden sm:block">
                  <div className="flex items-center space-x-1 bg-gray-700 rounded-lg p-1">
                    {tabs.map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as Tab)}
                        className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                          activeTab === tab.id
                            ? 'bg-blue-600 text-white'
                            : 'text-gray-300 hover:text-white hover:bg-gray-600'
                        }`}
                      >
                        <span>{tab.icon}</span>
                        <span>{tab.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-300">System Online</span>
                </div>
              </div>
            </div>
            
            {/* Mobile tabs */}
            <div className="sm:hidden pb-4">
              <div className="flex space-x-1 bg-gray-700 rounded-lg p-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as Tab)}
                    className={`flex-1 flex flex-col items-center space-y-1 px-2 py-2 rounded-md text-xs font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-600'
                    }`}
                  >
                    <span className="text-base">{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {activeTab === 'live' && <VideoStream />}
          {activeTab === 'training' && <TrainingPanel />}
          {activeTab === 'alerts' && <AlertsPanel />}
          {activeTab === 'config' && <ConfigPanel />}
        </main>

        <footer className="bg-gray-800 border-t border-gray-700 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex justify-between items-center text-sm text-gray-400">
              <div>
                Atlas MVP - Edge AI Perception System
              </div>
              <div>
                Built for MacBook Air M1
              </div>
            </div>
          </div>
        </footer>
      </div>
    </QueryClientProvider>
  );
}

export default App;