import { useEffect, useRef, useState, useCallback } from 'react';

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  timestamp: string;
}

export interface Alert {
  type: string;
  message: string;
  timestamp: string;
  detection?: Detection;
}

export interface WebSocketMessage {
  type: 'frame' | 'detection' | 'alert' | 'pong';
  data?: any;
}

interface UseWebSocketProps {
  url?: string;
  onFrame?: (frameData: string) => void;
  onDetection?: (detection: Detection) => void;
  onAlert?: (alert: Alert) => void;
  autoConnect?: boolean;
}

export const useWebSocket = ({
  url = 'ws://localhost:8000/ws',
  onFrame,
  onDetection,
  onAlert,
  autoConnect = true,
}: UseWebSocketProps = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFrame, setLastFrame] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        
        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          switch (message.type) {
            case 'frame':
              setLastFrame(message.data);
              onFrame?.(message.data);
              break;
              
            case 'detection':
              const detection = message.data as Detection;
              setDetections(prev => [detection, ...prev.slice(0, 99)]);
              onDetection?.(detection);
              break;
              
            case 'alert':
              const alert = message.data as Alert;
              setAlerts(prev => [alert, ...prev.slice(0, 49)]);
              onAlert?.(alert);
              break;
              
            case 'pong':
              // Keep-alive response
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        // Auto-reconnect after delay
        if (autoConnect && !event.wasClean) {
          setError('Connection lost. Attempting to reconnect...');
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setError('Failed to connect to server');
    }
  }, [url, onFrame, onDetection, onAlert, autoConnect]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  const startStream = useCallback(() => {
    return sendMessage({ type: 'start_stream' });
  }, [sendMessage]);

  const stopStream = useCallback(() => {
    return sendMessage({ type: 'stop_stream' });
  }, [sendMessage]);

  const clearDetections = useCallback(() => {
    setDetections([]);
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect, autoConnect]);

  return {
    isConnected,
    error,
    lastFrame,
    detections,
    alerts,
    connect,
    disconnect,
    sendMessage,
    startStream,
    stopStream,
    clearDetections,
    clearAlerts,
  };
};
