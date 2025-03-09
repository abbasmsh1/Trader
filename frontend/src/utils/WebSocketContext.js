import React, { createContext, useContext, useState, useEffect } from 'react';

const WebSocketContext = createContext(null);

export const useWebSocket = () => useContext(WebSocketContext);

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState({
    traders: [],
    market: {},
    communications: []
  });

  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);

    // Connection opened
    ws.addEventListener('open', (event) => {
      console.log('Connected to WebSocket server');
      setConnected(true);
    });

    // Listen for messages
    ws.addEventListener('message', (event) => {
      try {
        const message = JSON.parse(event.data);
        setData({
          traders: message.traders || [],
          market: message.market || {},
          communications: message.communications || []
        });
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    // Connection closed
    ws.addEventListener('close', (event) => {
      console.log('Disconnected from WebSocket server');
      setConnected(false);
    });

    // Connection error
    ws.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
      setConnected(false);
    });

    setSocket(ws);

    // Clean up on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Reconnect function
  const reconnect = () => {
    if (socket) {
      socket.close();
    }
    
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
    
    ws.addEventListener('open', (event) => {
      console.log('Reconnected to WebSocket server');
      setConnected(true);
    });
    
    ws.addEventListener('message', (event) => {
      try {
        const message = JSON.parse(event.data);
        setData({
          traders: message.traders || [],
          market: message.market || {},
          communications: message.communications || []
        });
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });
    
    ws.addEventListener('close', (event) => {
      console.log('Disconnected from WebSocket server');
      setConnected(false);
    });
    
    ws.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
      setConnected(false);
    });
    
    setSocket(ws);
  };

  return (
    <WebSocketContext.Provider value={{ connected, data, reconnect }}>
      {children}
    </WebSocketContext.Provider>
  );
}; 