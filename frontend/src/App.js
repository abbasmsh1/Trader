import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Navbar from './components/Navbar';
import TradersPage from './pages/TradersPage';
import TraderDetailPage from './pages/TraderDetailPage';
import MarketPage from './pages/MarketPage';
import CommunicationsPage from './pages/CommunicationsPage';
import { WebSocketProvider } from './utils/WebSocketContext';

function App() {
  return (
    <WebSocketProvider>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/traders" element={<TradersPage />} />
            <Route path="/traders/:traderId" element={<TraderDetailPage />} />
            <Route path="/market" element={<MarketPage />} />
            <Route path="/communications" element={<CommunicationsPage />} />
          </Routes>
        </Container>
      </Box>
    </WebSocketProvider>
  );
}

export default App; 