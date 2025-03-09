import React from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Chip,
  useTheme
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  ShowChart as ShowChartIcon,
  Forum as ForumIcon
} from '@mui/icons-material';
import { useWebSocket } from '../utils/WebSocketContext';

const Navbar = () => {
  const theme = useTheme();
  const location = useLocation();
  const { connected } = useWebSocket();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            textDecoration: 'none',
            color: 'inherit',
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }}
        >
          <ShowChartIcon />
          AI Crypto Trader
        </Typography>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            component={RouterLink}
            to="/"
            color={isActive('/') ? 'secondary' : 'inherit'}
            startIcon={<DashboardIcon />}
          >
            Dashboard
          </Button>

          <Button
            component={RouterLink}
            to="/traders"
            color={isActive('/traders') ? 'secondary' : 'inherit'}
            startIcon={<PeopleIcon />}
          >
            Traders
          </Button>

          <Button
            component={RouterLink}
            to="/market"
            color={isActive('/market') ? 'secondary' : 'inherit'}
            startIcon={<ShowChartIcon />}
          >
            Market
          </Button>

          <Button
            component={RouterLink}
            to="/communications"
            color={isActive('/communications') ? 'secondary' : 'inherit'}
            startIcon={<ForumIcon />}
          >
            Chat
          </Button>
        </Box>

        <Chip
          label={connected ? 'Connected' : 'Disconnected'}
          color={connected ? 'success' : 'error'}
          size="small"
          sx={{ ml: 2 }}
        />
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 