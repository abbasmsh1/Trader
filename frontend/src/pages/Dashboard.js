import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Divider,
  Paper,
  Button,
  Avatar,
  Chip
} from '@mui/material';
import { useWebSocket } from '../utils/WebSocketContext';

const Dashboard = () => {
  const { data } = useWebSocket();
  const { traders, market, communications } = data;

  // Sort traders by portfolio value
  const sortedTraders = [...traders].sort((a, b) => {
    const aTotal = Object.entries(a.wallet).reduce((sum, [symbol, amount]) => {
      if (symbol === 'USDT') return sum + amount;
      const price = market[`${symbol}USDT`]?.price || 0;
      return sum + amount * price;
    }, 0);

    const bTotal = Object.entries(b.wallet).reduce((sum, [symbol, amount]) => {
      if (symbol === 'USDT') return sum + amount;
      const price = market[`${symbol}USDT`]?.price || 0;
      return sum + amount * price;
    }, 0);

    return bTotal - aTotal;
  });

  // Get recent communications
  const recentCommunications = communications.slice(0, 5);

  return (
    <Box className="fade-in">
      <Typography variant="h4" gutterBottom>
        AI Crypto Trader Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Leaderboard */}
        <Grid item xs={12} md={6}>
          <Card className="card-hover">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trader Leaderboard
              </Typography>
              <Divider sx={{ mb: 2 }} />

              {sortedTraders.map((trader, index) => {
                const totalValue = Object.entries(trader.wallet).reduce((sum, [symbol, amount]) => {
                  if (symbol === 'USDT') return sum + amount;
                  const price = market[`${symbol}USDT`]?.price || 0;
                  return sum + amount * price;
                }, 0);

                const progress = (totalValue / 100) * 100; // Progress toward $100 goal

                return (
                  <Box key={trader.id} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Avatar sx={{ bgcolor: index === 0 ? 'gold' : index === 1 ? 'silver' : index === 2 ? '#cd7f32' : 'grey', mr: 1 }}>
                        {index + 1}
                      </Avatar>
                      <Typography variant="subtitle1" component={RouterLink} to={`/traders/${trader.id}`} sx={{ textDecoration: 'none', color: 'inherit', flexGrow: 1 }}>
                        {trader.name}
                      </Typography>
                      <Typography variant="subtitle1" color={totalValue >= 20 ? 'success.main' : 'error.main'}>
                        ${totalValue.toFixed(2)}
                      </Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={Math.min(progress, 100)} color={progress >= 100 ? 'success' : 'primary'} sx={{ height: 8, borderRadius: 4 }} />
                  </Box>
                );
              })}

              <Button component={RouterLink} to="/traders" variant="outlined" fullWidth sx={{ mt: 2 }}>
                View All Traders
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Market Overview */}
        <Grid item xs={12} md={6}>
          <Card className="card-hover">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Overview
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Grid container spacing={2}>
                {Object.entries(market).map(([symbol, data]) => (
                  <Grid item xs={6} key={symbol}>
                    <Paper elevation={3} sx={{ p: 2 }}>
                      <Typography variant="subtitle1">{symbol}</Typography>
                      <Typography variant="h6">${data.price?.toFixed(2) || 'N/A'}</Typography>
                      <Chip
                        label={`${data.price_change_percent_24h?.toFixed(2) || 0}%`}
                        color={data.price_change_percent_24h > 0 ? 'success' : 'error'}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                    </Paper>
                  </Grid>
                ))}
              </Grid>

              <Button component={RouterLink} to="/market" variant="outlined" fullWidth sx={{ mt: 2 }}>
                View Market Details
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Communications */}
        <Grid item xs={12}>
          <Card className="card-hover">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Communications
              </Typography>
              <Divider sx={{ mb: 2 }} />

              {recentCommunications.length > 0 ? (
                recentCommunications.map((comm) => (
                  <Box key={comm.id} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
                      <Avatar sx={{ mr: 1, bgcolor: 'primary.main' }}>
                        {comm.trader_name.charAt(0)}
                      </Avatar>
                      <Box>
                        <Typography variant="subtitle1">
                          {comm.trader_name}
                          <Typography variant="caption" sx={{ ml: 1 }}>
                            {new Date(comm.timestamp).toLocaleTimeString()}
                          </Typography>
                        </Typography>
                        <Typography variant="body2">{comm.content}</Typography>
                      </Box>
                    </Box>
                  </Box>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                  No communications yet
                </Typography>
              )}

              <Button component={RouterLink} to="/communications" variant="outlined" fullWidth sx={{ mt: 2 }}>
                View All Communications
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 