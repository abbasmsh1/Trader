// Crypto Trader Dashboard JavaScript

// Socket.io connection
const socket = io();

// Chart.js configuration
let portfolioChart;
const portfolioChartData = {
    labels: [],
    datasets: [{
        label: 'Portfolio Value (USD)',
        backgroundColor: 'rgba(78, 115, 223, 0.05)',
        borderColor: 'rgba(78, 115, 223, 1)',
        pointBackgroundColor: 'rgba(78, 115, 223, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(78, 115, 223, 1)',
        pointRadius: 3,
        pointHoverRadius: 5,
        data: [],
        fill: true,
        tension: 0.1
    }]
};

// Price change data for 24h comparison
let priceChangeData = {};

// Store portfolio history to calculate changes
let portfolioHistory = [];

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the portfolio chart
    initPortfolioChart();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize UI with default data
    updateUIWithDefaults();
    
    // Connect system status indicators
    connectSystemControls();
});

// Initialize Chart.js portfolio chart
function initPortfolioChart() {
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: portfolioChartData,
        options: {
            maintainAspectRatio: false,
            layout: {
                padding: {
                    left: 10,
                    right: 25,
                    top: 25,
                    bottom: 0
                }
            },
            scales: {
                x: {
                    time: {
                        unit: 'hour'
                    },
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        maxTicksLimit: 7
                    }
                },
                y: {
                    ticks: {
                        maxTicksLimit: 5,
                        padding: 10,
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    },
                    grid: {
                        color: "rgb(234, 236, 244)",
                        zeroLineColor: "rgb(234, 236, 244)",
                        drawBorder: false,
                        borderDash: [2],
                        zeroLineBorderDash: [2]
                    }
                },
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: "rgb(255,255,255)",
                    bodyColor: "#858796",
                    titleMarginBottom: 10,
                    titleColor: '#6e707e',
                    titleFontSize: 14,
                    borderColor: '#dddfeb',
                    borderWidth: 1,
                    xPadding: 15,
                    yPadding: 15,
                    displayColors: false,
                    intersect: false,
                    mode: 'index',
                    caretPadding: 10,
                    callbacks: {
                        label: function(context) {
                            return 'Portfolio Value: $' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Set up event listeners for the page
function setupEventListeners() {
    // Start system button
    document.getElementById('start-system').addEventListener('click', function() {
        startTradingSystem();
    });
    
    // Stop system button
    document.getElementById('stop-system').addEventListener('click', function() {
        stopTradingSystem();
    });
    
    // Socket.io event listeners
    socket.on('portfolio_update', function(data) {
        updatePortfolioData(data);
    });
    
    socket.on('price_update', function(data) {
        updatePriceData(data);
    });
    
    socket.on('system_status', function(data) {
        updateSystemStatus(data);
    });
    
    // For demo mode: Buy form submission
    const buyForm = document.getElementById('demo-buy-form');
    if (buyForm) {
        buyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            executeDemoBuy();
        });
    }
    
    // For demo mode: Sell form submission
    const sellForm = document.getElementById('demo-sell-form');
    if (sellForm) {
        sellForm.addEventListener('submit', function(e) {
            e.preventDefault();
            executeDemoSell();
        });
    }
    
    // Sell symbol change event to update balance display
    const sellSymbol = document.getElementById('sell-symbol');
    if (sellSymbol) {
        sellSymbol.addEventListener('change', function() {
            updateCurrentBalance();
        });
    }
}

// Update UI with default values before data starts coming in
function updateUIWithDefaults() {
    // Add 24 hours of empty data for the chart
    const now = new Date();
    for (let i = 24; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 60 * 60 * 1000));
        portfolioChartData.labels.push(formatTime(time));
        portfolioChartData.datasets[0].data.push(null);
    }
    portfolioChart.update();
    
    // Fetch initial data from API
    fetchInitialData();
}

// Format time for chart labels
function formatTime(date) {
    return date.getHours().toString().padStart(2, '0') + ':' + 
           date.getMinutes().toString().padStart(2, '0');
}

// Fetch initial data from API endpoints
function fetchInitialData() {
    // Get portfolio data
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            updatePortfolioData(data);
        })
        .catch(error => console.error('Error fetching portfolio:', error));

    // Get price data
    fetch('/api/prices')
        .then(response => response.json())
        .then(data => {
            updatePriceData(data);
        })
        .catch(error => console.error('Error fetching prices:', error));
        
    // Get system status
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => console.error('Error fetching system status:', error));
}

// Update portfolio data in the UI
function updatePortfolioData(data) {
    // Add to portfolio history for calculating changes
    portfolioHistory.push({
        timestamp: new Date(data.timestamp),
        value: data.total_value || 0
    });
    
    // Limit history to last 24 hours
    if (portfolioHistory.length > 144) { // 144 = 24 hours of 10 minute updates
        portfolioHistory.shift();
    }
    
    // Update portfolio value
    const portfolioValue = document.getElementById('portfolio-value');
    if (portfolioValue) {
        portfolioValue.textContent = '$' + (data.total_value || 0).toFixed(2);
    }
    
    // Update active positions count
    const activePositions = document.getElementById('active-positions');
    if (activePositions) {
        let positionCount = 0;
        if (data.holdings) {
            positionCount = Object.keys(data.holdings).filter(key => key !== 'USDT').length;
        }
        activePositions.textContent = positionCount;
    }
    
    // Update profit/loss
    const profitLoss = document.getElementById('profit-loss');
    if (profitLoss) {
        // Calculate 24h change
        let changeValue = 0;
        let changePercent = 0;
        
        if (portfolioHistory.length > 1) {
            const oldestValue = portfolioHistory[0].value;
            const newestValue = portfolioHistory[portfolioHistory.length - 1].value;
            
            changeValue = newestValue - oldestValue;
            if (oldestValue > 0) {
                changePercent = (changeValue / oldestValue) * 100;
            }
            
            const sign = changeValue >= 0 ? '+' : '';
            profitLoss.textContent = sign + '$' + changeValue.toFixed(2) + ' (' + sign + changePercent.toFixed(2) + '%)';
            
            // Set color based on positive/negative
            profitLoss.classList.remove('text-success', 'text-danger');
            profitLoss.classList.add(changeValue >= 0 ? 'text-success' : 'text-danger');
        }
    }
    
    // Update holdings table
    const holdingsTable = document.getElementById('holdings-table');
    if (holdingsTable && data.holdings) {
        let tableContent = '';
        Object.keys(data.holdings).forEach(asset => {
            if (asset !== 'USDT' && data.holdings[asset].value > 0) {
                tableContent += `
                    <tr>
                        <td>${asset}</td>
                        <td>${data.holdings[asset].amount.toFixed(6)}</td>
                        <td>$${data.holdings[asset].value.toFixed(2)}</td>
                    </tr>
                `;
            }
        });
        
        // Add USDT balance
        if (data.balances && data.balances.USDT) {
            tableContent += `
                <tr>
                    <td>USDT</td>
                    <td>${data.balances.USDT.toFixed(2)}</td>
                    <td>$${data.balances.USDT.toFixed(2)}</td>
                </tr>
            `;
        }
        
        holdingsTable.innerHTML = tableContent;
    }
    
    // Update chart with new data point
    if (data.timestamp && data.total_value) {
        const time = new Date(data.timestamp);
        
        // Add new data point to chart
        portfolioChartData.labels.push(formatTime(time));
        portfolioChartData.datasets[0].data.push(data.total_value);
        
        // Remove oldest data point if we have more than 24 hours
        if (portfolioChartData.labels.length > 144) {
            portfolioChartData.labels.shift();
            portfolioChartData.datasets[0].data.shift();
        }
        
        // Update chart
        portfolioChart.update();
    }
}

// Update price data in the UI
function updatePriceData(data) {
    // Calculate price changes if we have previous data
    Object.keys(data).forEach(symbol => {
        const currentPrice = data[symbol];
        
        if (!priceChangeData[symbol]) {
            // First data point, no change calculation possible
            priceChangeData[symbol] = {
                price: currentPrice,
                lastPrice: currentPrice,
                change24h: 0,
                changePercent24h: 0
            };
        } else {
            // Already have data, calculate change since last update
            const lastPrice = priceChangeData[symbol].price;
            const change = currentPrice - lastPrice;
            const changePercent = (change / lastPrice) * 100;
            
            priceChangeData[symbol] = {
                price: currentPrice,
                lastPrice: lastPrice,
                change24h: change,
                changePercent24h: changePercent
            };
        }
    });
    
    // Update price table
    const priceTable = document.getElementById('price-table');
    if (priceTable) {
        let tableContent = '';
        Object.keys(data).forEach(symbol => {
            const price = data[symbol];
            const change = priceChangeData[symbol].changePercent24h || 0;
            const changeClass = change >= 0 ? 'text-success' : 'text-danger';
            const changeSign = change >= 0 ? '+' : '';
            
            tableContent += `
                <tr>
                    <td>${symbol}</td>
                    <td>$${price.toFixed(2)}</td>
                    <td class="${changeClass}">${changeSign}${change.toFixed(2)}%</td>
                </tr>
            `;
        });
        priceTable.innerHTML = tableContent;
    }
    
    // Update current balance in sell form
    updateCurrentBalance();
}

// Update the current balance display in the sell form
function updateCurrentBalance() {
    const sellSymbol = document.getElementById('sell-symbol');
    const currentBalance = document.getElementById('current-balance');
    
    if (sellSymbol && currentBalance) {
        // Fetch portfolio data to get current balance
        fetch('/api/portfolio')
            .then(response => response.json())
            .then(data => {
                const symbol = sellSymbol.value;
                let balance = 0;
                
                if (data.holdings && data.holdings[symbol]) {
                    balance = data.holdings[symbol].amount;
                }
                
                currentBalance.textContent = balance.toFixed(6);
            })
            .catch(error => console.error('Error fetching balance:', error));
    }
}

// Update system status in the UI
function updateSystemStatus(data) {
    const statusIndicator = document.getElementById('status-indicator');
    const startButton = document.getElementById('start-system');
    const stopButton = document.getElementById('stop-system');
    
    if (statusIndicator && startButton && stopButton) {
        const isRunning = data.status === 'running';
        
        // Update status indicator
        statusIndicator.textContent = isRunning ? 'Running' : 'Stopped';
        statusIndicator.classList.remove('bg-success', 'bg-danger');
        statusIndicator.classList.add(isRunning ? 'bg-success' : 'bg-danger');
        
        // Update buttons
        startButton.disabled = isRunning;
        stopButton.disabled = !isRunning;
    }
}

// Connect system control buttons
function connectSystemControls() {
    const startButton = document.getElementById('start-system');
    const stopButton = document.getElementById('stop-system');
    
    if (startButton) {
        startButton.addEventListener('click', startTradingSystem);
    }
    
    if (stopButton) {
        stopButton.addEventListener('click', stopTradingSystem);
    }
}

// Start the trading system
function startTradingSystem() {
    fetch('/api/system/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'started') {
            updateSystemStatus({ status: 'running' });
        }
    })
    .catch(error => console.error('Error starting system:', error));
}

// Stop the trading system
function stopTradingSystem() {
    fetch('/api/system/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'stopped') {
            updateSystemStatus({ status: 'stopped' });
        }
    })
    .catch(error => console.error('Error stopping system:', error));
}

// Execute a demo buy order
function executeDemoBuy() {
    const symbol = document.getElementById('buy-symbol').value;
    const amount = parseFloat(document.getElementById('buy-amount').value);
    
    fetch('/api/demo/buy', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            symbol: symbol,
            amount: amount
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success notification
            alert(`Successfully bought ${data.quantity.toFixed(6)} ${data.symbol} for $${data.amount.toFixed(2)}`);
            
            // Refresh data
            fetchInitialData();
        } else {
            // Show error
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => console.error('Error executing buy:', error));
}

// Execute a demo sell order
function executeDemoSell() {
    const symbol = document.getElementById('sell-symbol').value;
    const amountInput = document.getElementById('sell-amount');
    
    // If amount is empty or 0, sell all
    let quantity = amountInput.value ? parseFloat(amountInput.value) : 0;
    
    fetch('/api/demo/sell', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            symbol: symbol,
            quantity: quantity
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success notification
            alert(`Successfully sold ${data.quantity.toFixed(6)} ${data.symbol} for $${data.amount.toFixed(2)}`);
            
            // Refresh data
            fetchInitialData();
            
            // Clear amount input
            if (amountInput) {
                amountInput.value = '';
            }
        } else {
            // Show error
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => console.error('Error executing sell:', error));
} 