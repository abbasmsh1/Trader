/**
 * Portfolio management JavaScript.
 */

// Global variables
let currentAgentId = null;
let portfolioData = null;
let tradeHistory = [];
let performanceMetrics = {};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Get agent ID from URL or use default
    const urlParams = new URLSearchParams(window.location.search);
    currentAgentId = urlParams.get('agent_id') || 'default_agent';
    
    // Load initial data
    loadPortfolioData();
    loadTradeHistory();
    loadPerformanceMetrics();
    
    // Set up refresh button
    document.getElementById('refresh-btn').addEventListener('click', function() {
        loadPortfolioData();
        loadTradeHistory();
        loadPerformanceMetrics();
    });
});

// Load portfolio data
async function loadPortfolioData() {
    try {
        const response = await fetch(`/api/portfolio/${currentAgentId}`);
        if (!response.ok) {
            throw new Error('Failed to load portfolio data');
        }
        
        const data = await response.json();
        portfolioData = data;
        
        // Update UI
        updatePortfolioUI(data);
    } catch (error) {
        console.error('Error loading portfolio:', error);
        showError('Failed to load portfolio data');
    }
}

// Load trade history
async function loadTradeHistory() {
    try {
        const response = await fetch(`/api/trades/${currentAgentId}`);
        if (!response.ok) {
            throw new Error('Failed to load trade history');
        }
        
        tradeHistory = await response.json();
        
        // Update trade history table
        updateTradeHistoryTable();
    } catch (error) {
        console.error('Error loading trade history:', error);
        showError('Failed to load trade history');
    }
}

// Load performance metrics
async function loadPerformanceMetrics() {
    try {
        const response = await fetch(`/api/performance/${currentAgentId}`);
        if (!response.ok) {
            throw new Error('Failed to load performance metrics');
        }
        
        const metrics = await response.json();
        performanceMetrics = metrics[0] || {};  // Get latest metrics
        
        // Update performance metrics display
        updatePerformanceMetrics();
    } catch (error) {
        console.error('Error loading performance metrics:', error);
        showError('Failed to load performance metrics');
    }
}

// Update portfolio UI
function updatePortfolioUI(data) {
    if (!data) return;
    
    // Update summary cards
    document.getElementById('total-value').textContent = formatCurrency(data.portfolio.value);
    document.getElementById('base-currency').textContent = formatCurrency(data.portfolio.base_currency);
    document.getElementById('assets-value').textContent = formatCurrency(data.portfolio.assets_value);
    document.getElementById('num-assets').textContent = data.portfolio.num_assets;
    
    // Update portfolio table
    updatePortfolioTable(data.portfolio.holdings);
    
    // Update recent trades
    updateRecentTrades(data.recent_trades);
}

// Update portfolio table
function updatePortfolioTable(holdings) {
    const tableBody = document.getElementById('portfolio-table-body');
    tableBody.innerHTML = '';
    
    holdings.forEach(holding => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${holding.symbol}</td>
            <td>${formatNumber(holding.quantity)}</td>
            <td>${formatCurrency(holding.price)}</td>
            <td class="${holding.price_change >= 0 ? 'text-success' : 'text-danger'}">
                ${formatPercentage(holding.price_change)}
            </td>
            <td>${formatCurrency(holding.value)}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="showTradeModal('buy', '${holding.symbol}')">Buy</button>
                <button class="btn btn-sm btn-danger" onclick="showTradeModal('sell', '${holding.symbol}')">Sell</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Update trade history table
function updateTradeHistoryTable() {
    const tableBody = document.getElementById('trade-history-body');
    tableBody.innerHTML = '';
    
    tradeHistory.forEach(trade => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${formatDateTime(trade.timestamp)}</td>
            <td>${trade.symbol}</td>
            <td class="${trade.type === 'buy' ? 'text-success' : 'text-danger'}">${trade.type.toUpperCase()}</td>
            <td>${formatNumber(trade.quantity)}</td>
            <td>${formatCurrency(trade.price)}</td>
            <td>${formatCurrency(trade.value)}</td>
            <td>${formatCurrency(trade.fee)}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Update performance metrics
function updatePerformanceMetrics() {
    const metrics = performanceMetrics;
    
    // Update metrics display
    document.getElementById('total-trades').textContent = metrics.total_trades || 0;
    document.getElementById('win-rate').textContent = formatPercentage(metrics.win_rate || 0);
    document.getElementById('profit-factor').textContent = formatNumber(metrics.profit_factor || 0);
    document.getElementById('profit-percentage').textContent = formatPercentage(metrics.profit_percentage || 0);
    document.getElementById('average-win').textContent = formatPercentage(metrics.average_win || 0);
    document.getElementById('average-loss').textContent = formatPercentage(metrics.average_loss || 0);
    document.getElementById('largest-win').textContent = formatPercentage(metrics.largest_win || 0);
    document.getElementById('largest-loss').textContent = formatPercentage(metrics.largest_loss || 0);
}

// Show trade modal
function showTradeModal(traderId) {
    const modal = new bootstrap.Modal(document.getElementById('trade-modal'));
    
    // Store trader ID
    $('#trade-trader-id').val(traderId);
    
    // Set up symbol change handler
    $('#trade-symbol').on('change', updateCurrentPrice);
    
    // Set up action change handler
    $('#trade-action').on('change', updateTradeForm);
    
    // Set up amount input handler
    $('#trade-amount').on('input', updateTradeValue);
    
    // Initial update
    updateCurrentPrice();
    updateTradeForm();
    
    modal.show();
}

// Update current price display
async function updateCurrentPrice() {
    const symbol = $('#trade-symbol').val();
    try {
        const response = await fetch('/api/prices');
        const prices = await response.json();
        
        if (prices[symbol]) {
            const priceData = prices[symbol];
            const priceChange = priceData.change_24h;
            const changeClass = priceChange >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = priceChange >= 0 ? 'arrow-up' : 'arrow-down';
            
            $('#trade-price').html(`
                <div class="price-info">
                    <div class="current">Current Price: $${priceData.price.toFixed(2)}</div>
                    <div class="change ${changeClass}">
                        <i class="fas fa-${changeIcon}"></i> ${Math.abs(priceChange).toFixed(2)}%
                    </div>
                    <div class="range">24h Range: $${priceData.low_24h.toFixed(2)} - $${priceData.high_24h.toFixed(2)}</div>
                    <div class="volume">Volume: $${formatNumber(priceData.volume)}</div>
                </div>
            `);
        } else {
            $('#trade-price').html('<div class="text-danger">Price data not available</div>');
        }
    } catch (error) {
        console.error('Error fetching price:', error);
        $('#trade-price').html('<div class="text-danger">Error fetching price data</div>');
    }
}

// Update trade form based on action
function updateTradeForm() {
    const action = $('#trade-action').val();
    const amountLabel = action === 'buy' ? 'Amount (USDT)' : 'Amount (Crypto)';
    $('#trade-amount').prev('label').text(amountLabel);
}

// Update trade value calculation
function updateTradeValue() {
    const amount = parseFloat($('#trade-amount').val()) || 0;
    const symbol = $('#trade-symbol').val();
    const action = $('#trade-action').val();
    
    try {
        const priceText = $('#trade-price .current').text();
        const price = parseFloat(priceText.match(/\$([0-9.]+)/)[1]);
        
        const value = action === 'buy' ? amount : amount * price;
        const otherAmount = action === 'buy' ? amount / price : amount * price;
        
        const valueText = `Total Value: $${value.toFixed(2)}`;
        const otherAmountText = `${action === 'buy' ? 'You will receive' : 'You will pay'}: ${otherAmount.toFixed(8)} ${symbol.split('/')[0]}`;
        
        $('#trade-amount').next('.form-text').html(`${valueText}<br>${otherAmountText}`);
    } catch (error) {
        console.error('Error calculating trade value:', error);
    }
}

// Execute trade
async function executeTrade() {
    const traderId = $('#trade-trader-id').val();
    const action = $('#trade-action').val();
    const symbol = $('#trade-symbol').val();
    const amount = parseFloat($('#trade-amount').val());
    
    if (!amount || amount < 5) {
        showErrorMessage('Minimum trade amount is $5.00');
        return;
    }
    
    try {
        const response = await fetch('/api/trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                trader_id: traderId,
                action: action,
                symbol: symbol,
                amount: amount
            }),
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccessMessage(result.message);
            bootstrap.Modal.getInstance(document.getElementById('trade-modal')).hide();
            loadTraderPortfolios(); // Refresh portfolios
        } else {
            showErrorMessage(result.message || 'Trade failed');
        }
    } catch (error) {
        console.error('Error executing trade:', error);
        showErrorMessage('Failed to execute trade');
    }
}

// Format large numbers
function formatNumber(num) {
    if (num >= 1e9) {
        return (num / 1e9).toFixed(2) + 'B';
    }
    if (num >= 1e6) {
        return (num / 1e6).toFixed(2) + 'M';
    }
    if (num >= 1e3) {
        return (num / 1e3).toFixed(2) + 'K';
    }
    return num.toFixed(2);
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

function formatDateTime(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function showError(message) {
    // Implement error notification
    console.error(message);
}

function showSuccess(message) {
    // Implement success notification
    console.log(message);
}

function getCurrentPrice(symbol) {
    // Get current price from portfolio data
    const holding = portfolioData.portfolio.holdings.find(h => h.symbol === symbol);
    return holding ? holding.price : 0;
}

// Portfolio JavaScript

// Document ready handler
$(document).ready(function() {
    loadTraderPortfolios();
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    // Trade modal form submission
    $('#trade-form').on('submit', function(e) {
        e.preventDefault();
        executeTrade();
    });
}

// Load trader portfolios
async function loadTraderPortfolios() {
    try {
        // First get the list of traders
        const tradersResponse = await fetch('/api/traders');
        if (!tradersResponse.ok) throw new Error('Failed to fetch traders');
        
        const tradersData = await tradersResponse.json();
        
        const portfolioGrid = document.getElementById('portfolioGrid');
        portfolioGrid.innerHTML = '';
        
        // Load portfolio for each trader
        for (const trader of tradersData.traders) {
            try {
                const portfolioResponse = await fetch(`/api/portfolio/${trader.id}`);
                if (!portfolioResponse.ok) {
                    console.warn(`No portfolio found for trader ${trader.id}`);
                    continue;
                }
                
                const portfolio = await portfolioResponse.json();
                
                const portfolioCard = document.createElement('div');
                portfolioCard.className = 'portfolio-card';
                
                // Calculate total value and performance
                const totalValue = portfolio.total_value || 0;
                const initialValue = 20.00; // Initial investment
                const performancePercent = ((totalValue - initialValue) / initialValue * 100).toFixed(2);
                const performanceClass = performancePercent >= 0 ? 'positive' : 'negative';
                
                // Create holdings list with performance indicators
                const holdingsList = portfolio.holdings.map(holding => `
                    <div class="holding-item">
                        <span class="holding-symbol">${holding.symbol}</span>
                        <span class="holding-value">$${holding.value.toFixed(2)}</span>
                    </div>
                `).join('');
                
                portfolioCard.innerHTML = `
                    <div class="card-header">
                        <h3>${trader.name}</h3>
                        <span class="trader-style">${trader.style}</span>
                    </div>
                    <div class="card-description">
                        ${trader.description}
                    </div>
                    <div class="portfolio-stats">
                        <div class="portfolio-value">$${totalValue.toFixed(2)}</div>
                        <div class="performance ${performanceClass}">
                            <i class="fas fa-${performancePercent >= 0 ? 'arrow-up' : 'arrow-down'}"></i>
                            ${Math.abs(performancePercent)}%
                        </div>
                    </div>
                    <div class="holdings">
                        ${holdingsList}
                    </div>
                    <div class="card-footer">
                        <button class="btn-trade" onclick="showTradeModal('${trader.id}')">
                            <i class="fas fa-exchange-alt"></i> Trade
                        </button>
                    </div>
                `;
                
                portfolioGrid.appendChild(portfolioCard);
            } catch (error) {
                console.error(`Error loading portfolio for ${trader.id}:`, error);
            }
        }
    } catch (error) {
        console.error('Error loading trader portfolios:', error);
        showErrorMessage('Failed to load trader portfolios');
    }
}

// Show success message
function showSuccessMessage(message) {
    const alertContainer = document.getElementById('alert-container');
    const alert = document.createElement('div');
    alert.className = 'alert alert-success';
    alert.textContent = message;
    alertContainer.appendChild(alert);
    
    setTimeout(() => alert.remove(), 5000);
}

// Show error message
function showErrorMessage(message) {
    const alertContainer = document.getElementById('alert-container');
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger';
    alert.textContent = message;
    alertContainer.appendChild(alert);
    
    setTimeout(() => alert.remove(), 5000);
} 