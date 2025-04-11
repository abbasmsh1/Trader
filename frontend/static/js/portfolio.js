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
function showTradeModal(type, symbol) {
    const modal = document.getElementById('trade-modal');
    const modalTitle = document.getElementById('trade-modal-title');
    const tradeForm = document.getElementById('trade-form');
    
    // Update modal title
    modalTitle.textContent = `${type.toUpperCase()} ${symbol}`;
    
    // Update form
    tradeForm.innerHTML = `
        <div class="mb-3">
            <label for="trade-amount" class="form-label">Amount (${type === 'buy' ? 'USDT' : symbol})</label>
            <input type="number" class="form-control" id="trade-amount" min="5" step="0.000001" required>
            <div class="form-text">Minimum trade amount: $5.00</div>
        </div>
        <div class="mb-3">
            <label for="trade-price" class="form-label">Price (USDT)</label>
            <input type="number" class="form-control" id="trade-price" readonly>
        </div>
        <div class="mb-3">
            <label for="trade-value" class="form-label">Value (USDT)</label>
            <input type="number" class="form-control" id="trade-value" readonly>
        </div>
        <button type="submit" class="btn btn-primary">Execute ${type.toUpperCase()}</button>
    `;
    
    // Show modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    // Set up form submission
    tradeForm.onsubmit = function(e) {
        e.preventDefault();
        executeTrade(type, symbol);
    };
    
    // Set up amount input handler
    const amountInput = document.getElementById('trade-amount');
    amountInput.oninput = function() {
        updateTradePreview(type, symbol);
    };
}

// Update trade preview
function updateTradePreview(type, symbol) {
    const amount = parseFloat(document.getElementById('trade-amount').value) || 0;
    const price = getCurrentPrice(symbol);
    const value = type === 'buy' ? amount : amount * price;
    
    document.getElementById('trade-price').value = price.toFixed(2);
    document.getElementById('trade-value').value = value.toFixed(2);
}

// Execute trade
async function executeTrade(type, symbol) {
    const amount = parseFloat(document.getElementById('trade-amount').value);
    const value = parseFloat(document.getElementById('trade-value').value);
    
    // Validate minimum trade amount
    if (value < 5) {
        showError('Minimum trade amount is $5.00');
        return;
    }
    
    try {
        const response = await fetch('/api/trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                agent_id: currentAgentId,
                type: type,
                symbol: symbol,
                amount: amount,
                value: value
            })
        });
        
        if (!response.ok) {
            throw new Error('Trade execution failed');
        }
        
        const result = await response.json();
        if (result.success) {
            showSuccess(`Trade executed successfully: ${type.toUpperCase()} ${amount} ${symbol}`);
            // Refresh data
            loadPortfolioData();
            loadTradeHistory();
            loadPerformanceMetrics();
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('trade-modal')).hide();
        } else {
            showError(result.message || 'Trade execution failed');
        }
    } catch (error) {
        console.error('Error executing trade:', error);
        showError('Failed to execute trade');
    }
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 8
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