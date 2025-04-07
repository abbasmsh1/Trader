// Crypto Trader Portfolio JavaScript

// Socket.io connection
const socket = io();

// Price data cache
let priceData = {};

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupEventListeners();
    
    // Fetch initial data
    fetchPortfolioData();
    
    // Connect system status indicators
    connectSystemControls();
});

// Set up event listeners for the page
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', fetchPortfolioData);
    }
    
    // Sell buttons
    const sellButtons = document.querySelectorAll('.sell-btn');
    sellButtons.forEach(button => {
        button.addEventListener('click', function() {
            const asset = this.getAttribute('data-asset');
            openSellModal(asset);
        });
    });
    
    // Buy button
    const buyButtons = document.querySelectorAll('.buy-btn');
    buyButtons.forEach(button => {
        button.addEventListener('click', function() {
            window.location.href = '/#demo-buy-form';
        });
    });
    
    // Modal sell all link
    const sellAllLink = document.getElementById('sell-all-link');
    if (sellAllLink) {
        sellAllLink.addEventListener('click', function(e) {
            e.preventDefault();
            const balance = parseFloat(document.getElementById('modal-asset-balance').textContent);
            document.getElementById('modal-sell-amount').value = balance;
            updateSellValue();
        });
    }
    
    // Modal sell amount input
    const modalSellAmount = document.getElementById('modal-sell-amount');
    if (modalSellAmount) {
        modalSellAmount.addEventListener('input', updateSellValue);
    }
    
    // Modal confirm sell button
    const confirmSellBtn = document.getElementById('confirm-sell-btn');
    if (confirmSellBtn) {
        confirmSellBtn.addEventListener('click', executeSell);
    }
    
    // Socket.io event listeners
    socket.on('portfolio_update', function(data) {
        updatePortfolioUI(data);
    });
    
    socket.on('price_update', function(data) {
        priceData = data;
        updatePrices();
    });
    
    socket.on('system_status', function(data) {
        updateSystemStatus(data);
    });
}

// Fetch portfolio data from API
function fetchPortfolioData() {
    // Show loading state
    document.getElementById('refresh-btn').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    
    // Get portfolio data
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            updatePortfolioUI(data);
            
            // Get prices data
            return fetch('/api/prices');
        })
        .then(response => response.json())
        .then(data => {
            priceData = data;
            updatePrices();
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        })
        .finally(() => {
            // Restore refresh button
            document.getElementById('refresh-btn').innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
        });
    
    // Get system status
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => console.error('Error fetching system status:', error));
}

// Update portfolio UI with data
function updatePortfolioUI(data) {
    // Update total value
    const totalValue = document.getElementById('total-value');
    if (totalValue) {
        totalValue.textContent = (data.total_value || 0).toFixed(2);
    }
    
    // Update base currency
    const baseCurrency = document.getElementById('base-currency');
    if (baseCurrency && data.balances) {
        baseCurrency.textContent = (data.balances.USDT || 0).toFixed(2);
    }
    
    // Update assets value
    const assetsValue = document.getElementById('assets-value');
    if (assetsValue && data.balances) {
        const baseBalance = data.balances.USDT || 0;
        const assetVal = (data.total_value || 0) - baseBalance;
        assetsValue.textContent = assetVal.toFixed(2);
    }
    
    // Update number of assets
    const numAssets = document.getElementById('num-assets');
    if (numAssets && data.holdings) {
        const assetCount = Object.keys(data.holdings).filter(key => key !== 'USDT' && data.holdings[key].value > 0).length;
        numAssets.textContent = assetCount;
    }
    
    // Update portfolio table
    updatePortfolioTable(data);
}

// Update the portfolio table
function updatePortfolioTable(data) {
    const portfolioTable = document.getElementById('portfolio-table');
    if (!portfolioTable) return;
    
    let tableContent = '';
    
    // Add rows for crypto assets
    if (data.holdings) {
        // Sort holdings by value (descending)
        const sortedAssets = Object.keys(data.holdings)
            .filter(asset => asset !== 'USDT' && data.holdings[asset].value > 0)
            .sort((a, b) => data.holdings[b].value - data.holdings[a].value);
        
        // Add rows for each asset
        for (const asset of sortedAssets) {
            const holding = data.holdings[asset];
            const percentOfPortfolio = (holding.value / data.total_value * 100).toFixed(1);
            
            tableContent += `
                <tr>
                    <td><strong>${asset}</strong></td>
                    <td>${holding.amount.toFixed(6)}</td>
                    <td>$${holding.price.toFixed(2)}</td>
                    <td>$${holding.value.toFixed(2)}</td>
                    <td>${percentOfPortfolio}%</td>
                    <td>
                        <button class="btn btn-sm btn-danger sell-btn" data-asset="${asset}">Sell</button>
                    </td>
                </tr>
            `;
        }
    }
    
    // Add row for base currency (USDT)
    if (data.balances && data.balances.USDT) {
        const usdtBalance = data.balances.USDT;
        const percentOfPortfolio = (usdtBalance / data.total_value * 100).toFixed(1);
        
        tableContent += `
            <tr>
                <td><strong>USDT</strong></td>
                <td>${usdtBalance.toFixed(2)}</td>
                <td>$1.00</td>
                <td>$${usdtBalance.toFixed(2)}</td>
                <td>${percentOfPortfolio}%</td>
                <td>
                    <button class="btn btn-sm btn-success buy-btn">Buy</button>
                </td>
            </tr>
        `;
    }
    
    // Update table content
    portfolioTable.innerHTML = tableContent;
    
    // Re-attach event listeners to new buttons
    const sellButtons = document.querySelectorAll('.sell-btn');
    sellButtons.forEach(button => {
        button.addEventListener('click', function() {
            const asset = this.getAttribute('data-asset');
            openSellModal(asset);
        });
    });
    
    const buyButtons = document.querySelectorAll('.buy-btn');
    buyButtons.forEach(button => {
        button.addEventListener('click', function() {
            window.location.href = '/#demo-buy-form';
        });
    });
}

// Update price information
function updatePrices() {
    // If we have a modal open, update its values
    updateSellValue();
}

// Open the sell modal for an asset
function openSellModal(asset) {
    const sellModal = new bootstrap.Modal(document.getElementById('sellModal'));
    
    // Fetch current balance for the asset
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            if (data.holdings && data.holdings[asset]) {
                // Set modal fields
                document.getElementById('modal-sell-asset').value = asset;
                document.getElementById('modal-asset-symbol').textContent = asset;
                document.getElementById('modal-asset-balance').textContent = data.holdings[asset].amount.toFixed(6);
                
                // Clear amount field
                document.getElementById('modal-sell-amount').value = '';
                document.getElementById('modal-sell-value').value = '';
                
                // Show the modal
                sellModal.show();
            } else {
                alert(`No balance found for ${asset}`);
            }
        })
        .catch(error => {
            console.error('Error fetching asset balance:', error);
            alert('Could not fetch asset balance. Please try again.');
        });
}

// Update the sell value based on amount
function updateSellValue() {
    const asset = document.getElementById('modal-sell-asset').value;
    const amountInput = document.getElementById('modal-sell-amount');
    const valueInput = document.getElementById('modal-sell-value');
    
    if (!asset || !amountInput || !valueInput) return;
    
    const amount = parseFloat(amountInput.value) || 0;
    
    // Get current price from price data
    if (priceData && priceData[asset]) {
        const price = priceData[asset];
        const value = amount * price;
        valueInput.value = value.toFixed(2);
    } else {
        // If price not available, fetch from API
        fetch('/api/prices')
            .then(response => response.json())
            .then(data => {
                priceData = data;
                if (data[asset]) {
                    const price = data[asset];
                    const value = amount * price;
                    valueInput.value = value.toFixed(2);
                }
            })
            .catch(error => console.error('Error fetching prices:', error));
    }
}

// Execute sell order
function executeSell() {
    const asset = document.getElementById('modal-sell-asset').value;
    const amountInput = document.getElementById('modal-sell-amount');
    const amount = parseFloat(amountInput.value);
    
    if (!asset || isNaN(amount) || amount <= 0) {
        alert('Please enter a valid amount to sell.');
        return;
    }
    
    // Disable button during request
    const confirmButton = document.getElementById('confirm-sell-btn');
    confirmButton.disabled = true;
    confirmButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
    
    // Send sell request
    fetch('/api/demo/sell', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            symbol: asset,
            quantity: amount
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Close modal
            const sellModal = bootstrap.Modal.getInstance(document.getElementById('sellModal'));
            sellModal.hide();
            
            // Show success message
            alert(`Successfully sold ${data.quantity.toFixed(6)} ${data.symbol} for $${data.amount.toFixed(2)}`);
            
            // Refresh portfolio data
            fetchPortfolioData();
        } else {
            // Show error
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error executing sell:', error);
        alert('An error occurred while executing the sell order. Please try again.');
    })
    .finally(() => {
        // Restore button
        confirmButton.disabled = false;
        confirmButton.innerHTML = 'Sell';
    });
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