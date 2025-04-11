// Dashboard JavaScript

// Document ready handler
$(document).ready(function() {
    // Initialize the dashboard
    initDashboard();
    
    // Set up event listeners
    setupEventListeners();
    
    // Fetch initial data
    fetchPortfolioData();
    fetchPriceData();
    
    // Set up refresh interval (every 30 seconds)
    setInterval(function() {
        fetchPortfolioData();
        fetchPriceData();
    }, 30000);
});

// Initialize dashboard components
function initDashboard() {
    // Initialize charts with empty data
    initPortfolioValueChart();
    initAssetAllocationChart();
    
    // Initialize trader dropdown
    populateTraderDropdown();
}

// Set up event listeners
function setupEventListeners() {
    // Refresh button click
    $('.refresh-btn').on('click', function() {
        $(this).addClass('fa-spin');
        fetchPortfolioData().then(() => {
            setTimeout(() => $(this).removeClass('fa-spin'), 500);
        });
    });
    
    // Trader dropdown change
    $('#trader-select').on('change', function() {
        fetchPortfolioData();
    });
    
    // Modals
    setupTradeModal();
}

// Populate trader dropdown
function populateTraderDropdown() {
    $.ajax({
        url: '/api/traders',
        method: 'GET',
        success: function(response) {
            const select = $('#trader-select');
            select.empty();
            
            response.traders.forEach(trader => {
                select.append(
                    $('<option>', {
                        value: trader.id,
                        text: trader.name
                    })
                );
            });
            
            // Trigger change to load initial data
            select.trigger('change');
        },
        error: function(error) {
            console.error('Error fetching traders:', error);
            showErrorMessage('Failed to load traders');
        }
    });
}

// Fetch portfolio data
function fetchPortfolioData() {
    const traderId = $('#trader-select').val();
    
    if (!traderId) return Promise.resolve();
    
    return $.ajax({
        url: `/api/portfolio/${traderId}`,
        method: 'GET',
        beforeSend: function() {
            $('#portfolio-loading').show();
        },
        success: function(data) {
            updatePortfolioSummary(data.summary);
            updatePortfolioTable(data.holdings);
            updatePortfolioValueChart(data.history);
            updateAssetAllocationChart(data.holdings);
            $('#portfolio-loading').hide();
        },
        error: function(error) {
            console.error('Error fetching portfolio data:', error);
            showErrorMessage('Failed to load portfolio data');
            $('#portfolio-loading').hide();
        }
    });
}

// Fetch price data
function fetchPriceData() {
    $.ajax({
        url: '/api/prices',
        method: 'GET',
        success: function(data) {
            updatePriceData(data);
        },
        error: function(error) {
            console.error('Error fetching price data:', error);
        }
    });
}

// Update portfolio summary cards
function updatePortfolioSummary(summary) {
    $('#total-value').text('$' + summary.totalValue.toFixed(2));
    $('#base-currency').text('$' + summary.baseCurrency.toFixed(2));
    $('#assets-value').text('$' + summary.assetsValue.toFixed(2));
    $('#asset-count').text(summary.assetCount);
}

// Update portfolio table
function updatePortfolioTable(holdings) {
    const tableBody = $('#portfolio-table tbody');
    tableBody.empty();
    
    holdings.forEach(asset => {
        const row = $('<tr>');
        const priceChangeClass = asset.priceChange >= 0 ? 'price-up' : 'price-down';
        const priceChangeIcon = asset.priceChange >= 0 ? 'fa-caret-up' : 'fa-caret-down';
        
        row.append(`<td><img src="/static/img/crypto/${asset.symbol.toLowerCase()}.png" width="24" class="me-2" onerror="this.src='/static/img/crypto/generic.png'">${asset.symbol}</td>`);
        row.append(`<td>${asset.quantity.toFixed(8)}</td>`);
        row.append(`<td>$${asset.price.toFixed(2)}</td>`);
        row.append(`<td class="${priceChangeClass}"><i class="fas ${priceChangeIcon}"></i> ${Math.abs(asset.priceChange).toFixed(2)}%</td>`);
        row.append(`<td>$${asset.value.toFixed(2)}</td>`);
        row.append(`<td class="actions-col">
            <button class="btn btn-sm btn-buy" data-symbol="${asset.symbol}" data-action="buy">Buy</button>
            <button class="btn btn-sm btn-sell" data-symbol="${asset.symbol}" data-action="sell" ${asset.quantity <= 0 ? 'disabled' : ''}>Sell</button>
        </td>`);
        
        tableBody.append(row);
    });
    
    // Set up trade buttons
    setupTradeButtons();
}

// Set up trade buttons
function setupTradeButtons() {
    $('.btn-buy, .btn-sell').off('click').on('click', function() {
        const action = $(this).data('action');
        const symbol = $(this).data('symbol');
        showTradeModal(symbol, action);
    });
}

// Update price data in the UI
function updatePriceData(prices) {
    // Update price data in portfolio table
    prices.forEach(price => {
        const priceCell = $(`#portfolio-table td:contains('${price.symbol}')`).siblings().eq(1);
        if (priceCell.length) {
            priceCell.text('$' + price.price.toFixed(2));
        }
    });
}

// Initialize portfolio value chart
function initPortfolioValueChart() {
    const ctx = document.getElementById('portfolio-value-chart').getContext('2d');
    window.portfolioValueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#4e73df',
                backgroundColor: 'rgba(78, 115, 223, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Update portfolio value chart with new data
function updatePortfolioValueChart(history) {
    window.portfolioValueChart.data.labels = history.map(entry => entry.date);
    window.portfolioValueChart.data.datasets[0].data = history.map(entry => entry.value);
    window.portfolioValueChart.update();
}

// Initialize asset allocation chart
function initAssetAllocationChart() {
    const ctx = document.getElementById('asset-allocation-chart').getContext('2d');
    window.assetAllocationChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b',
                    '#5a5c69', '#858796', '#2ecc71', '#3498db', '#e67e22'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((acc, val) => acc + val, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${context.label}: $${value.toFixed(2)} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Update asset allocation chart with new data
function updateAssetAllocationChart(holdings) {
    const labels = [];
    const data = [];
    
    holdings.forEach(asset => {
        if (asset.value > 0) {
            labels.push(asset.symbol);
            data.push(asset.value);
        }
    });
    
    window.assetAllocationChart.data.labels = labels;
    window.assetAllocationChart.data.datasets[0].data = data;
    window.assetAllocationChart.update();
}

// Set up trade modal
function setupTradeModal() {
    // Trade form submission
    $('#trade-form').on('submit', function(e) {
        e.preventDefault();
        executeTrade();
    });
}

// Show trade modal
function showTradeModal(symbol, action) {
    const modal = $('#trade-modal');
    const title = $('#trade-modal-title');
    const actionInput = $('#trade-action');
    const symbolInput = $('#trade-symbol');
    const amountInput = $('#trade-amount');
    const submitBtn = $('#trade-submit');
    
    // Set modal values
    title.text(`${action.charAt(0).toUpperCase() + action.slice(1)} ${symbol}`);
    actionInput.val(action);
    symbolInput.val(symbol);
    amountInput.val('').focus();
    
    // Set button styles based on action
    submitBtn.removeClass('btn-success btn-danger')
        .addClass(action === 'buy' ? 'btn-success' : 'btn-danger')
        .text(action.charAt(0).toUpperCase() + action.slice(1));
    
    // Get current price
    const currentPrice = parseFloat($(`#portfolio-table td:contains('${symbol}')`).siblings().eq(1).text().replace('$', ''));
    $('#trade-price').text(`Current price: $${currentPrice.toFixed(2)}`);
    
    // Show the modal
    modal.modal('show');
}

// Execute trade
function executeTrade() {
    const action = $('#trade-action').val();
    const symbol = $('#trade-symbol').val();
    const amount = parseFloat($('#trade-amount').val());
    const traderId = $('#trader-select').val();
    
    // Validate amount (minimum $5)
    if (isNaN(amount) || amount < 5) {
        showErrorMessage('Minimum trade amount is $5');
        return;
    }
    
    // Disable submit button
    const submitBtn = $('#trade-submit');
    submitBtn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
    
    // Execute trade via API
    $.ajax({
        url: `/api/${action}`,
        method: 'POST',
        data: JSON.stringify({
            trader_id: traderId,
            symbol: symbol,
            amount: amount
        }),
        contentType: 'application/json',
        success: function(response) {
            // Show success message
            showSuccessMessage(`Successfully ${action === 'buy' ? 'bought' : 'sold'} ${symbol}`);
            
            // Close modal
            $('#trade-modal').modal('hide');
            
            // Refresh portfolio data
            fetchPortfolioData();
        },
        error: function(error) {
            console.error(`Error ${action}ing ${symbol}:`, error);
            showErrorMessage(error.responseJSON?.message || `Failed to ${action} ${symbol}`);
        },
        complete: function() {
            // Re-enable submit button
            submitBtn.prop('disabled', false).text(action.charAt(0).toUpperCase() + action.slice(1));
        }
    });
}

// Show success message
function showSuccessMessage(message) {
    const alertDiv = $('<div class="alert alert-success alert-dismissible fade show" role="alert">')
        .text(message)
        .append('<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>');
    
    $('#alert-container').append(alertDiv);
    
    // Auto dismiss after 3 seconds
    setTimeout(() => {
        alertDiv.alert('close');
    }, 3000);
}

// Show error message
function showErrorMessage(message) {
    const alertDiv = $('<div class="alert alert-danger alert-dismissible fade show" role="alert">')
        .text(message)
        .append('<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>');
    
    $('#alert-container').append(alertDiv);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.alert('close');
    }, 5000);
}

// Load market prices and create charts
async function loadMarketPrices() {
    try {
        const response = await fetch('/api/prices');
        if (!response.ok) throw new Error('Failed to fetch market prices');
        
        const data = await response.json();
        
        const marketPricesContainer = document.getElementById('marketPrices');
        marketPricesContainer.innerHTML = '';
        
        for (const [symbol, price] of Object.entries(data)) {
            const priceCard = document.createElement('div');
            priceCard.className = 'price-card';
            
            const changeClass = price.change_24h >= 0 ? 'positive' : 'negative';
            const changeSymbol = price.change_24h >= 0 ? '+' : '';
            
            // Create chart container
            const chartContainer = document.createElement('div');
            chartContainer.className = 'chart-container';
            chartContainer.id = `chart-${symbol.replace('/', '')}`;
            
            priceCard.innerHTML = `
                <div class="symbol">${symbol}</div>
                <div class="price">$${price.price.toFixed(2)}</div>
                <div class="change ${changeClass}">${changeSymbol}${price.change_24h.toFixed(2)}%</div>
            `;
            
            priceCard.appendChild(chartContainer);
            marketPricesContainer.appendChild(priceCard);
            
            // Load historical data and create chart
            loadPriceChart(symbol, chartContainer.id);
        }
    } catch (error) {
        console.error('Error loading market prices:', error);
        showErrorMessage('Failed to load market prices');
    }
}

// Load historical price data and create chart
async function loadPriceChart(symbol, containerId) {
    try {
        // Convert symbol for URL (replace / with -)
        const urlSymbol = symbol.replace('/', '-');
        const response = await fetch(`/api/historical-prices/${urlSymbol}?interval=1h`);
        if (!response.ok) throw new Error('Failed to fetch historical prices');
        
        const data = await response.json();
        
        // Create chart configuration
        const chartConfig = {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
                datasets: [{
                    label: symbol,
                    data: data.map(d => d.price),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: document.body.classList.contains('dark-mode') ? '#fff' : '#2c3e50'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: document.body.classList.contains('dark-mode') ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            color: document.body.classList.contains('dark-mode') ? '#fff' : '#2c3e50'
                        }
                    },
                    y: {
                        grid: {
                            color: document.body.classList.contains('dark-mode') ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            color: document.body.classList.contains('dark-mode') ? '#fff' : '#2c3e50',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        };
        
        const ctx = document.getElementById(containerId).getContext('2d');
        new Chart(ctx, chartConfig);
    } catch (error) {
        console.error(`Error loading chart for ${symbol}:`, error);
    }
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

// Create alert container if it doesn't exist
function createAlertContainer() {
    const container = document.createElement('div');
    container.id = 'alertContainer';
    document.body.appendChild(container);
    return container;
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadMarketPrices();
    loadTraderPortfolios();
    
    // Refresh data every 30 seconds
    setInterval(() => {
        loadMarketPrices();
        loadTraderPortfolios();
    }, 30000);
}); 