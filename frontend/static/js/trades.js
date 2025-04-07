// Crypto Trader Trade History JavaScript

// Socket.io connection
const socket = io();

// Trade data cache
let tradesData = [];

// Current filter settings
let filterSettings = {
    dateRange: 'all',
    tradeType: 'all',
    asset: 'all',
    exchange: 'all',
    sortOrder: 'newest',
    searchQuery: '',
    currentPage: 1,
    itemsPerPage: 10
};

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupEventListeners();
    
    // Fetch initial data
    fetchTradeData();
    
    // Connect system status indicators
    connectSystemControls();
});

// Set up event listeners for the page
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', fetchTradeData);
    }
    
    // Filter form
    const filterForm = document.getElementById('filter-form');
    if (filterForm) {
        filterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            applyFilters();
        });
        
        // Reset button
        filterForm.addEventListener('reset', function() {
            // Reset filter settings to defaults
            filterSettings = {
                dateRange: 'all',
                tradeType: 'all',
                asset: 'all',
                exchange: 'all',
                sortOrder: 'newest',
                searchQuery: '',
                currentPage: 1,
                itemsPerPage: 10
            };
            
            // Small timeout to let the form reset before applying
            setTimeout(() => {
                applyFilters();
            }, 10);
        });
    }
    
    // Search input
    const searchInput = document.querySelector('input[aria-label="Search"]');
    if (searchInput) {
        searchInput.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') {
                filterSettings.searchQuery = this.value.trim();
                filterSettings.currentPage = 1;
                applyFilters();
            }
        });
        
        // Search button
        const searchButton = searchInput.nextElementSibling;
        if (searchButton) {
            searchButton.addEventListener('click', function() {
                filterSettings.searchQuery = searchInput.value.trim();
                filterSettings.currentPage = 1;
                applyFilters();
            });
        }
    }
    
    // Sorting options
    const sortNewestLink = document.querySelector('a[href="#"][title*="Newest"]');
    if (sortNewestLink) {
        sortNewestLink.addEventListener('click', function(e) {
            e.preventDefault();
            filterSettings.sortOrder = 'newest';
            applyFilters();
        });
    }
    
    const sortOldestLink = document.querySelector('a[href="#"][title*="Oldest"]');
    if (sortOldestLink) {
        sortOldestLink.addEventListener('click', function(e) {
            e.preventDefault();
            filterSettings.sortOrder = 'oldest';
            applyFilters();
        });
    }
    
    // Pagination
    setupPaginationListeners();
    
    // Trade detail view buttons
    setupTradeDetailButtons();
    
    // Socket.io event listeners
    socket.on('system_status', function(data) {
        updateSystemStatus(data);
    });
}

// Fetch trade data from API
function fetchTradeData() {
    // Show loading state
    document.getElementById('refresh-btn').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    
    // Get trades data
    fetch('/api/trades')
        .then(response => response.json())
        .then(data => {
            if (data.trades) {
                tradesData = data.trades;
                applyFilters();
            }
        })
        .catch(error => {
            console.error('Error fetching trades:', error);
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

// Apply filters to trade data
function applyFilters() {
    let filteredTrades = tradesData.slice();
    
    // Apply date range filter
    if (filterSettings.dateRange !== 'all') {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        
        switch (filterSettings.dateRange) {
            case 'today':
                filteredTrades = filteredTrades.filter(trade => 
                    new Date(trade.timestamp) >= today);
                break;
            case 'yesterday':
                const yesterday = new Date(today);
                yesterday.setDate(today.getDate() - 1);
                filteredTrades = filteredTrades.filter(trade => 
                    new Date(trade.timestamp) >= yesterday && 
                    new Date(trade.timestamp) < today);
                break;
            case 'this_week':
                const startOfWeek = new Date(today);
                startOfWeek.setDate(today.getDate() - today.getDay());
                filteredTrades = filteredTrades.filter(trade => 
                    new Date(trade.timestamp) >= startOfWeek);
                break;
            case 'this_month':
                const startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
                filteredTrades = filteredTrades.filter(trade => 
                    new Date(trade.timestamp) >= startOfMonth);
                break;
            // Custom range would need start/end date pickers
        }
    }
    
    // Apply trade type filter
    if (filterSettings.tradeType !== 'all') {
        filteredTrades = filteredTrades.filter(trade => 
            trade.trade_type === filterSettings.tradeType);
    }
    
    // Apply asset filter
    if (filterSettings.asset !== 'all') {
        filteredTrades = filteredTrades.filter(trade => 
            trade.from_currency === filterSettings.asset || 
            trade.to_currency === filterSettings.asset);
    }
    
    // Apply exchange filter
    if (filterSettings.exchange !== 'all') {
        filteredTrades = filteredTrades.filter(trade => 
            trade.exchange === filterSettings.exchange);
    }
    
    // Apply search query
    if (filterSettings.searchQuery) {
        const query = filterSettings.searchQuery.toLowerCase();
        filteredTrades = filteredTrades.filter(trade =>
            trade.id.toString().includes(query) ||
            trade.from_currency.toLowerCase().includes(query) ||
            trade.to_currency.toLowerCase().includes(query) ||
            trade.exchange.toLowerCase().includes(query)
        );
    }
    
    // Apply sorting
    filteredTrades.sort((a, b) => {
        const dateA = new Date(a.timestamp);
        const dateB = new Date(b.timestamp);
        
        if (filterSettings.sortOrder === 'newest') {
            return dateB - dateA;
        } else {
            return dateA - dateB;
        }
    });
    
    // Update trade count statistics
    updateTradeStatistics(filteredTrades);
    
    // Apply pagination
    const totalPages = Math.ceil(filteredTrades.length / filterSettings.itemsPerPage);
    // Adjust current page if needed
    if (filterSettings.currentPage > totalPages) {
        filterSettings.currentPage = Math.max(1, totalPages);
    }
    
    const startIndex = (filterSettings.currentPage - 1) * filterSettings.itemsPerPage;
    const endIndex = startIndex + filterSettings.itemsPerPage;
    const paginatedTrades = filteredTrades.slice(startIndex, endIndex);
    
    // Update the trades table
    updateTradesTable(paginatedTrades);
    
    // Update pagination controls
    updatePagination(filteredTrades.length, totalPages);
}

// Update the trades table with filtered data
function updateTradesTable(trades) {
    const tradesTable = document.getElementById('trades-table');
    if (!tradesTable) return;
    
    let tableContent = '';
    
    if (trades.length === 0) {
        // No trades found
        tableContent = `
            <tr>
                <td colspan="10" class="text-center">No trades found matching your filters.</td>
            </tr>
        `;
    } else {
        // Add rows for each trade
        trades.forEach(trade => {
            const tradeType = trade.trade_type;
            const tradeClass = tradeType === 'buy' ? 'table-success' : 'table-danger';
            const badgeClass = tradeType === 'buy' ? 'bg-success' : 'bg-danger';
            
            const tradeDate = new Date(trade.timestamp);
            const formattedDate = tradeDate.toLocaleString();
            
            // Format amount based on trade type
            let amount, pair;
            if (tradeType === 'buy') {
                amount = `${trade.to_amount.toFixed(5)} ${trade.to_currency}`;
                pair = `${trade.from_currency}/${trade.to_currency}`;
            } else {
                amount = `${trade.from_amount.toFixed(5)} ${trade.from_currency}`;
                pair = `${trade.from_currency}/${trade.to_currency}`;
            }
            
            // Format total based on trade type
            const total = tradeType === 'buy' 
                ? `$${trade.from_amount.toFixed(2)}`
                : `$${trade.to_amount.toFixed(2)}`;
            
            tableContent += `
                <tr class="${tradeClass}">
                    <td>${trade.id}</td>
                    <td>${formattedDate}</td>
                    <td><span class="badge rounded-pill ${badgeClass}">${tradeType.charAt(0).toUpperCase() + tradeType.slice(1)}</span></td>
                    <td>${pair}</td>
                    <td>${amount}</td>
                    <td>$${trade.price.toFixed(2)}</td>
                    <td>${total}</td>
                    <td>$${trade.fee.toFixed(4)}</td>
                    <td>${trade.exchange}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary view-trade-details" title="View Details" data-trade-id="${trade.id}">
                            <i class="bi bi-info-circle"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
    }
    
    // Update table content
    tradesTable.innerHTML = tableContent;
    
    // Re-attach event listeners to new detail buttons
    setupTradeDetailButtons();
}

// Update trade statistics
function updateTradeStatistics(trades) {
    const totalTrades = document.querySelector('.card-body .h5[id^="total-trades"]');
    if (totalTrades) {
        totalTrades.textContent = trades.length;
    }
    
    const buyTrades = document.querySelector('.card-body .h5[id^="buy-trades"]');
    if (buyTrades) {
        const buyCount = trades.filter(trade => trade.trade_type === 'buy').length;
        buyTrades.textContent = buyCount;
    }
    
    const sellTrades = document.querySelector('.card-body .h5[id^="sell-trades"]');
    if (sellTrades) {
        const sellCount = trades.filter(trade => trade.trade_type === 'sell').length;
        sellTrades.textContent = sellCount;
    }
    
    const totalFees = document.querySelector('.card-body .h5[id^="total-fees"]');
    if (totalFees) {
        const feesSum = trades.reduce((sum, trade) => sum + trade.fee, 0);
        totalFees.textContent = '$' + feesSum.toFixed(2);
    }
}

// Update pagination controls
function updatePagination(totalItems, totalPages) {
    const paginationUl = document.querySelector('.pagination');
    if (!paginationUl) return;
    
    const currentPage = filterSettings.currentPage;
    
    let paginationHTML = '';
    
    // Previous button
    paginationHTML += `
        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" data-page="${currentPage - 1}" tabindex="-1" ${currentPage === 1 ? 'aria-disabled="true"' : ''}>Previous</a>
        </li>
    `;
    
    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    for (let i = startPage; i <= endPage; i++) {
        paginationHTML += `
            <li class="page-item ${i === currentPage ? 'active' : ''}">
                <a class="page-link" href="#" data-page="${i}">${i}</a>
            </li>
        `;
    }
    
    // Next button
    paginationHTML += `
        <li class="page-item ${currentPage === totalPages || totalPages === 0 ? 'disabled' : ''}">
            <a class="page-link" href="#" data-page="${currentPage + 1}" ${currentPage === totalPages || totalPages === 0 ? 'aria-disabled="true"' : ''}>Next</a>
        </li>
    `;
    
    paginationUl.innerHTML = paginationHTML;
    
    // Set up pagination click handlers
    setupPaginationListeners();
}

// Set up pagination listeners
function setupPaginationListeners() {
    const paginationLinks = document.querySelectorAll('.pagination .page-link');
    paginationLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (this.parentElement.classList.contains('disabled')) {
                return;
            }
            
            const page = parseInt(this.getAttribute('data-page'));
            filterSettings.currentPage = page;
            applyFilters();
            
            // Scroll to top of table
            const tradesTable = document.querySelector('.table-responsive');
            if (tradesTable) {
                tradesTable.scrollTop = 0;
            }
        });
    });
}

// Set up trade detail buttons
function setupTradeDetailButtons() {
    const detailButtons = document.querySelectorAll('.view-trade-details');
    detailButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tradeId = this.getAttribute('data-trade-id');
            showTradeDetails(tradeId);
        });
    });
}

// Show trade details in modal
function showTradeDetails(tradeId) {
    const trade = tradesData.find(t => t.id.toString() === tradeId.toString());
    
    if (!trade) {
        console.error('Trade not found:', tradeId);
        return;
    }
    
    // Populate modal with trade details
    document.getElementById('modal-trade-id').textContent = trade.id;
    document.getElementById('modal-trade-time').textContent = new Date(trade.timestamp).toLocaleString();
    
    const tradeType = trade.trade_type.charAt(0).toUpperCase() + trade.trade_type.slice(1);
    document.getElementById('modal-trade-type').textContent = tradeType;
    document.getElementById('modal-trade-type').className = trade.trade_type === 'buy' ? 'text-success' : 'text-danger';
    
    document.getElementById('modal-trade-pair').textContent = `${trade.from_currency}/${trade.to_currency}`;
    document.getElementById('modal-trade-price').textContent = `$${trade.price.toFixed(2)}`;
    
    // Format amount based on trade type
    let amount;
    if (trade.trade_type === 'buy') {
        amount = `${trade.to_amount.toFixed(6)} ${trade.to_currency}`;
    } else {
        amount = `${trade.from_amount.toFixed(6)} ${trade.from_currency}`;
    }
    document.getElementById('modal-trade-amount').textContent = amount;
    
    // Format total based on trade type
    const total = trade.trade_type === 'buy' 
        ? `$${trade.from_amount.toFixed(2)}`
        : `$${trade.to_amount.toFixed(2)}`;
    document.getElementById('modal-trade-total').textContent = total;
    
    document.getElementById('modal-trade-fee').textContent = `$${trade.fee.toFixed(4)}`;
    document.getElementById('modal-trade-exchange').textContent = trade.exchange;
    
    // Set agent if available
    const agentElement = document.getElementById('modal-trade-agent');
    if (agentElement) {
        agentElement.textContent = trade.agent || 'System';
    }
    
    // Set notes if available
    const notesElement = document.getElementById('modal-trade-notes');
    if (notesElement) {
        notesElement.textContent = trade.notes || 'No notes available for this trade.';
    }
    
    // Show the modal
    const tradeDetailsModal = new bootstrap.Modal(document.getElementById('tradeDetailsModal'));
    tradeDetailsModal.show();
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