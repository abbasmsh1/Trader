// Trade History JavaScript

// Document ready handler
$(document).ready(function() {
    loadTraders();
    loadHistory();
    loadSignals();
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    // History filters
    $('#trader-select, #date-range, #action-filter, #symbol-filter, #min-amount').on('change', function() {
        loadHistory();
    });
    
    $('#refresh-history').on('click', function() {
        loadHistory();
    });
    
    // Signal filters
    $('#signal-trader-select, #signal-date-range, #signal-status').on('change', function() {
        loadSignals();
    });
    
    $('#refresh-signals').on('click', function() {
        loadSignals();
    });
    
    // Pagination
    $('#prev-page, #next-page').on('click', function() {
        const currentPage = parseInt($('#page-info').text().split(' ')[1]);
        const newPage = $(this).attr('id') === 'prev-page' ? currentPage - 1 : currentPage + 1;
        loadHistory(newPage);
    });
    
    $('#signal-prev-page, #signal-next-page').on('click', function() {
        const currentPage = parseInt($('#signal-page-info').text().split(' ')[1]);
        const newPage = $(this).attr('id') === 'signal-prev-page' ? currentPage - 1 : currentPage + 1;
        loadSignals(newPage);
    });
}

// Load traders into dropdowns
async function loadTraders() {
    try {
        const response = await fetch('/api/traders');
        if (!response.ok) throw new Error('Failed to fetch traders');
        
        const traders = await response.json();
        
        // Populate trader selects
        const traderSelects = ['#trader-select', '#signal-trader-select'];
        traderSelects.forEach(select => {
            const $select = $(select);
            $select.empty();
            $select.append('<option value="all">All Traders</option>');
            
            traders.forEach(trader => {
                $select.append(`<option value="${trader.id}">${trader.name}</option>`);
            });
        });
        
    } catch (error) {
        console.error('Error loading traders:', error);
    }
}

// Load trade history
async function loadHistory(page = 1) {
    try {
        const traderId = $('#trader-select').val();
        const dateRange = $('#date-range').val();
        const action = $('#action-filter').val();
        const symbol = $('#symbol-filter').val();
        const minAmount = $('#min-amount').val();
        
        const response = await fetch(`/api/history/${traderId}?page=${page}&date_range=${dateRange}&action=${action}&symbol=${symbol}&min_amount=${minAmount}`);
        if (!response.ok) throw new Error('Failed to fetch history');
        
        const data = await response.json();
        
        // Update pagination
        updatePagination(data.pagination, 'history');
        
        // Display history
        const historyGrid = $('.history-grid');
        historyGrid.empty();
        
        if (data.trades.length === 0) {
            historyGrid.html('<div class="no-data">No trades found</div>');
            return;
        }
        
        data.trades.forEach(trade => {
            const tradeCard = createTradeCard(trade);
            historyGrid.append(tradeCard);
        });
        
    } catch (error) {
        console.error('Error loading history:', error);
        $('.history-grid').html('<div class="error">Error loading trade history</div>');
    }
}

// Load signals
async function loadSignals(page = 1) {
    try {
        const traderId = $('#signal-trader-select').val();
        const dateRange = $('#signal-date-range').val();
        const status = $('#signal-status').val();
        
        const response = await fetch(`/api/signals/${traderId}?page=${page}&date_range=${dateRange}&status=${status}`);
        if (!response.ok) throw new Error('Failed to fetch signals');
        
        const data = await response.json();
        
        // Update pagination
        updatePagination(data.pagination, 'signal');
        
        // Display signals
        const signalsGrid = $('.signals-grid');
        signalsGrid.empty();
        
        if (data.signals.length === 0) {
            signalsGrid.html('<div class="no-data">No signals found</div>');
            return;
        }
        
        data.signals.forEach(signal => {
            const signalCard = createSignalCard(signal);
            signalsGrid.append(signalCard);
        });
        
    } catch (error) {
        console.error('Error loading signals:', error);
        $('.signals-grid').html('<div class="error">Error loading signals</div>');
    }
}

// Create trade card
function createTradeCard(trade) {
    const profitLoss = trade.profit_loss || 0;
    const profitLossClass = profitLoss > 0 ? 'profit' : profitLoss < 0 ? 'loss' : '';
    
    return `
        <div class="trade-card">
            <div class="trade-header">
                <span class="trade-date">${new Date(trade.timestamp).toLocaleString()}</span>
                <span class="trade-trader">${trade.trader_id}</span>
            </div>
            <div class="trade-details">
                <div class="trade-action ${trade.side}">${trade.side.toUpperCase()}</div>
                <div class="trade-symbol">${trade.symbol}</div>
                <div class="trade-amount">${trade.amount}</div>
                <div class="trade-price">${trade.price}</div>
                <div class="trade-value">${(trade.amount * trade.price).toFixed(2)}</div>
                <div class="trade-pl ${profitLossClass}">${profitLoss.toFixed(2)}</div>
            </div>
        </div>
    `;
}

// Create signal card
function createSignalCard(signal) {
    const statusClass = signal.status.toLowerCase();
    
    return `
        <div class="signal-card">
            <div class="signal-header">
                <span class="signal-date">${new Date(signal.timestamp).toLocaleString()}</span>
                <span class="signal-trader">${signal.trader_id}</span>
                <span class="signal-status ${statusClass}">${signal.status}</span>
            </div>
            <div class="signal-details">
                <div class="signal-type">${signal.type}</div>
                <div class="signal-symbol">${signal.symbol}</div>
                <div class="signal-price">${signal.price}</div>
                <div class="signal-target">${signal.target_price || '-'}</div>
                <div class="signal-stop">${signal.stop_loss || '-'}</div>
                <div class="signal-strength">${signal.strength || '-'}</div>
            </div>
        </div>
    `;
}

// Update pagination
function updatePagination(pagination, type) {
    const prefix = type === 'history' ? '' : 'signal-';
    const $prevBtn = $(`#${prefix}prev-page`);
    const $nextBtn = $(`#${prefix}next-page`);
    const $pageInfo = $(`#${prefix}page-info`);
    
    $prevBtn.prop('disabled', pagination.current_page === 1);
    $nextBtn.prop('disabled', pagination.current_page === pagination.total_pages);
    $pageInfo.text(`Page ${pagination.current_page} of ${pagination.total_pages}`);
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