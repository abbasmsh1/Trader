// Trade History JavaScript

// Document ready handler
$(document).ready(function() {
    // Initialize the history components
    initHistory();
    
    // Set up event listeners
    setupEventListeners();
    
    // Fetch initial data
    fetchTraders().then(() => {
        fetchHistoryData();
    });
});

// Initialize history components
function initHistory() {
    // Populate symbols dropdown
    populateSymbolsDropdown();
}

// Set up event listeners
function setupEventListeners() {
    // Refresh button click
    $('.refresh-btn').on('click', function() {
        $(this).addClass('fa-spin');
        fetchHistoryData().then(() => {
            setTimeout(() => $(this).removeClass('fa-spin'), 500);
        });
    });
    
    // Trader dropdown change
    $('#trader-select').on('change', function() {
        fetchHistoryData();
    });
    
    // Apply filters button click
    $('#apply-filters').on('click', function() {
        fetchHistoryData();
    });
    
    // Reset filters button click
    $('#reset-filters').on('click', function() {
        $('#date-range').val('30');
        $('#action-filter').val('all');
        $('#symbol-filter').val('all');
        $('#min-amount').val('5');
        fetchHistoryData();
    });
    
    // Pagination clicks
    $('.pagination .page-link').on('click', function(e) {
        e.preventDefault();
        if (!$(this).parent().hasClass('disabled') && !$(this).parent().hasClass('active')) {
            const page = $(this).text();
            if (page === 'Previous') {
                const activePage = parseInt($('.pagination .active').text());
                if (activePage > 1) {
                    updatePagination(activePage - 1);
                    fetchHistoryData(activePage - 1);
                }
            } else if (page === 'Next') {
                const activePage = parseInt($('.pagination .active').text());
                const maxPage = $('.pagination .page-item:not(.disabled):not(:contains("Previous")):not(:contains("Next"))').length;
                if (activePage < maxPage) {
                    updatePagination(activePage + 1);
                    fetchHistoryData(activePage + 1);
                }
            } else {
                updatePagination(parseInt(page));
                fetchHistoryData(parseInt(page));
            }
        }
    });
}

// Fetch traders list
function fetchTraders() {
    return $.ajax({
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
            
            // Trigger change to load initial data for first trader
            select.trigger('change');
        },
        error: function(error) {
            console.error('Error fetching traders:', error);
            showErrorMessage('Failed to load traders');
        }
    });
}

// Populate symbols dropdown
function populateSymbolsDropdown() {
    // This would be dynamic in production
    const symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'];
    const select = $('#symbol-filter');
    
    symbols.forEach(symbol => {
        select.append(
            $('<option>', {
                value: symbol,
                text: symbol
            })
        );
    });
}

// Fetch history data
function fetchHistoryData(page = 1) {
    const traderId = $('#trader-select').val();
    const dateRange = $('#date-range').val();
    const actionFilter = $('#action-filter').val();
    const symbolFilter = $('#symbol-filter').val();
    const minAmount = $('#min-amount').val();
    
    if (!traderId) return Promise.resolve();
    
    return $.ajax({
        url: `/api/history/${traderId}`,
        method: 'GET',
        data: {
            page: page,
            date_range: dateRange,
            action: actionFilter,
            symbol: symbolFilter,
            min_amount: minAmount
        },
        beforeSend: function() {
            $('#history-loading').show();
        },
        success: function(data) {
            updateHistoryTable(data.trades);
            updatePagination(page, data.total_pages);
            $('#history-loading').hide();
        },
        error: function(error) {
            console.error('Error fetching history data:', error);
            showErrorMessage('Failed to load trade history');
            $('#history-loading').hide();
        }
    });
}

// Update history table
function updateHistoryTable(trades) {
    const tableBody = $('#history-table tbody');
    tableBody.empty();
    
    if (trades.length === 0) {
        tableBody.append('<tr><td colspan="8" class="text-center">No trades found</td></tr>');
        return;
    }
    
    trades.forEach(trade => {
        const row = $('<tr>');
        const date = new Date(trade.timestamp);
        const actionClass = trade.action === 'buy' ? 'text-success' : 'text-danger';
        const statusClass = trade.status === 'completed' ? 'text-success' : (trade.status === 'pending' ? 'text-warning' : 'text-danger');
        
        row.append(`<td>${date.toLocaleString()}</td>`);
        row.append(`<td class="${actionClass}">${trade.action.toUpperCase()}</td>`);
        row.append(`<td>${trade.symbol}</td>`);
        row.append(`<td>${parseFloat(trade.quantity).toFixed(8)}</td>`);
        row.append(`<td>$${parseFloat(trade.price).toFixed(2)}</td>`);
        row.append(`<td>$${parseFloat(trade.value).toFixed(2)}</td>`);
        row.append(`<td>$${parseFloat(trade.fee).toFixed(2)}</td>`);
        row.append(`<td class="${statusClass}">${trade.status.charAt(0).toUpperCase() + trade.status.slice(1)}</td>`);
        
        tableBody.append(row);
    });
}

// Update pagination
function updatePagination(currentPage, totalPages = 5) {
    const pagination = $('.pagination');
    const prevItem = pagination.find('.page-item:first-child');
    const nextItem = pagination.find('.page-item:last-child');
    
    // Reset pagination
    pagination.find('.page-item:not(:first-child):not(:last-child)').remove();
    
    // Add page links
    for (let i = 1; i <= totalPages; i++) {
        const pageItem = $('<li class="page-item"><a class="page-link" href="#">' + i + '</a></li>');
        if (i === currentPage) {
            pageItem.addClass('active');
        }
        nextItem.before(pageItem);
    }
    
    // Update prev/next buttons
    if (currentPage === 1) {
        prevItem.addClass('disabled');
    } else {
        prevItem.removeClass('disabled');
    }
    
    if (currentPage === totalPages) {
        nextItem.addClass('disabled');
    } else {
        nextItem.removeClass('disabled');
    }
    
    // Re-bind click events
    setupEventListeners();
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