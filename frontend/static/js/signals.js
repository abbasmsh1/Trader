// Signals JavaScript

// Document ready handler
$(document).ready(function() {
    loadTraders();
    loadSignals();
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    // Filter change handlers
    $('#trader-select').on('change', loadSignals);
    $('#status-select').on('change', loadSignals);
    
    // Refresh button handler
    $('#refresh-btn').on('click', loadSignals);
}

// Load traders into the select dropdown
async function loadTraders() {
    try {
        const response = await fetch('/api/traders');
        if (!response.ok) throw new Error('Failed to fetch traders');
        
        const traders = await response.json();
        const traderSelect = $('#trader-select');
        
        // Clear existing options except "All Traders"
        traderSelect.find('option:not(:first)').remove();
        
        // Add trader options
        traders.forEach(trader => {
            traderSelect.append(`<option value="${trader.id}">${trader.name}</option>`);
        });
    } catch (error) {
        console.error('Error loading traders:', error);
    }
}

// Load signals based on current filters
async function loadSignals() {
    try {
        const traderId = $('#trader-select').val();
        const status = $('#status-select').val();
        
        // Show loading state
        const signalsGrid = $('#signals-grid');
        signalsGrid.html('<div class="loading">Loading signals...</div>');
        
        // Fetch signals based on filters
        let url = '/api/signals';
        if (traderId !== 'all') {
            url += `/${traderId}`;
            if (status === 'active') {
                url += '/active';
            }
        }
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch signals');
        
        const signals = await response.json();
        
        // Clear and update signals grid
        signalsGrid.empty();
        
        if (signals.length === 0) {
            signalsGrid.html('<div class="no-signals">No signals found</div>');
            return;
        }
        
        // Create signal cards
        signals.forEach(signal => {
            const card = createSignalCard(signal);
            signalsGrid.append(card);
        });
    } catch (error) {
        console.error('Error loading signals:', error);
        $('#signals-grid').html('<div class="error">Failed to load signals</div>');
    }
}

// Create a signal card element
function createSignalCard(signal) {
    const statusClass = signal.status.toLowerCase();
    const timestamp = new Date(signal.timestamp).toLocaleString();
    
    return `
        <div class="card signal-card ${statusClass}">
            <div class="card-header">
                <h3>${signal.symbol}</h3>
                <span class="status-badge ${statusClass}">${signal.status}</span>
            </div>
            <div class="card-body">
                <div class="signal-info">
                    <p><strong>Type:</strong> ${signal.type}</p>
                    <p><strong>Price:</strong> ${signal.price}</p>
                    <p><strong>Volume:</strong> ${signal.volume}</p>
                    <p><strong>Trader:</strong> ${signal.trader_id}</p>
                    <p><strong>Time:</strong> ${timestamp}</p>
                </div>
                ${signal.notes ? `<div class="notes">${signal.notes}</div>` : ''}
            </div>
        </div>
    `;
} 