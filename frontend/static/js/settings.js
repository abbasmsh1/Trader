// Crypto Trader Settings JavaScript

// Socket.io connection
const socket = io();

// Settings cache
let currentSettings = {};
let hasUnsavedChanges = false;

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupEventListeners();
    
    // Load settings from server
    loadSettings();
    
    // Connect system status indicators
    connectSystemControls();
});

// Set up event listeners for the page
function setupEventListeners() {
    // Save settings button (header)
    const saveAllSettingsBtn = document.getElementById('save-all-settings');
    if (saveAllSettingsBtn) {
        saveAllSettingsBtn.addEventListener('click', saveAllSettings);
    }
    
    // Save settings button (footer)
    const footerSaveBtn = document.getElementById('footer-save-settings');
    if (footerSaveBtn) {
        footerSaveBtn.addEventListener('click', saveAllSettings);
    }
    
    // Reset to defaults button
    const resetBtn = document.querySelector('button.btn-secondary:not([id])');
    if (resetBtn) {
        resetBtn.addEventListener('click', confirmResetSettings);
    }
    
    // Listen for form changes
    setupFormChangeListeners();
    
    // Socket.io event listeners
    socket.on('system_status', function(data) {
        updateSystemStatus(data);
    });
    
    // Before unload warning if unsaved changes
    window.addEventListener('beforeunload', function(e) {
        if (hasUnsavedChanges) {
            e.preventDefault();
            e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
            return e.returnValue;
        }
    });
}

// Set up form change listeners for all settings forms
function setupFormChangeListeners() {
    // Find all form inputs
    const allInputs = document.querySelectorAll('input, select, textarea');
    
    allInputs.forEach(input => {
        const eventType = input.type === 'checkbox' || input.type === 'radio' || input.tagName === 'SELECT' 
            ? 'change' 
            : 'input';
        
        input.addEventListener(eventType, function() {
            hasUnsavedChanges = true;
            updateSaveButtonState();
        });
    });
}

// Update save button state based on changes
function updateSaveButtonState() {
    const saveButtons = document.querySelectorAll('#save-all-settings, #footer-save-settings');
    
    saveButtons.forEach(button => {
        if (hasUnsavedChanges) {
            button.classList.remove('btn-primary');
            button.classList.add('btn-warning');
            button.innerHTML = '<i class="bi bi-save"></i> Save Changes*';
        } else {
            button.classList.remove('btn-warning');
            button.classList.add('btn-primary');
            button.innerHTML = '<i class="bi bi-save"></i> Save All Changes';
        }
    });
}

// Load settings from server
function loadSettings() {
    // Show loading state
    document.querySelectorAll('#save-all-settings, #footer-save-settings').forEach(button => {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    });
    
    // Since we're in demo mode, we'll simulate loading settings
    // In a real app, you would fetch from a server endpoint
    setTimeout(() => {
        // Simulate settings from server
        currentSettings = {
            general: {
                systemName: 'Crypto Trader',
                baseCurrency: 'USDT',
                initialBalance: 10000,
                updateInterval: 5,
                loggingEnabled: true,
                autoStart: true
            },
            ui: {
                theme: 'light',
                chartStyle: 'line',
                realTimeUpdates: true
            },
            trading: {
                tradingMode: 'paper',
                strategy: 'trend_following',
                tradableAssets: ['BTC', 'ETH', 'SOL', 'BNB'],
                timeframe: '15m',
                autoRebalance: true
            },
            risk: {
                maxPositionSize: 20,
                stopLoss: 5,
                takeProfit: 15,
                maxDrawdown: 25,
                trailingStops: true,
                autoAdjustRisk: true
            },
            exchanges: {
                binance: {
                    connected: true,
                    apiKey: '••••••••••••••••••••••',
                    apiSecret: '••••••••••••••••••••••',
                    enableTrading: true
                },
                coinbase: {
                    connected: false,
                    apiKey: '',
                    apiSecret: '',
                    passphrase: '',
                    enableTrading: false
                }
            },
            notifications: {
                email: '',
                notifyTrades: true,
                notifyProfit: true,
                notifyErrors: true,
                notifyOpportunities: false,
                telegramChatId: '',
                pushNotifications: false
            }
        };
        
        // Populate form fields with settings
        populateSettingsForms(currentSettings);
        
        // Restore save buttons
        document.querySelectorAll('#save-all-settings, #footer-save-settings').forEach(button => {
            button.disabled = false;
            button.innerHTML = '<i class="bi bi-save"></i> Save All Changes';
        });
        
        // Reset unsaved changes flag
        hasUnsavedChanges = false;
    }, 800);
    
    // Get system status
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => console.error('Error fetching system status:', error));
}

// Populate form fields with settings
function populateSettingsForms(settings) {
    // General settings
    document.getElementById('system-name').value = settings.general.systemName;
    document.getElementById('base-currency').value = settings.general.baseCurrency;
    document.getElementById('initial-balance').value = settings.general.initialBalance;
    document.getElementById('update-interval').value = settings.general.updateInterval;
    document.getElementById('logging-enabled').checked = settings.general.loggingEnabled;
    document.getElementById('auto-start').checked = settings.general.autoStart;
    
    // UI settings
    document.getElementById('theme').value = settings.ui.theme;
    document.getElementById('chart-style').value = settings.ui.chartStyle;
    document.getElementById('real-time-updates').checked = settings.ui.realTimeUpdates;
    
    // Trading settings
    document.getElementById('trading-mode').value = settings.trading.tradingMode;
    document.getElementById('strategy').value = settings.trading.strategy;
    document.getElementById('asset-btc').checked = settings.trading.tradableAssets.includes('BTC');
    document.getElementById('asset-eth').checked = settings.trading.tradableAssets.includes('ETH');
    document.getElementById('asset-sol').checked = settings.trading.tradableAssets.includes('SOL');
    document.getElementById('asset-bnb').checked = settings.trading.tradableAssets.includes('BNB');
    document.getElementById('timeframe').value = settings.trading.timeframe;
    document.getElementById('auto-rebalance').checked = settings.trading.autoRebalance;
    
    // Risk settings
    document.getElementById('max-position-size').value = settings.risk.maxPositionSize;
    document.getElementById('stop-loss').value = settings.risk.stopLoss;
    document.getElementById('take-profit').value = settings.risk.takeProfit;
    document.getElementById('max-drawdown').value = settings.risk.maxDrawdown;
    document.getElementById('trailing-stops').checked = settings.risk.trailingStops;
    document.getElementById('auto-adjust-risk').checked = settings.risk.autoAdjustRisk;
    
    // Exchange settings
    document.getElementById('binance-api-key').value = settings.exchanges.binance.apiKey;
    document.getElementById('binance-api-secret').value = settings.exchanges.binance.apiSecret;
    document.getElementById('binance-enable-trading').checked = settings.exchanges.binance.enableTrading;
    
    document.getElementById('coinbase-api-key').value = settings.exchanges.coinbase.apiKey;
    document.getElementById('coinbase-api-secret').value = settings.exchanges.coinbase.apiSecret;
    if (document.getElementById('coinbase-passphrase')) {
        document.getElementById('coinbase-passphrase').value = settings.exchanges.coinbase.passphrase;
    }
    document.getElementById('coinbase-enable-trading').checked = settings.exchanges.coinbase.enableTrading;
    
    // Notification settings
    document.getElementById('notification-email').value = settings.notifications.email;
    document.getElementById('notify-trades').checked = settings.notifications.notifyTrades;
    document.getElementById('notify-profit').checked = settings.notifications.notifyProfit;
    document.getElementById('notify-errors').checked = settings.notifications.notifyErrors;
    document.getElementById('notify-opportunities').checked = settings.notifications.notifyOpportunities;
    document.getElementById('telegram-chat-id').value = settings.notifications.telegramChatId;
    document.getElementById('push-notifications').checked = settings.notifications.pushNotifications;
}

// Get current settings from form values
function getFormSettings() {
    const settings = {
        general: {
            systemName: document.getElementById('system-name').value,
            baseCurrency: document.getElementById('base-currency').value,
            initialBalance: parseFloat(document.getElementById('initial-balance').value),
            updateInterval: parseInt(document.getElementById('update-interval').value),
            loggingEnabled: document.getElementById('logging-enabled').checked,
            autoStart: document.getElementById('auto-start').checked
        },
        ui: {
            theme: document.getElementById('theme').value,
            chartStyle: document.getElementById('chart-style').value,
            realTimeUpdates: document.getElementById('real-time-updates').checked
        },
        trading: {
            tradingMode: document.getElementById('trading-mode').value,
            strategy: document.getElementById('strategy').value,
            tradableAssets: [],
            timeframe: document.getElementById('timeframe').value,
            autoRebalance: document.getElementById('auto-rebalance').checked
        },
        risk: {
            maxPositionSize: parseFloat(document.getElementById('max-position-size').value),
            stopLoss: parseFloat(document.getElementById('stop-loss').value),
            takeProfit: parseFloat(document.getElementById('take-profit').value),
            maxDrawdown: parseFloat(document.getElementById('max-drawdown').value),
            trailingStops: document.getElementById('trailing-stops').checked,
            autoAdjustRisk: document.getElementById('auto-adjust-risk').checked
        },
        exchanges: {
            binance: {
                connected: true, // This would be determined by the server
                apiKey: document.getElementById('binance-api-key').value,
                apiSecret: document.getElementById('binance-api-secret').value,
                enableTrading: document.getElementById('binance-enable-trading').checked
            },
            coinbase: {
                connected: false, // This would be determined by the server
                apiKey: document.getElementById('coinbase-api-key').value,
                apiSecret: document.getElementById('coinbase-api-secret').value,
                passphrase: document.getElementById('coinbase-passphrase') ? document.getElementById('coinbase-passphrase').value : '',
                enableTrading: document.getElementById('coinbase-enable-trading').checked
            }
        },
        notifications: {
            email: document.getElementById('notification-email').value,
            notifyTrades: document.getElementById('notify-trades').checked,
            notifyProfit: document.getElementById('notify-profit').checked,
            notifyErrors: document.getElementById('notify-errors').checked,
            notifyOpportunities: document.getElementById('notify-opportunities').checked,
            telegramChatId: document.getElementById('telegram-chat-id').value,
            pushNotifications: document.getElementById('push-notifications').checked
        }
    };
    
    // Get tradable assets
    if (document.getElementById('asset-btc').checked) settings.trading.tradableAssets.push('BTC');
    if (document.getElementById('asset-eth').checked) settings.trading.tradableAssets.push('ETH');
    if (document.getElementById('asset-sol').checked) settings.trading.tradableAssets.push('SOL');
    if (document.getElementById('asset-bnb').checked) settings.trading.tradableAssets.push('BNB');
    
    return settings;
}

// Save all settings
function saveAllSettings() {
    // Get settings from forms
    const newSettings = getFormSettings();
    
    // Show loading state
    document.querySelectorAll('#save-all-settings, #footer-save-settings').forEach(button => {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
    });
    
    // In a real app, you would send to a server endpoint
    // For demo, simulate saving with a timeout
    setTimeout(() => {
        // Update settings cache
        currentSettings = newSettings;
        
        // Show success message
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show fixed-top w-50 mx-auto mt-3';
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `
            <strong>Success!</strong> Your settings have been saved.
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Remove alert after 3 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 3000);
        
        // Restore save buttons
        document.querySelectorAll('#save-all-settings, #footer-save-settings').forEach(button => {
            button.disabled = false;
            button.innerHTML = '<i class="bi bi-save"></i> Save All Changes';
        });
        
        // Reset unsaved changes flag
        hasUnsavedChanges = false;
        updateSaveButtonState();
    }, 1000);
}

// Confirm reset to defaults
function confirmResetSettings() {
    if (confirm('Are you sure you want to reset all settings to default values? This cannot be undone.')) {
        resetSettingsToDefaults();
    }
}

// Reset settings to defaults
function resetSettingsToDefaults() {
    // In a real app, you would fetch default settings from server
    // For demo, we'll use hardcoded defaults
    const defaultSettings = {
        general: {
            systemName: 'Crypto Trader',
            baseCurrency: 'USDT',
            initialBalance: 10000,
            updateInterval: 5,
            loggingEnabled: true,
            autoStart: false
        },
        ui: {
            theme: 'light',
            chartStyle: 'line',
            realTimeUpdates: true
        },
        trading: {
            tradingMode: 'paper',
            strategy: 'trend_following',
            tradableAssets: ['BTC', 'ETH'],
            timeframe: '15m',
            autoRebalance: false
        },
        risk: {
            maxPositionSize: 10,
            stopLoss: 5,
            takeProfit: 15,
            maxDrawdown: 25,
            trailingStops: true,
            autoAdjustRisk: false
        },
        exchanges: {
            binance: {
                connected: false,
                apiKey: '',
                apiSecret: '',
                enableTrading: false
            },
            coinbase: {
                connected: false,
                apiKey: '',
                apiSecret: '',
                passphrase: '',
                enableTrading: false
            }
        },
        notifications: {
            email: '',
            notifyTrades: true,
            notifyProfit: true,
            notifyErrors: true,
            notifyOpportunities: false,
            telegramChatId: '',
            pushNotifications: false
        }
    };
    
    // Populate form fields with default settings
    populateSettingsForms(defaultSettings);
    
    // Mark as having unsaved changes
    hasUnsavedChanges = true;
    updateSaveButtonState();
    
    // Show message
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-info alert-dismissible fade show fixed-top w-50 mx-auto mt-3';
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        <strong>Settings Reset!</strong> All settings have been reset to defaults. Click Save to apply these changes.
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.body.appendChild(alertDiv);
    
    // Remove alert after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
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