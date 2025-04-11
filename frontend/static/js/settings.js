// Settings JavaScript

// Document ready handler
$(document).ready(function() {
    // Initialize settings components
    initSettings();
    
    // Set up event listeners
    setupEventListeners();
});

// Initialize settings components
function initSettings() {
    // Load saved settings (would fetch from server in production)
    loadSavedSettings();
}

// Set up event listeners
function setupEventListeners() {
    // General settings form
    $('#general-settings-form').on('submit', function(e) {
        e.preventDefault();
        saveGeneralSettings();
    });
    
    // Trader settings forms
    $('#aggressive-trader-form').on('submit', function(e) {
        e.preventDefault();
        saveTraderSettings('aggressive');
    });
    
    $('#conservative-trader-form').on('submit', function(e) {
        e.preventDefault();
        saveTraderSettings('conservative');
    });
    
    // Add event listeners for other traders as needed
    
    // Notifications settings form
    $('#notifications-settings-form').on('submit', function(e) {
        e.preventDefault();
        saveNotificationSettings();
    });
    
    // API settings form
    $('#api-settings-form').on('submit', function(e) {
        e.preventDefault();
        saveAPISettings();
    });
    
    // Dark mode toggle
    $('#dark-mode').on('change', function() {
        toggleDarkMode($(this).is(':checked'));
    });
}

// Load saved settings
function loadSavedSettings() {
    // In production, this would load from server/local storage
    // For now, we'll use defaults set in the HTML
    
    // Apply dark mode if enabled
    if ($('#dark-mode').is(':checked')) {
        toggleDarkMode(true);
    }
}

// Save general settings
function saveGeneralSettings() {
    const settings = {
        baseCurrency: $('#base-currency').val(),
        defaultAmount: $('#default-amount').val(),
        refreshInterval: $('#refresh-interval').val(),
        darkMode: $('#dark-mode').is(':checked')
    };
    
    // In production, this would save to server/local storage
    console.log('Saving general settings:', settings);
    
    // Simulate API call
    simulateAPICall('General settings saved successfully!');
}

// Save trader settings
function saveTraderSettings(traderType) {
    const settings = {
        enabled: $(`#${traderType}-enabled`).is(':checked'),
        riskLevel: $(`#${traderType}-risk`).val(),
        allocation: $(`#${traderType}-allocation`).val(),
        confidence: $(`#${traderType}-confidence`).val(),
        timeout: $(`#${traderType}-timeout`).val(),
        symbols: Array.from($(`#${traderType}-symbols option:selected`)).map(option => option.value)
    };
    
    // In production, this would save to server/local storage
    console.log(`Saving ${traderType} trader settings:`, settings);
    
    // Simulate API call
    simulateAPICall(`${traderType.charAt(0).toUpperCase() + traderType.slice(1)} trader settings saved successfully!`);
}

// Save notification settings
function saveNotificationSettings() {
    const settings = {
        emailNotifications: $('#email-notifications').is(':checked'),
        emailAddress: $('#email-address').val(),
        pushNotifications: $('#push-notifications').is(':checked'),
        triggers: {
            tradeExecuted: $('#trade-executed').is(':checked'),
            priceAlert: $('#price-alert').is(':checked'),
            portfolioChange: $('#portfolio-change').is(':checked'),
            traderError: $('#trader-error').is(':checked')
        }
    };
    
    // In production, this would save to server/local storage
    console.log('Saving notification settings:', settings);
    
    // Simulate API call
    simulateAPICall('Notification settings saved successfully!');
}

// Save API settings
function saveAPISettings() {
    const settings = {
        exchange: $('#exchange-select').val(),
        apiKey: $('#api-key').val(),
        apiSecret: $('#api-secret').val(),
        enableTrading: $('#enable-trading').is(':checked')
    };
    
    // In production, this would save to server/local storage
    console.log('Saving API settings:', settings);
    
    // Simulate API call
    simulateAPICall('API settings saved successfully!');
}

// Toggle dark mode
function toggleDarkMode(enabled) {
    if (enabled) {
        $('body').addClass('dark-mode');
    } else {
        $('body').removeClass('dark-mode');
    }
    
    // In production, this would save to server/local storage
    console.log('Dark mode:', enabled);
}

// Simulate API call with delay
function simulateAPICall(successMessage) {
    // Show loading indicator
    const submitBtn = $(document.activeElement);
    const originalText = submitBtn.text();
    submitBtn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...');
    
    // Simulate delay
    setTimeout(() => {
        // Show success message
        showSuccessMessage(successMessage);
        
        // Reset button
        submitBtn.prop('disabled', false).text(originalText);
    }, 1000);
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