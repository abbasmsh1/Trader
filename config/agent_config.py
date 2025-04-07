"""
Agent Configuration - Defines the hierarchical structure of trading agents.

This module provides the configuration for setting up the agent hierarchy,
including agent types, relationships, and execution order.
"""

# Agent dependencies - defines which agents depend on results from other agents
AGENT_DEPENDENCIES = {
    # Market data analyzers have no dependencies
    "btc_analyzer": [],
    "eth_analyzer": [],
    "general_market_analyzer": [],
    
    # Strategy agents depend on market analyzers
    "trend_strategy": ["btc_analyzer", "eth_analyzer", "general_market_analyzer"],
    "swing_strategy": ["btc_analyzer", "eth_analyzer", "general_market_analyzer"],
    "breakout_strategy": ["btc_analyzer", "eth_analyzer", "general_market_analyzer"],
    
    # Portfolio manager depends on strategies and market analyzers
    "portfolio_manager": ["trend_strategy", "swing_strategy", "breakout_strategy", 
                         "btc_analyzer", "eth_analyzer", "general_market_analyzer"],
    
    # Execution agent depends on portfolio manager
    "execution_agent": ["portfolio_manager"],
    
    # Risk manager depends on portfolio manager and market analyzers
    "risk_manager": ["portfolio_manager", "btc_analyzer", "eth_analyzer", "general_market_analyzer"],
    
    # System controller depends on all agents (though in practice, it controls them)
    "system_controller": ["portfolio_manager", "execution_agent", "risk_manager"]
}

# Agent execution order - defines the sequence in which agents are executed
EXECUTION_ORDER = [
    # First, analyze market data
    "btc_analyzer",
    "eth_analyzer", 
    "general_market_analyzer",
    
    # Then, run strategies based on analysis
    "trend_strategy",
    "swing_strategy",
    "breakout_strategy",
    
    # Manage portfolio based on strategies
    "portfolio_manager",
    
    # Execute trades based on portfolio decisions
    "execution_agent",
    
    # Monitor risk
    "risk_manager",
    
    # System-level decisions
    "system_controller"
]

# Agent hierarchy - defines the parent-child relationships
AGENT_HIERARCHY = {
    "system_controller": {
        "id": "system_controller",
        "name": "System Controller",
        "type": "system_controller",
        "description": "Top-level agent that orchestrates the entire trading system",
        "config": {
            "update_interval": 60,  # seconds
            "training_interval": 24 * 60 * 60,  # daily
            "save_state_interval": 60 * 60,  # hourly
            "agent_timeout": 30,  # seconds
            "max_consecutive_errors": 5,
            "recovery_wait_time": 300,  # seconds
            "trading_hours": {
                "enabled": False, 
                "start": "00:00", 
                "end": "23:59"
            },
            "maintenance_window": {
                "enabled": True, 
                "day": 6,  # Sunday
                "hour": 2  # 2 AM
            },
            "emergency_shutdown_drawdown": 30,  # 30% drawdown triggers shutdown
            "agent_dependencies": AGENT_DEPENDENCIES,
            "execution_order": EXECUTION_ORDER
        },
        "children": [
            # Market data analyzers
            {
                "id": "btc_analyzer",
                "name": "BTC Analyzer",
                "type": "market_analyzer",
                "description": "Analyzes BTC market data",
                "config": {
                    "symbols": ["BTC/USDT"],
                    "timeframes": ["1h", "4h", "1d"],
                    "short_ma_period": 20,
                    "medium_ma_period": 50,
                    "long_ma_period": 200,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std_dev": 2
                }
            },
            {
                "id": "eth_analyzer",
                "name": "ETH Analyzer",
                "type": "market_analyzer",
                "description": "Analyzes ETH market data",
                "config": {
                    "symbols": ["ETH/USDT"],
                    "timeframes": ["1h", "4h", "1d"],
                    "short_ma_period": 20,
                    "medium_ma_period": 50,
                    "long_ma_period": 200,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std_dev": 2
                }
            },
            {
                "id": "general_market_analyzer",
                "name": "General Market Analyzer",
                "type": "market_analyzer",
                "description": "Analyzes overall market conditions",
                "config": {
                    "symbols": ["BNB/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT"],
                    "timeframes": ["1d"],
                    "short_ma_period": 20,
                    "medium_ma_period": 50,
                    "long_ma_period": 200,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std_dev": 2
                }
            },
            
            # Strategy agents
            {
                "id": "trend_strategy",
                "name": "Trend Following Strategy",
                "type": "trend_strategy",
                "description": "Implements trend following strategies",
                "config": {
                    "fast_ma": 10,
                    "slow_ma": 30,
                    "trend_strength_threshold": 0.02,
                    "confirmation_period": 3,
                    "position_size_pct": 10,
                    "stop_loss_pct": 5,
                    "profit_target_pct": 15,
                    "max_trades": 5
                }
            },
            {
                "id": "swing_strategy",
                "name": "Swing Trading Strategy",
                "type": "swing_strategy",
                "description": "Implements swing trading strategies",
                "config": {
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "reversion_strength": 0.05,
                    "min_holding_period": 12,  # hours
                    "max_holding_period": 96,  # hours
                    "position_size_pct": 8,
                    "stop_loss_pct": 8,
                    "profit_target_pct": 12,
                    "max_trades": 3
                }
            },
            {
                "id": "breakout_strategy",
                "name": "Breakout Strategy",
                "type": "breakout_strategy",
                "description": "Implements breakout trading strategies",
                "config": {
                    "breakout_period": 20,
                    "volume_multiplier_threshold": 2.0,
                    "confirmation_candles": 2,
                    "false_breakout_filter": True,
                    "position_size_pct": 5,
                    "stop_loss_pct": 7,
                    "profit_target_pct": 20,
                    "max_trades": 3
                }
            },
            
            # Portfolio management
            {
                "id": "portfolio_manager",
                "name": "Portfolio Manager",
                "type": "portfolio_manager",
                "description": "Manages asset allocation and risk across the portfolio",
                "config": {
                    "initial_balance": 10000,
                    "max_allocation_pct": 20,
                    "max_risk_per_trade_pct": 2,
                    "min_cash_reserve_pct": 15,
                    "rebalance_threshold_pct": 10,
                    "drawdown_protective_threshold": 15,
                    "volatility_adjustment": True,
                    "portfolio_targets": {
                        "BTC/USDT": 40,
                        "ETH/USDT": 30,
                        "SOL/USDT": 15,
                        "BNB/USDT": 10,
                        "ADA/USDT": 5
                    }
                }
            },
            
            # Execution agent
            {
                "id": "execution_agent",
                "name": "Execution Agent",
                "type": "execution_agent",
                "description": "Handles the execution of trades",
                "config": {
                    "exchange": "binance",
                    "use_testnet": True,
                    "slippage_model": "conservative",
                    "max_slippage_pct": 1.0,
                    "retry_attempts": 3,
                    "retry_delay": 5,  # seconds
                    "execution_strategies": {
                        "market": True,
                        "limit": True,
                        "twap": False,
                        "iceberg": False
                    },
                    "order_lifetime": 300  # seconds
                }
            },
            
            # Risk management
            {
                "id": "risk_manager",
                "name": "Risk Manager",
                "type": "risk_manager",
                "description": "Monitors and manages overall system risk",
                "config": {
                    "max_portfolio_drawdown": 25,  # percent
                    "max_single_asset_allocation": 40,  # percent
                    "intraday_loss_limit": 5,  # percent
                    "correlation_threshold": 0.7,
                    "volatility_scaling": True,
                    "risk_free_rate": 2.0,  # percent
                    "risk_metrics": ["sharpe", "sortino", "calmar", "max_drawdown"],
                    "var_confidence_level": 95,  # percent
                    "stress_test_scenarios": ["2018_bear", "2020_crash", "2021_may_crash"]
                }
            }
        ]
    }
}

# Function to create agent instances from hierarchy
def create_agent_instances(agent_manager, db_handler):
    """
    Create agent instances based on the defined hierarchy.
    
    Args:
        agent_manager: AgentManager instance to use for creating agents
        db_handler: Database handler for persistence
        
    Returns:
        Dictionary mapping agent IDs to agent instances
    """
    # Create system controller first
    system_config = AGENT_HIERARCHY["system_controller"]
    system_controller = agent_manager.create_agent(
        agent_type="system_controller",
        name=system_config["name"],
        description=system_config["description"],
        config=system_config["config"]
    )
    
    # Dictionary to hold all created agents
    created_agents = {system_controller.id: system_controller}
    
    # Create all child agents
    for child_config in system_config["children"]:
        agent = agent_manager.create_agent(
            agent_type=child_config["type"],
            name=child_config["name"],
            description=child_config["description"],
            config=child_config["config"],
            parent_id=system_controller.id
        )
        
        if agent:
            created_agents[child_config["id"]] = agent
            
            # Register the agent with the system controller
            system_controller.register_agent(child_config["id"], agent)
    
    # Try to load any previously saved state
    agent_manager.load_agents_state()
    
    return created_agents 