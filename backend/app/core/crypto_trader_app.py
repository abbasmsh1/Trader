import logging
import threading
import time
from typing import Dict, Any, Optional

from db.pickle_db import PickleDBHandler
from models.wallet import Wallet
from agents.trader.base_trader import BaseTraderAgent
from agents.trader.buffett_trader import BuffettTraderAgent
from agents.trader.soros_trader import SorosTraderAgent
from agents.trader.simons_trader import SimonsTraderAgent
from agents.trader.lynch_trader import LynchTraderAgent

class CryptoTraderApp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = PickleDBHandler(config['db_path'])
        self.wallets = {}
        self.agents = {}
        self.initialized = False
        self.initialization_error = None
        self.initialization_lock = threading.Lock()
        self.initialization_condition = threading.Condition(self.initialization_lock)
        self.initialization_thread = None
        self.initialization_timeout = 30  # seconds
        self.initialization_start_time = None

    def initialize(self) -> bool:
        """Initialize the application with proper error handling and timeout."""
        with self.initialization_lock:
            if self.initialized:
                return True
            if self.initialization_thread is not None:
                return False

            self.initialization_thread = threading.Thread(target=self._initialize_internal)
            self.initialization_thread.daemon = True
            self.initialization_start_time = time.time()
            self.initialization_thread.start()

        # Wait for initialization to complete or timeout
        with self.initialization_condition:
            self.initialization_condition.wait(timeout=self.initialization_timeout)

        with self.initialization_lock:
            if not self.initialized and self.initialization_error is None:
                self.initialization_error = "Initialization timed out"
                self.logger.error(self.initialization_error)
            return self.initialized

    def _initialize_internal(self):
        """Internal initialization method that runs in a separate thread."""
        try:
            self.logger.info("Starting application initialization...")
            
            # Initialize wallets
            self._initialize_wallets()
            
            # Initialize agents
            self._initialize_agents()
            
            # Set initialized flag
            with self.initialization_lock:
                self.initialized = True
                self.initialization_condition.notify_all()
            
            self.logger.info("Application initialization completed successfully")
            
        except Exception as e:
            with self.initialization_lock:
                self.initialization_error = str(e)
                self.initialization_condition.notify_all()
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_wallets(self):
        """Initialize wallets from configuration."""
        try:
            self.logger.info("Initializing wallets...")
            wallets_config = self.config.get('wallets', {})
            
            for wallet_id, wallet_config in wallets_config.items():
                try:
                    wallet = Wallet(
                        wallet_id=wallet_id,
                        initial_balance=wallet_config.get('initial_balance', 0.0),
                        currency=wallet_config.get('currency', 'USD'),
                        db_handler=self.db_handler
                    )
                    self.wallets[wallet_id] = wallet
                    self.logger.info(f"Initialized wallet {wallet_id}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize wallet {wallet_id}: {str(e)}")
                    raise
                    
            if not self.wallets:
                raise ValueError("No wallets configured")
                
        except Exception as e:
            self.logger.error(f"Wallet initialization failed: {str(e)}")
            raise

    def _initialize_agents(self):
        """Initialize trading agents from configuration."""
        try:
            self.logger.info("Initializing trading agents...")
            agents_config = self.config.get('agents', {})
            
            for agent_id, agent_config in agents_config.items():
                try:
                    agent_type = agent_config.get('type')
                    if not agent_type:
                        raise ValueError(f"Agent type not specified for {agent_id}")
                        
                    # Get the wallet for this agent
                    wallet_id = agent_config.get('wallet_id')
                    if not wallet_id:
                        raise ValueError(f"Wallet ID not specified for agent {agent_id}")
                        
                    wallet = self.wallets.get(wallet_id)
                    if not wallet:
                        raise ValueError(f"Wallet {wallet_id} not found for agent {agent_id}")
                        
                    # Create the appropriate agent type
                    if agent_type == 'buffett':
                        agent = BuffettTraderAgent(
                            name=agent_config.get("name", "Buffett Trader"),
                            description=agent_config.get("description", "Value investor focused on fundamentals"),
                            config=agent_config,
                            parent_id=None
                        )
                    elif agent_type == 'soros':
                        agent = SorosTraderAgent(
                            name=agent_config.get("name", "Soros Trader"),
                            description=agent_config.get("description", "Macro investor focused on reflexivity"),
                            config=agent_config,
                            parent_id=None
                        )
                    elif agent_type == 'simons':
                        agent = SimonsTraderAgent(
                            name=agent_config.get("name", "Simons Trader"),
                            description=agent_config.get("description", "Quantitative trader using statistical models"),
                            config=agent_config,
                            parent_id=None
                        )
                    elif agent_type == 'lynch':
                        agent = LynchTraderAgent(
                            name=agent_config.get("name", "Lynch Trader"),
                            description=agent_config.get("description", "Growth investor looking for tenbaggers"),
                            config=agent_config,
                            parent_id=None
                        )
                    else:
                        agent = BaseTraderAgent(
                            name=agent_config.get("name", f"Trader-{agent_type}"),
                            description=agent_config.get("description", "Base trader agent"),
                            config=agent_config,
                            parent_id=None
                        )
                        
                    # Set the wallet for the agent
                    agent.set_wallet(wallet)
                        
                    self.agents[agent_id] = agent
                    self.logger.info(f"Initialized agent {agent_id} of type {agent_type}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
                    raise
                    
            if not self.agents:
                raise ValueError("No agents configured")
                
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the application."""
        with self.initialization_lock:
            return {
                'initialized': self.initialized,
                'error': self.initialization_error,
                'wallets': {wallet_id: wallet.get_balance() for wallet_id, wallet in self.wallets.items()},
                'agents': list(self.agents.keys())
            }

    def get_wallet(self, wallet_id: str) -> Optional[Wallet]:
        """Get a wallet by ID."""
        return self.wallets.get(wallet_id)

    def get_agent(self, agent_id: str) -> Optional[BaseTraderAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def start(self, run_once: bool = False) -> bool:
        """
        Start the trading system.
        
        Args:
            run_once: If True, run one iteration and stop. If False, run continuously.
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if not self.initialized:
                if not self.initialize():
                    self.logger.error("Failed to initialize trading system")
                    return False
                    
            self.logger.info("Trading system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {str(e)}")
            return False

    def execute_trade(self, agent_id: str, symbol: str, amount: float, action: str) -> Dict[str, Any]:
        """Execute a trade through a specific agent."""
        if not self.initialized:
            raise RuntimeError("Application not initialized")
            
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
            
        return agent.execute_trade(symbol, amount, action) 