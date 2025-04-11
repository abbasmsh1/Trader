"""
System Controller Agent - Manages the execution of all agents in the system.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from agents.execution_agent import ExecutionAgent
from agents.trader.base_trader import BaseTraderAgent
from agents.trader.buffett_trader import BuffettTraderAgent
from models.wallet import Wallet
from utils.shared_memory import SharedMemory
from agents.analyzer.historical_analyzer import HistoricalDataAnalyzerAgent

class SystemControllerAgent(BaseAgent):
    """
    System controller agent that manages the execution of all agents.
    
    Responsibilities:
    - Agent lifecycle management
    - Execution order coordination
    - Plan management and execution
    - System state monitoring
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the system controller.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="system_controller")
        
        # System state
        self.system_state = "initializing"
        self.execution_order = []
        
        # Agent management
        self.children = []
        self.agent_registry = {}
        
        # Shared memory
        self.shared_memory = SharedMemory()
        
        # Initialize child agents
        self._initialize_child_agents()
        
        # Derive execution order
        self.execution_order = self._derive_execution_order()
        
        self.logger.info(f"System Controller initialized with {len(self.children)} agents")
    
    def _initialize_child_agents(self) -> None:
        """Initialize child agents based on configuration."""
        try:
            agent_config = self.config.get("agents", {})
            
            # Create historical data analyzer
            if "historical_analyzer" in agent_config and agent_config["historical_analyzer"].get("enabled", True):
                try:
                    hist_config = agent_config["historical_analyzer"]
                    hist_analyzer = HistoricalDataAnalyzerAgent(
                        name=hist_config.get("name", "HistoricalAnalyzer"),
                        description="Analyzes historical market data and provides insights",
                        config=hist_config,
                        parent_id=self.id
                    )
                    
                    # Add to children
                    self.add_child_agent(hist_analyzer)
                    self.register_agent(hist_analyzer.id, hist_analyzer)
                    self.logger.info(f"Created historical data analyzer: {hist_analyzer.name}")
                except Exception as e:
                    self.logger.error(f"Failed to create historical data analyzer: {str(e)}")
            
            # Create market data service if configured
            if "market_data_service" in agent_config and agent_config["market_data_service"].get("enabled", True):
                try:
                    market_config = agent_config["market_data_service"]
                    market_service = MarketDataService(
                        api_key=market_config.get("api_key"),
                        secret=market_config.get("secret")
                    )
                    self.market_data_service = market_service
                    self.logger.info("Created market data service")
                except Exception as e:
                    self.logger.error(f"Failed to create market data service: {str(e)}")
            
            # Create execution agent if configured
            if "execution_agent" in agent_config and agent_config["execution_agent"].get("enabled", True):
                try:
                    exec_config = agent_config["execution_agent"]
                    exec_agent = ExecutionAgent(
                        name=exec_config.get("name", "ExecutionAgent"),
                        description="Executes trades in the market",
                        config=exec_config,
                        parent_id=self.id
                    )
                    
                    # Add to children
                    self.add_child_agent(exec_agent)
                    self.register_agent(exec_agent.id, exec_agent)
                    self.logger.info(f"Created execution agent: {exec_agent.name}")
                except Exception as e:
                    self.logger.error(f"Failed to create execution agent: {str(e)}")
            
            # Create trader agents if configured
            if "trader_agents" in agent_config and isinstance(agent_config["trader_agents"], list):
                trader_configs = agent_config["trader_agents"]
                
                for trader_config in trader_configs:
                    if not trader_config.get("enabled", True):
                        continue
                    
                    try:
                        # Get the trader type/class
                        trader_type = trader_config.get("type", "base_trader")
                        trader_name = trader_config.get("name", f"Trader-{trader_type}")
                        
                        # Import the trader module dynamically
                        trader_class = None
                        try:
                            if trader_type == "base_trader":
                                from agents.trader.base_trader import BaseTraderAgent
                                trader_class = BaseTraderAgent
                            elif trader_type == "buffett":
                                from agents.trader.buffett_trader import BuffettTraderAgent
                                trader_class = BuffettTraderAgent
                            elif trader_type == "soros":
                                from agents.trader.soros_trader import SorosTraderAgent
                                trader_class = SorosTraderAgent
                            elif trader_type == "simons":
                                from agents.trader.simons_trader import SimonsTraderAgent
                                trader_class = SimonsTraderAgent
                            elif trader_type == "lynch":
                                from agents.trader.lynch_trader import LynchTraderAgent
                                trader_class = LynchTraderAgent
                            else:
                                # Try to import specialized trader types
                                module_name = f"agents.trader.{trader_type}_trader"
                                class_name = trader_type.split('_')[-1].title() + "TraderAgent"
                                
                                module = __import__(module_name, fromlist=[class_name])
                                trader_class = getattr(module, class_name)
                                
                        except (ImportError, AttributeError) as e:
                            self.logger.warning(f"Could not import trader class for {trader_type}: {str(e)}")
                            self.logger.info(f"Falling back to BaseTraderAgent for {trader_name}")
                            from agents.trader.base_trader import BaseTraderAgent
                            trader_class = BaseTraderAgent
                        
                        # Create trader instance
                        trader = trader_class(
                            name=trader_name,
                            description=trader_config.get("description", ""),
                            config=trader_config,
                            parent_id=self.id
                        )
                        
                        # Add to children
                        self.add_child_agent(trader)
                        self.register_agent(trader.id, trader)
                        self.logger.info(f"Created trader agent: {trader.name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create trader agent: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error initializing child agents: {str(e)}")
            raise
    
    def _derive_execution_order(self) -> List[str]:
        """
        Derive the optimal execution order for agents.
        
        Returns:
            List of agent IDs in execution order
        """
        # A simple implementation would be:
        # 1. Data collection agents first
        # 2. Analysis agents next
        # 3. Decision-making agents next
        # 4. Execution agents last
        
        # For now, just use a predefined order
        order = []
        
        # Add historical analyzer first
        for agent in self.children:
            if hasattr(agent, 'agent_type') and agent.agent_type == "historical_analyzer":
                order.append(agent.id)
        
        # Add market analyzer agents (if any)
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'analyzer' in agent.agent_type and agent.agent_type != "historical_analyzer":
                order.append(agent.id)
        
        # Add strategy agents (if any)
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'strategy' in agent.agent_type:
                order.append(agent.id)
        
        # Add trader agents
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'trader' in agent.agent_type:
                order.append(agent.id)
        
        # Add portfolio manager (if any)
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'portfolio' in agent.agent_type:
                order.append(agent.id)
        
        # Add risk manager (if any)
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'risk' in agent.agent_type:
                order.append(agent.id)
        
        # Add execution agent (if any)
        for agent in self.children:
            if hasattr(agent, 'agent_type') and 'execution' in agent.agent_type:
                order.append(agent.id)
        
        # Add any remaining agents
        for agent in self.children:
            if agent.id not in order:
                order.append(agent.id)
        
        return order
    
    def add_child_agent(self, agent: BaseAgent) -> None:
        """
        Add a child agent to the system.
        
        Args:
            agent: Agent to add
        """
        self.children.append(agent)
        self.logger.info(f"Added child agent: {agent.name}")
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register an agent in the system registry.
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        self.agent_registry[agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def update(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update the system controller and all child agents.
        
        Args:
            data: Input data for the update cycle
            
        Returns:
            Dictionary with results of the update
        """
        if not self.active:
            self.logger.warning("Attempted to update inactive system controller")
            return {"success": False, "error": "System controller inactive"}
        
        try:
            self.last_update = datetime.now()
            
            # Collect market data
            market_data = self._collect_market_data(data)
            
            # Update system state
            self.system_state = "updating"
            self.shared_memory.update_system_data({
                "system_state": self.system_state,
                "active_agents": [agent.id for agent in self.children if agent.active]
            })
            
            # Execute agents in order
            results = {}
            for agent_id in self.execution_order:
                agent = self.agent_registry.get(agent_id)
                if not agent or not agent.active:
                    continue
                
                # Check dependencies
                if not self._check_dependencies(agent):
                    self.logger.warning(f"Skipping agent {agent.name} due to unmet dependencies")
                    continue
                
                # Update agent with market data
                agent_data = {"market_data": market_data}
                
                # Get agent's plan
                plan = agent.plan(agent_data)
                
                # Execute plan steps
                while True:
                    result = agent.execute(plan)
                    if not result.get("success", False):
                        self.logger.error(f"Agent {agent.name} execution failed: {result.get('error', 'Unknown error')}")
                        break
                    
                    if result.get("plan_status") == "complete":
                        break
                
                # Process result
                self._process_agent_result(agent, result)
                results[agent_id] = result
            
            # Update system state
            self.system_state = "idle"
            self.shared_memory.update_system_data({
                "system_state": self.system_state,
                "last_update": self.last_update.isoformat()
            })
            
            return {
                "success": True,
                "results": results,
                "system_state": self.system_state
            }
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error updating system controller: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _collect_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect market data for the current update cycle.
        
        Args:
            data: Input data that may contain market data
            
        Returns:
            Market data for the update cycle
        """
        # If input data contains market data, use it
        if "market_data" in data:
            return data["market_data"]
        
        # If using live data, get it from the market data service
        if self.use_live_data and self.market_data_service:
            return self.market_data_service.get_latest_data()
        
        # Otherwise, return an empty dict (agents will need to handle this)
        return {}
    
    def _check_dependencies(self, agent: BaseAgent) -> bool:
        """
        Check if dependencies are met for an agent.
        
        Args:
            agent: Agent to check dependencies for
            
        Returns:
            True if all dependencies are met, False otherwise
        """
        # In a more complex system, we would check if all required
        # input data is available for the agent to function
        # For now, just return True
        return True
    
    def _process_agent_result(self, agent: BaseAgent, result: Dict[str, Any]) -> None:
        """
        Process the result of an agent update.
        
        Args:
            agent: Agent that produced the result
            result: Result data from the agent update
        """
        # In a more complex system, we would process agent-specific results
        # For now, just log the result
        if not result.get("success", True):
            self.logger.warning(f"Agent {agent.name} update failed: {result.get('error', 'Unknown error')}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the system controller.
        
        Returns:
            Dictionary representing system controller state
        """
        # Get base state from parent class
        state = super().get_state()
        
        # Add system controller specific state
        state["system_state"] = self.system_state
        state["execution_order"] = self.execution_order
        
        # Add child agent IDs
        state["children"] = [agent.id for agent in self.children]
        
        return state 