"""
System Controller Agent - Orchestrates the entire trading system.

This agent is responsible for coordinating the execution of all other agents,
managing the system lifecycle, and providing high-level decision making.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime, timedelta
import threading
import json

from agents.base_agent import BaseAgent
from models.wallet import Wallet

class SystemControllerAgent(BaseAgent):
    """
    System Controller Agent.
    
    Acts as the central coordinator for the entire trading system, orchestrating
    the execution of all child agents and making high-level system decisions.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 config: Dict[str, Any],
                 parent_id: Optional[str] = None):
        """
        Initialize the system controller agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            config: Configuration parameters
            parent_id: ID of the parent agent
        """
        super().__init__(
            name=name,
            description=description,
            agent_type="system_controller",
            config=config,
            parent_id=parent_id
        )
        
        # System parameters
        self.update_interval = config.get("update_interval", 60)  # seconds
        self.training_interval = config.get("training_interval", 24 * 60 * 60)  # daily
        self.save_state_interval = config.get("save_state_interval", 60 * 60)  # hourly
        self.agent_timeout = config.get("agent_timeout", 60)  # seconds
        self.max_consecutive_errors = config.get("max_consecutive_errors", 5)
        self.recovery_wait_time = config.get("recovery_wait_time", 300)  # seconds
        self.trading_hours = config.get("trading_hours", {"enabled": False, "start": "00:00", "end": "23:59"})
        self.maintenance_window = config.get("maintenance_window", {"enabled": False, "day": 6, "hour": 0})
        self.emergency_shutdown_drawdown = config.get("emergency_shutdown_drawdown", 30)  # 30% drawdown triggers shutdown
        
        # System state
        self.system_state = {
            "status": "initializing",  # initializing, running, maintenance, error, shutdown
            "last_update": 0,
            "last_training": 0,
            "last_state_save": 0,
            "consecutive_errors": 0,
            "error_log": [],
            "active_agent_count": 0,
            "next_scheduled_training": time.time() + self.training_interval,
            "next_scheduled_maintenance": self._calculate_next_maintenance(),
            "metrics": {
                "uptime": 0,
                "total_cycles": 0,
                "successful_cycles": 0,
                "total_signals": 0,
                "total_trades": 0
            }
        }
        
        # Agent dependencies and execution order
        self.agent_dependencies = config.get("agent_dependencies", {})
        self.execution_order = config.get("execution_order", [])
        
        # Agent references (to be populated after initialization)
        self.agent_references = {}
        
        # Background workers
        self.worker_thread = None
        self.worker_running = False
        
        # Initialize wallet
        self.wallet = Wallet(
            initial_balance=self.config.get("initial_balance", 10000.0),
            base_currency=self.config.get("base_currency", "USDT"),
            name="System-Wallet"
        )
        
        self.logger.info(f"System Controller Agent {self.name} initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the system controller and prepare for execution.
        
        This method is called after the controller is created and the wallet is set.
        It sets up any required child agents and initializes the system for running.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing {self.name} with configuration...")
            
            # Set system state to initializing
            self.system_state["status"] = "initializing"
            
            # Create child agents based on configuration
            self._initialize_child_agents()
            
            # Derive the execution order for agents
            if not self.execution_order:
                self.execution_order = self._derive_execution_order()
                
            # Count active agents
            active_count = sum(1 for agent in self.children if agent.active)
            self.system_state["active_agent_count"] = active_count
            
            # Mark as initialized
            self.system_state["status"] = "ready"
            self.initialized = True
            
            self.logger.info(f"{self.name} initialized successfully with {active_count} active agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing system controller: {str(e)}", exc_info=True)
            self.system_state["status"] = "error"
            return False
    
    def _initialize_child_agents(self) -> None:
        """
        Initialize child agents based on configuration.
        """
        # For demo mode, we don't actually create real agents
        # In a real implementation, this would create the portfolio manager,
        # strategy agents, data agent, etc. based on the configuration
        
        if self.config.get("is_demo", False):
            self.logger.info("Running in demo mode, skipping actual agent creation")
            # In demo mode, we'll just create placeholder objects for demo purposes
            return
    
    def set_wallet(self, wallet: Wallet) -> None:
        """
        Set the wallet for the system controller.
        
        Args:
            wallet: Wallet instance to use for the system
        """
        self.wallet = wallet
        self.logger.info(f"Wallet set for {self.name}: {wallet.name}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a full system cycle.
        
        Args:
            data: Dictionary containing system state and input data
            
        Returns:
            Updated system state and results
        """
        # Update system state
        self.system_state["last_update"] = time.time()
        start_time = time.time()
        
        # Check for emergency shutdown condition
        if self._check_emergency_shutdown(data):
            self.logger.warning("EMERGENCY SHUTDOWN triggered due to excessive drawdown")
            self.system_state["status"] = "shutdown"
            return {
                "system_state": self.system_state,
                "error": "Emergency shutdown triggered"
            }
        
        # Check if we're in a valid trading window
        if not self._is_trading_time():
            self.logger.info("Outside trading hours, system in standby mode")
            self.system_state["status"] = "standby"
            return {
                "system_state": self.system_state,
                "message": "System in standby mode (outside trading hours)"
            }
        
        # Check if maintenance is scheduled
        if self._is_maintenance_time():
            self.logger.info("Scheduled maintenance in progress")
            self.system_state["status"] = "maintenance"
            self._perform_maintenance(data)
            return {
                "system_state": self.system_state,
                "message": "System in maintenance mode"
            }
        
        try:
            # Set status to running
            self.system_state["status"] = "running"
            
            # Execute the agent workflow
            results = self._execute_agent_workflow(data)
            
            # Process results
            processed_results = self._process_workflow_results(results)
            
            # Check if training is scheduled
            if self._should_run_training():
                self.logger.info("Scheduled training triggered")
                training_results = self._execute_training(data)
                processed_results["training_results"] = training_results
                self.system_state["last_training"] = time.time()
                self.system_state["next_scheduled_training"] = time.time() + self.training_interval
            
            # Check if state saving is needed
            if self._should_save_state():
                self.logger.info("Saving system state")
                self._save_system_state(data)
                self.system_state["last_state_save"] = time.time()
            
            # Update metrics
            self.system_state["metrics"]["total_cycles"] += 1
            self.system_state["metrics"]["successful_cycles"] += 1
            self.system_state["consecutive_errors"] = 0
            self.system_state["metrics"]["uptime"] = time.time() - self.created_at
            
            # Record execution
            self.performance_metrics["total_decisions"] += 1
            self.performance_metrics["successful_decisions"] += 1
            self.last_run = time.time()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return {
                "system_state": self.system_state,
                "results": processed_results,
                "execution_time": execution_time,
                "status": {
                    "status": "running",
                    "portfolio_value": self.wallet.get_total_value()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing system cycle: {e}", exc_info=True)
            
            # Update error tracking
            self.system_state["consecutive_errors"] += 1
            self.system_state["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            
            # Keep error log at reasonable size
            if len(self.system_state["error_log"]) > 100:
                self.system_state["error_log"] = self.system_state["error_log"][-100:]
            
            # Check if system should enter error state
            if self.system_state["consecutive_errors"] >= self.max_consecutive_errors:
                self.system_state["status"] = "error"
                
            return {
                "system_state": self.system_state,
                "error": f"System error: {str(e)}"
            }
    
    def train(self, training_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Train the system and its child agents.
        
        Args:
            training_data: Dictionary containing training data
            
        Returns:
            Tuple of (success, metrics)
        """
        self.logger.info(f"Training System Controller {self.name}")
        
        try:
            # Execute training for all child agents
            agent_results = {}
            success_count = 0
            error_count = 0
            
            for child in self.children:
                if not child.active:
                    continue
                    
                try:
                    agent_success, agent_metrics = child.train(training_data)
                    
                    agent_results[child.id] = {
                        "success": agent_success,
                        "metrics": agent_metrics
                    }
                    
                    if agent_success:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error training agent {child.name}: {e}", exc_info=True)
                    agent_results[child.id] = {
                        "success": False,
                        "error": str(e)
                    }
                    error_count += 1
            
            # Record training
            self.record_training()
            
            # Calculate success ratio
            total_agents = success_count + error_count
            success_ratio = success_count / total_agents if total_agents > 0 else 0
            
            return success_ratio > 0.5, {
                "agent_results": agent_results,
                "success_count": success_count,
                "error_count": error_count,
                "success_ratio": success_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error in system training: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def start_worker(self, system_data: Dict[str, Any], db_handler: Any) -> None:
        """
        Start the background worker thread for continuous operation.
        
        Args:
            system_data: Initial system data
            db_handler: Database handler for state persistence
        """
        if self.worker_thread and self.worker_thread.is_alive():
            self.logger.warning("Worker thread already running")
            return
            
        self.worker_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(system_data, db_handler),
            daemon=True
        )
        self.worker_thread.start()
        self.logger.info("Started system controller worker thread")
    
    def stop_worker(self) -> None:
        """Stop the background worker thread."""
        if not self.worker_running:
            return
            
        self.worker_running = False
        
        if self.worker_thread:
            # Wait for the thread to finish (with timeout)
            self.worker_thread.join(timeout=10)
            
        self.logger.info("Stopped system controller worker thread")
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register an agent for direct reference.
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        self.agent_references[agent_id] = agent
        self.system_state["active_agent_count"] = len(self.agent_references)
        self.logger.debug(f"Registered agent {agent.name} ({agent_id})")
        
        # If this is a portfolio manager or execution agent, pass the wallet
        if agent.agent_type == "portfolio_manager":
            if hasattr(agent, 'wallet') and not agent.wallet:
                agent.wallet = self.wallet
                self.logger.info(f"Shared wallet with portfolio manager {agent_id}")
        
        elif agent.agent_type == "execution":
            if hasattr(agent, 'wallet') and not agent.wallet:
                agent.wallet = self.wallet
                self.logger.info(f"Shared wallet with execution agent {agent_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status summary.
        
        Returns:
            Dictionary with system status information
        """
        # Count agents by type
        agent_counts = {}
        for agent in self.agent_references.values():
            agent_type = agent.agent_type
            if agent_type not in agent_counts:
                agent_counts[agent_type] = 0
            agent_counts[agent_type] += 1
        
        # Get training status
        training_status = self.get_training_status()
        training_status["next_scheduled"] = datetime.fromtimestamp(
            self.system_state["next_scheduled_training"]
        ).isoformat()
        
        # Calculate system health score (simplified)
        health_score = 100
        
        # Reduce score for consecutive errors
        health_score -= self.system_state["consecutive_errors"] * 10
        
        # Reduce score if not in running state
        if self.system_state["status"] != "running":
            health_score -= 30
        
        # Ensure score is within range
        health_score = max(0, min(100, health_score))
        
        return {
            "status": self.system_state["status"],
            "health_score": health_score,
            "active_agents": self.system_state["active_agent_count"],
            "agent_counts": agent_counts,
            "uptime": self.system_state["metrics"]["uptime"],
            "last_update": datetime.fromtimestamp(self.system_state["last_update"]).isoformat() if self.system_state["last_update"] > 0 else None,
            "training_status": training_status,
            "metrics": self.system_state["metrics"],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_detailed_system_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the system.
        
        Returns:
            Dictionary with detailed system information
        """
        # Get basic status
        basic_status = self.get_system_status()
        
        # Get detailed agent information
        agent_info = {}
        for agent_id, agent in self.agent_references.items():
            agent_info[agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "active": agent.active,
                "last_run": datetime.fromtimestamp(agent.last_run).isoformat() if agent.last_run > 0 else None,
                "last_training": datetime.fromtimestamp(agent.last_training).isoformat() if agent.last_training > 0 else None,
                "training_iterations": agent.training_iterations,
                "performance_metrics": agent.performance_metrics
            }
        
        # Get dependency graph
        dependency_graph = self.agent_dependencies
        
        # Get execution order
        execution_order = self.execution_order
        
        # Get error log
        error_log = self.system_state["error_log"]
        
        return {
            "basic_status": basic_status,
            "agent_info": agent_info,
            "dependency_graph": dependency_graph,
            "execution_order": execution_order,
            "error_log": error_log,
            "system_state": self.system_state,
            "timestamp": datetime.now().isoformat()
        }
    
    def _worker_loop(self, system_data: Dict[str, Any], db_handler: Any) -> None:
        """
        Main worker loop for continuous operation.
        
        Args:
            system_data: Initial system data
            db_handler: Database handler for state persistence
        """
        self.logger.info("Worker loop started")
        
        while self.worker_running:
            start_time = time.time()
            
            try:
                # Update system data if needed
                # In a real system, this would fetch latest market data, etc.
                
                # Execute a system cycle
                results = self.run(system_data)
                
                # Process results if needed
                if "error" in results:
                    self.logger.error(f"Error in system cycle: {results['error']}")
                    
                    # If in error state, wait longer before retrying
                    if self.system_state["status"] == "error":
                        time.sleep(self.recovery_wait_time)
                        continue
                
                # Save state if needed
                if time.time() - self.system_state["last_state_save"] > self.save_state_interval:
                    self._save_system_state(db_handler)
                    self.system_state["last_state_save"] = time.time()
                
            except Exception as e:
                self.logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
            
            # Calculate elapsed time and sleep for the remainder of the interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _execute_agent_workflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent workflow based on dependencies and execution order.
        
        Args:
            data: Input data for the workflow
            
        Returns:
            Results from all agents
        """
        results = {}
        workflow_data = data.copy()
        
        # Ensure we have an execution order
        if not self.execution_order:
            # If no explicit order, create one based on dependencies
            self.execution_order = self._derive_execution_order()
        
        # Execute agents in order
        for agent_id in self.execution_order:
            agent = self.agent_references.get(agent_id)
            
            if not agent or not agent.active:
                continue
                
            # Check if dependencies are met
            dependencies = self.agent_dependencies.get(agent_id, [])
            if not self._check_dependencies(dependencies, results):
                self.logger.warning(f"Dependencies not met for agent {agent.name}, skipping")
                continue
            
            # Prepare input data for the agent
            agent_input = self._prepare_agent_input(agent, workflow_data, results)
            
            # Execute the agent with timeout
            start_time = time.time()
            agent_result = self._execute_agent_with_timeout(agent, agent_input)
            execution_time = time.time() - start_time
            
            # Store the result
            results[agent_id] = {
                "result": agent_result,
                "execution_time": execution_time
            }
            
            # Update workflow data with agent results
            workflow_data[f"agent_{agent_id}_result"] = agent_result
        
        return results
    
    def _execute_agent_with_timeout(self, agent: BaseAgent, agent_input: Dict[str, Any]) -> Any:
        """
        Execute an agent with a timeout.
        
        Args:
            agent: Agent to execute
            agent_input: Input data for the agent
            
        Returns:
            Agent execution result or error
        """
        result = None
        error = None
        
        def target():
            nonlocal result, error
            try:
                result = agent.run(agent_input)
            except Exception as e:
                error = str(e)
        
        # Create and start the thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete with timeout
        thread.join(timeout=self.agent_timeout)
        
        # Check if thread is still alive (timeout occurred)
        if thread.is_alive():
            return {"error": f"Agent execution timed out after {self.agent_timeout} seconds"}
        
        # Check for error
        if error:
            return {"error": error}
        
        return result
    
    def _derive_execution_order(self) -> List[str]:
        """
        Derive execution order based on dependencies.
        
        Returns:
            List of agent IDs in execution order
        """
        order = []
        visited = set()
        
        def visit(agent_id):
            if agent_id in visited:
                return
            
            visited.add(agent_id)
            
            # Visit dependencies first
            for dep_id in self.agent_dependencies.get(agent_id, []):
                visit(dep_id)
            
            order.append(agent_id)
        
        # Visit all agents
        for agent_id in self.agent_references.keys():
            visit(agent_id)
        
        return order
    
    def _check_dependencies(self, dependencies: List[str], results: Dict[str, Any]) -> bool:
        """
        Check if all dependencies have been executed successfully.
        
        Args:
            dependencies: List of dependency agent IDs
            results: Current execution results
            
        Returns:
            True if all dependencies met, False otherwise
        """
        for dep_id in dependencies:
            if dep_id not in results:
                return False
                
            result = results[dep_id]["result"]
            if isinstance(result, dict) and "error" in result:
                return False
                
        return True
    
    def _prepare_agent_input(self, agent: BaseAgent, workflow_data: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for a specific agent based on workflow data and dependencies.
        
        Args:
            agent: The agent to prepare input for
            workflow_data: Common workflow data available to all agents
            results: Results from previously executed agents
            
        Returns:
            Dictionary with input data tailored for the specific agent
        """
        agent_id = agent.agent_id
        agent_type = agent.agent_type
        
        # Base input that all agents receive
        agent_input = {
            "timestamp": workflow_data.get("timestamp", datetime.now().isoformat()),
            "mode": workflow_data.get("mode", "normal")
        }
        
        # Add market data if available
        if "market_data" in workflow_data:
            agent_input["market_data"] = workflow_data["market_data"]
        
        # Add dependencies' results if required
        dependencies = self.agent_dependencies.get(agent_id, [])
        for dep_id in dependencies:
            if dep_id in results:
                agent_input[dep_id] = results[dep_id]
        
        # Add wallet information for relevant agents
        if agent_type in ["portfolio_manager", "execution", "risk_manager"]:
            # Get the latest valuation
            if hasattr(self, 'wallet') and self.wallet:
                # If market data is available, use it to update prices
                price_data = {}
                if "market_data" in workflow_data:
                    for symbol, data in workflow_data["market_data"].items():
                        if isinstance(data, dict) and "ohlcv" in data and len(data["ohlcv"]) > 0:
                            # Extract base currency from symbol
                            if "/" in symbol:
                                base_currency = symbol.split("/")[0]
                                # Use latest close price
                                price_data[base_currency] = data["ohlcv"][-1][4]
                
                valuation = self.wallet.calculate_total_value(price_data)
                agent_input["wallet"] = {
                    "balances": self.wallet.get_all_balances(),
                    "total_value": valuation["total_value"],
                    "holdings": valuation["holdings"]
                }
        
        # Special handling for different agent types
        if agent_type == "portfolio_manager":
            # Add signals from strategy agents
            agent_input["signals"] = []
            for strategy_id in [id for id in results.keys() if "strategy" in id]:
                if "signals" in results[strategy_id]:
                    agent_input["signals"].extend(results[strategy_id]["signals"])
        
        elif agent_type == "execution":
            # Add orders from portfolio manager if available
            if "portfolio_manager" in results and "orders" in results["portfolio_manager"]:
                agent_input["orders"] = results["portfolio_manager"]["orders"]
        
        elif agent_type == "risk_manager":
            # Add portfolio state from portfolio manager
            if "portfolio_manager" in results and "portfolio" in results["portfolio_manager"]:
                agent_input["portfolio"] = results["portfolio_manager"]["portfolio"]
        
        return agent_input
    
    def _process_workflow_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the results from the agent workflow.
        
        Args:
            results: Raw execution results
            
        Returns:
            Processed results
        """
        processed_results = {
            "execution_times": {},
            "error_count": 0,
            "successful_count": 0,
            "signals": [],
            "trades": []
        }
        
        for agent_id, agent_result in results.items():
            # Track execution time
            processed_results["execution_times"][agent_id] = agent_result["execution_time"]
            
            # Count errors and successes
            result = agent_result["result"]
            if isinstance(result, dict) and "error" in result:
                processed_results["error_count"] += 1
            else:
                processed_results["successful_count"] += 1
            
            # Collect trading signals
            if isinstance(result, dict) and "signals" in result:
                processed_results["signals"].extend(result["signals"])
            
            # Collect executed trades
            if isinstance(result, dict) and "trades" in result:
                processed_results["trades"].extend(result["trades"])
        
        # Update system metrics
        self.system_state["metrics"]["total_signals"] += len(processed_results["signals"])
        self.system_state["metrics"]["total_trades"] += len(processed_results["trades"])
        
        return processed_results
    
    def _execute_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute training for all agents.
        
        Args:
            data: Training data
            
        Returns:
            Training results
        """
        training_results = {}
        
        for agent_id, agent in self.agent_references.items():
            if not agent.active:
                continue
                
            try:
                self.logger.info(f"Training agent {agent.name} ({agent_id})")
                success, metrics = agent.train(data)
                
                training_results[agent_id] = {
                    "success": success,
                    "metrics": metrics
                }
                
            except Exception as e:
                self.logger.error(f"Error training agent {agent.name}: {e}", exc_info=True)
                training_results[agent_id] = {
                    "success": False,
                    "error": str(e)
                }
        
        return training_results
    
    def _save_system_state(self, db_handler: Any) -> None:
        """Save the current state of the system."""
        try:
            # In a real implementation, this would save state to a database
            # For the demo, we'll just log it
            if self.config.get("is_demo", False):
                self.logger.info("Demo mode: Skipping system state save")
                return
                
            # Handle missing or invalid db_handler
            if db_handler is None or not hasattr(db_handler, 'save'):
                self.logger.info("No valid database handler for saving state")
                return
                
            # Save our own state
            self.save_state(db_handler)
            
            # Save child agent states
            for child in self.children:
                if hasattr(child, 'save_state'):
                    child.save_state(db_handler)
        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}", exc_info=True)
    
    def _perform_maintenance(self, data: Dict[str, Any]) -> None:
        """
        Perform system maintenance tasks.
        
        Args:
            data: Current system data
        """
        try:
            self.logger.info("Starting system maintenance")
            
            # Example maintenance tasks:
            
            # 1. Clean up error logs
            self.system_state["error_log"] = self.system_state["error_log"][-50:]
            
            # 2. Check and reset any stuck agents
            for agent_id, agent in self.agent_references.items():
                # Check if agent hasn't run in a long time
                if agent.last_run > 0 and time.time() - agent.last_run > 3600:  # 1 hour
                    self.logger.warning(f"Agent {agent.name} appears stuck, resetting")
                    agent.last_run = 0
            
            # 3. Perform database optimization or cleanup if needed
            # (Implementation would depend on the specific database in use)
            
            # 4. Calculate and log system performance metrics
            # (Implementation would depend on the specific metrics needed)
            
            self.logger.info("System maintenance completed")
            
        except Exception as e:
            self.logger.error(f"Error during maintenance: {e}", exc_info=True)
    
    def _should_run_training(self) -> bool:
        """
        Determine if training should be run.
        
        Returns:
            True if training should be run, False otherwise
        """
        # Check if scheduled training time has arrived
        return time.time() >= self.system_state["next_scheduled_training"]
    
    def _should_save_state(self) -> bool:
        """
        Determine if system state should be saved.
        
        Returns:
            True if state should be saved, False otherwise
        """
        return time.time() - self.system_state["last_state_save"] >= self.save_state_interval
    
    def _is_trading_time(self) -> bool:
        """
        Check if current time is within trading hours.
        
        Returns:
            True if within trading hours or if trading hours are disabled
        """
        if not self.trading_hours["enabled"]:
            return True
            
        # Parse trading hours
        try:
            start_time = self.trading_hours["start"]
            end_time = self.trading_hours["end"]
            
            # Convert to hours and minutes
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            # Get current time
            now = datetime.now()
            current_hour, current_minute = now.hour, now.minute
            
            # Check if current time is within range
            current_minutes = current_hour * 60 + current_minute
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            
            # Handle overnight trading sessions
            if start_minutes > end_minutes:
                return current_minutes >= start_minutes or current_minutes <= end_minutes
            else:
                return start_minutes <= current_minutes <= end_minutes
                
        except Exception as e:
            self.logger.error(f"Error checking trading hours: {e}")
            return True  # Default to trading allowed on error
    
    def _is_maintenance_time(self) -> bool:
        """
        Check if current time is within the maintenance window.
        
        Returns:
            True if within maintenance window, False otherwise
        """
        if not self.maintenance_window["enabled"]:
            return False
            
        try:
            # Get the scheduled maintenance day and hour
            maintenance_day = self.maintenance_window["day"]  # 0-6, where 0 is Monday
            maintenance_hour = self.maintenance_window["hour"]
            
            # Get current time
            now = datetime.now()
            current_day = now.weekday()  # 0-6, where 0 is Monday
            current_hour = now.hour
            
            # Check if current time is within maintenance window (1 hour window)
            return (current_day == maintenance_day and current_hour == maintenance_hour)
            
        except Exception as e:
            self.logger.error(f"Error checking maintenance window: {e}")
            return False
    
    def _calculate_next_maintenance(self) -> float:
        """
        Calculate the timestamp for the next maintenance window.
        
        Returns:
            Timestamp of next maintenance window
        """
        if not self.maintenance_window["enabled"]:
            return time.time() + 7 * 24 * 60 * 60  # 1 week from now
            
        try:
            # Get the scheduled maintenance day and hour
            maintenance_day = self.maintenance_window["day"]  # 0-6, where 0 is Monday
            maintenance_hour = self.maintenance_window["hour"]
            
            # Get current time
            now = datetime.now()
            current_day = now.weekday()  # 0-6, where 0 is Monday
            
            # Calculate days until next maintenance
            days_until_maintenance = (maintenance_day - current_day) % 7
            if days_until_maintenance == 0 and now.hour >= maintenance_hour:
                days_until_maintenance = 7
                
            # Create datetime for next maintenance
            next_maintenance = now.replace(
                hour=maintenance_hour,
                minute=0,
                second=0,
                microsecond=0
            ) + timedelta(days=days_until_maintenance)
            
            return next_maintenance.timestamp()
            
        except Exception as e:
            self.logger.error(f"Error calculating next maintenance: {e}")
            return time.time() + 7 * 24 * 60 * 60  # 1 week from now
    
    def _check_emergency_shutdown(self, data: Dict[str, Any]) -> bool:
        """
        Check if emergency shutdown is needed due to excessive drawdown.
        
        Args:
            data: Current system data
            
        Returns:
            True if emergency shutdown is needed, False otherwise
        """
        # Check portfolio data if available
        if "portfolio" in data:
            portfolio = data["portfolio"]
            
            # Check for max drawdown
            if "performance" in portfolio and "current_drawdown_pct" in portfolio["performance"]:
                current_drawdown = portfolio["performance"]["current_drawdown_pct"]
                
                # Trigger emergency shutdown if drawdown exceeds threshold
                if current_drawdown > self.emergency_shutdown_drawdown:
                    return True
        
        return False 