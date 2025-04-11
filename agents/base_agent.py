"""
Base Agent - Foundation class for all agents in the trading system.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.shared_memory import SharedMemory

class BaseAgent:
    """
    Base class for all agents in the trading system.
    
    Provides common functionality such as:
    - Unique identification
    - Activation/deactivation
    - State persistence
    - Messaging/communication
    - Planning and execution
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None,
                 agent_type: str = "base"):
        """
        Initialize the base agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
            agent_type: Type identifier for the agent
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.config = config
        self.parent_id = parent_id
        self.agent_type = agent_type
        
        # Status tracking
        self.active = False
        self.initialized = False
        self.last_update = None
        self.creation_time = datetime.now()
        self.error_count = 0
        
        # State tracking
        self.state = {}
        
        # Planning and execution
        self.current_plan = None
        self.plan_step = 0
        self.plan_complete = False
        
        # Set up logging
        self.logger = logging.getLogger(f"agent.{agent_type}.{self.id}")
        
        # Initialize shared memory
        self.shared_memory = SharedMemory()
    
    def plan(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a plan based on the current state and input data.
        To be overridden by subclasses for specific planning logic.
        
        Args:
            data: Input data for planning
            
        Returns:
            Dictionary containing the plan
        """
        # Base implementation - simple plan with one step
        plan = {
            "steps": [
                {
                    "action": "process",
                    "data": data or {},
                    "expected_result": "success"
                }
            ],
            "current_step": 0,
            "total_steps": 1,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        # Store plan in shared memory
        self.shared_memory.store_plan(self.id, plan)
        
        return plan
    
    def execute(self, plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single step of the plan.
        To be overridden by subclasses for specific execution logic.
        
        Args:
            plan: Plan to execute (if None, use current plan)
            
        Returns:
            Dictionary containing execution results
        """
        # Use provided plan or current plan
        if plan is None:
            plan = self.current_plan or self.shared_memory.get_plan(self.id)
            if plan is None:
                return {"success": False, "error": "No plan available"}
        
        # Get current step
        current_step = plan.get("current_step", 0)
        steps = plan.get("steps", [])
        
        # Check if plan is complete
        if current_step >= len(steps):
            return {"success": True, "result": "plan_complete"}
        
        # Get step to execute
        step = steps[current_step]
        
        try:
            # Execute step
            result = self._execute_step(step)
            
            # Update plan
            plan["current_step"] = current_step + 1
            plan["status"] = "in_progress" if current_step + 1 < len(steps) else "complete"
            
            # Store updated plan
            self.shared_memory.store_plan(self.id, plan)
            
            # Record action
            self.shared_memory.record_action(self.id, {
                "step": current_step,
                "action": step.get("action", "unknown"),
                "result": result
            })
            
            return {
                "success": True,
                "step": current_step,
                "result": result,
                "plan_status": plan["status"]
            }
            
        except Exception as e:
            self.logger.error(f"Error executing plan step: {str(e)}")
            return {
                "success": False,
                "step": current_step,
                "error": str(e)
            }
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step of the plan.
        To be overridden by subclasses for specific step execution logic.
        
        Args:
            step: Step to execute
            
        Returns:
            Dictionary containing step execution results
        """
        # Base implementation - just return the step data
        return {"message": "Base agent does not implement step execution", "step": step}
    
    def initialize(self) -> bool:
        """
        Initialize the agent. To be overridden by subclasses.
        
        Returns:
            Boolean indicating success
        """
        self.initialized = True
        return True
    
    def activate(self) -> bool:
        """
        Activate the agent.
        
        Returns:
            Boolean indicating success
        """
        if not self.initialized:
            success = self.initialize()
            if not success:
                self.logger.error(f"Failed to initialize {self.name} during activation")
                return False
        
        self.active = True
        self.logger.info(f"Agent {self.name} activated")
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the agent.
        
        Returns:
            Boolean indicating success
        """
        self.active = False
        self.logger.info(f"Agent {self.name} deactivated")
        return True
    
    def update(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update the agent with new data and trigger processing.
        
        Args:
            data: Input data for the agent to process
            
        Returns:
            Dictionary with results of the update
        """
        if not self.active:
            self.logger.warning(f"Attempted to update inactive agent {self.name}")
            return {"success": False, "error": "Agent inactive"}
        
        try:
            self.last_update = datetime.now()
            
            # Create plan
            plan = self.plan(data)
            
            # Execute plan
            result = self.execute(plan)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error updating agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _process_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an update. To be overridden by subclasses.
        
        Args:
            data: Input data for processing
            
        Returns:
            Results of processing
        """
        # Base implementation does nothing
        return {"message": "Base agent does not implement processing"}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent for persistence.
        
        Returns:
            Dictionary representing agent state
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "active": self.active,
            "initialized": self.initialized,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "creation_time": self.creation_time.isoformat(),
            "error_count": self.error_count,
            "state": self.state,
            "current_plan": self.current_plan,
            "plan_step": self.plan_step,
            "plan_complete": self.plan_complete
        }
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load agent state from persisted data.
        
        Args:
            state: State dictionary to load
            
        Returns:
            Boolean indicating success
        """
        try:
            # Only restore mutable/changeable properties
            if "active" in state:
                self.active = state["active"]
            
            if "initialized" in state:
                self.initialized = state["initialized"]
            
            if "last_update" in state and state["last_update"]:
                self.last_update = datetime.fromisoformat(state["last_update"])
            
            if "error_count" in state:
                self.error_count = state["error_count"]
            
            if "state" in state:
                self.state = state["state"]
            
            if "current_plan" in state:
                self.current_plan = state["current_plan"]
            
            if "plan_step" in state:
                self.plan_step = state["plan_step"]
            
            if "plan_complete" in state:
                self.plan_complete = state["plan_complete"]
            
            self.logger.info(f"Loaded state for agent {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading agent state: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} ({self.agent_type}, {self.id[:8]})" 