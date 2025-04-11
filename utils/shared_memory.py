"""
Shared memory for storing agent states, actions, and plans.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class SharedMemory:
    """Shared memory for storing agent states, actions, and plans."""
    
    def __init__(self):
        """Initialize the shared memory."""
        self.logger = logging.getLogger(__name__)
        
        # Store agent states
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Store agent actions
        self.agent_actions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Store agent plans
        self.agent_plans: Dict[str, Dict[str, Any]] = {}
        
        # Store system-wide data
        self.system_data: Dict[str, Any] = {
            "last_update": None,
            "active_agents": [],
            "system_state": "initializing"
        }
    
    def update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Update the state of an agent.
        
        Args:
            agent_id: ID of the agent
            state: New state dictionary
        """
        self.agent_states[agent_id] = state
        self.logger.debug(f"Updated state for agent {agent_id}")
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent state dictionary or None if not found
        """
        return self.agent_states.get(agent_id)
    
    def record_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """
        Record an action taken by an agent.
        
        Args:
            agent_id: ID of the agent
            action: Action dictionary
        """
        if agent_id not in self.agent_actions:
            self.agent_actions[agent_id] = []
        
        # Add timestamp to action
        action["timestamp"] = datetime.now().isoformat()
        
        # Store action
        self.agent_actions[agent_id].append(action)
        
        # Keep only last 1000 actions
        if len(self.agent_actions[agent_id]) > 1000:
            self.agent_actions[agent_id] = self.agent_actions[agent_id][-1000:]
        
        self.logger.debug(f"Recorded action for agent {agent_id}")
    
    def get_agent_actions(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent actions taken by an agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of actions to return
            
        Returns:
            List of action dictionaries
        """
        if agent_id not in self.agent_actions:
            return []
        
        return self.agent_actions[agent_id][-limit:]
    
    def store_plan(self, agent_id: str, plan: Dict[str, Any]) -> None:
        """
        Store a plan for an agent.
        
        Args:
            agent_id: ID of the agent
            plan: Plan dictionary
        """
        self.agent_plans[agent_id] = plan
        self.logger.debug(f"Stored plan for agent {agent_id}")
    
    def get_plan(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current plan for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Plan dictionary or None if not found
        """
        return self.agent_plans.get(agent_id)
    
    def update_system_data(self, data: Dict[str, Any]) -> None:
        """
        Update system-wide data.
        
        Args:
            data: Dictionary of system data
        """
        self.system_data.update(data)
        self.system_data["last_update"] = datetime.now().isoformat()
        self.logger.debug("Updated system data")
    
    def get_system_data(self) -> Dict[str, Any]:
        """
        Get current system-wide data.
        
        Returns:
            Dictionary of system data
        """
        return self.system_data
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.agent_states.clear()
        self.agent_actions.clear()
        self.agent_plans.clear()
        self.system_data = {
            "last_update": None,
            "active_agents": [],
            "system_state": "initializing"
        }
        self.logger.info("Cleared shared memory") 