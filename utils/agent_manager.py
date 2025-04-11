"""
Agent Manager - Handles creating and managing trading system agents.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Callable

class AgentManager:
    """
    Manages agent creation, registration, and lifecycle.
    """
    
    def __init__(self, db_handler, config: Dict[str, Any]):
        """
        Initialize the agent manager.
        
        Args:
            db_handler: Database handler for agent state persistence
            config: System configuration dictionary
        """
        self.db_handler = db_handler
        self.config = config
        self.agent_types = {}  # Map of agent type names to classes
        self.agent_instances = {}  # Map of agent IDs to instances
        
        self.logger = logging.getLogger('agent_manager')
        self.logger.info("Agent Manager initialized")
    
    def register_agent_type(self, agent_type: str, agent_class: Type) -> None:
        """
        Register an agent type with the manager.
        
        Args:
            agent_type: String identifier for the agent type
            agent_class: Class reference for the agent type
        """
        self.agent_types[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")
    
    def create_agent(self, agent_type: str, name: str, description: str, 
                    config: Dict[str, Any], parent_id: Optional[str] = None) -> Optional[Any]:
        """
        Create a new agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to create
            name: Name for the new agent
            description: Description of the agent
            config: Configuration for the agent
            parent_id: Optional ID of parent agent
            
        Returns:
            New agent instance or None if creation failed
        """
        if agent_type not in self.agent_types:
            self.logger.error(f"Unknown agent type: {agent_type}")
            return None
        
        try:
            # Create the agent instance
            agent_class = self.agent_types[agent_type]
            agent = agent_class(
                name=name,
                description=description,
                config=config,
                parent_id=parent_id
            )
            
            # Store the instance
            self.agent_instances[agent.id] = agent
            self.logger.info(f"Created agent: {name} ({agent.id})")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating agent {name}: {str(e)}")
            return None
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self.agent_instances.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agent instances.
        
        Returns:
            List of agent details dictionaries
        """
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type if hasattr(agent, "agent_type") else "unknown",
                "status": "active" if agent.active else "inactive" if hasattr(agent, "active") else "unknown"
            }
            for agent in self.agent_instances.values()
        ]
    
    def save_all_states(self) -> bool:
        """
        Save the state of all agents.
        
        Returns:
            Boolean indicating success
        """
        success = True
        for agent_id, agent in self.agent_instances.items():
            if hasattr(agent, "get_state") and callable(agent.get_state):
                try:
                    state = agent.get_state()
                    if not self.db_handler.save_agent_state(agent_id, state):
                        success = False
                except Exception as e:
                    self.logger.error(f"Error saving state for agent {agent_id}: {str(e)}")
                    success = False
        
        return success
    
    def load_all_states(self) -> bool:
        """
        Load the state of all agents.
        
        Returns:
            Boolean indicating success
        """
        success = True
        for agent_id, agent in self.agent_instances.items():
            if hasattr(agent, "load_state") and callable(agent.load_state):
                try:
                    state = self.db_handler.load_agent_state(agent_id)
                    if state:
                        if not agent.load_state(state):
                            success = False
                except Exception as e:
                    self.logger.error(f"Error loading state for agent {agent_id}: {str(e)}")
                    success = False
        
        return success
    
    def activate_all(self) -> None:
        """Activate all agents."""
        for agent in self.agent_instances.values():
            if hasattr(agent, "activate") and callable(agent.activate):
                agent.activate()
    
    def deactivate_all(self) -> None:
        """Deactivate all agents."""
        for agent in self.agent_instances.values():
            if hasattr(agent, "deactivate") and callable(agent.deactivate):
                agent.deactivate() 