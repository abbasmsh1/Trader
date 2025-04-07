"""
Base Agent Module - Foundation for all trading agents in the system.

This module provides the abstract base class that all agents must inherit from,
ensuring a consistent interface across different agent types.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List, Tuple
import uuid
import time
import json

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents in the system.
    
    Defines the common interface and functionality that all agents must implement.
    Handles agent identification, state management, and logging.
    """
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 agent_type: str,
                 config: Dict[str, Any],
                 parent_id: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Human-readable name of the agent
            description: Short description of the agent's purpose
            agent_type: Type classification of the agent
            config: Configuration parameters for the agent
            parent_id: ID of the parent agent if part of a hierarchy
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.config = config
        self.parent_id = parent_id
        self.children: List[BaseAgent] = []
        
        # Agent state
        self.active = True
        self.created_at = time.time()
        self.last_run = 0
        self.last_training = 0
        self.training_iterations = 0
        
        # Performance metrics
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "total_trades": 0,
            "profitable_trades": 0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{self.agent_type}.{self.id}")

    def add_child(self, agent: 'BaseAgent') -> None:
        """
        Add a child agent to this agent.
        
        Args:
            agent: The child agent to add
        """
        agent.parent_id = self.id
        self.children.append(agent)
        self.logger.info(f"Added child agent {agent.name} ({agent.id})")
    
    def remove_child(self, agent_id: str) -> bool:
        """
        Remove a child agent from this agent.
        
        Args:
            agent_id: ID of the child agent to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        for i, agent in enumerate(self.children):
            if agent.id == agent_id:
                self.children.pop(i)
                self.logger.info(f"Removed child agent {agent.name} ({agent.id})")
                return True
        return False
    
    def get_child(self, agent_id: str) -> Optional['BaseAgent']:
        """
        Get a child agent by ID.
        
        Args:
            agent_id: ID of the child agent
            
        Returns:
            The child agent if found, None otherwise
        """
        for agent in self.children:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_all_descendants(self) -> List['BaseAgent']:
        """
        Get all descendant agents recursively.
        
        Returns:
            List of all descendant agents
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def activate(self) -> None:
        """Activate this agent and log the event."""
        self.active = True
        self.logger.info(f"Agent {self.name} activated")
    
    def deactivate(self) -> None:
        """Deactivate this agent and log the event."""
        self.active = False
        self.logger.info(f"Agent {self.name} deactivated")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the agent's configuration.
        
        Args:
            new_config: New configuration parameters to apply
        """
        self.config.update(new_config)
        self.logger.info(f"Updated configuration for agent {self.name}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "parent_id": self.parent_id,
            "active": self.active,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "last_training": self.last_training,
            "training_iterations": self.training_iterations,
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "children": [child.id for child in self.children]
        }
    
    def save_state(self, storage_handler) -> None:
        """
        Save the agent's state using the provided storage handler.
        
        Args:
            storage_handler: Object with a save method to persist the state
        """
        state = self.get_state()
        storage_handler.save(f"agent_{self.id}", state)
        self.logger.debug(f"Saved state for agent {self.name}")
    
    def load_state(self, storage_handler) -> bool:
        """
        Load the agent's state using the provided storage handler.
        
        Args:
            storage_handler: Object with a load method to retrieve the state
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        state = storage_handler.load(f"agent_{self.id}")
        if not state:
            return False
            
        # Update fields from loaded state
        self.active = state.get("active", self.active)
        self.last_run = state.get("last_run", self.last_run)
        self.last_training = state.get("last_training", self.last_training)
        self.training_iterations = state.get("training_iterations", self.training_iterations)
        self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
        self.config = state.get("config", self.config)
        
        self.logger.debug(f"Loaded state for agent {self.name}")
        return True
    
    def record_training(self) -> None:
        """Record that a training iteration has occurred."""
        self.last_training = time.time()
        self.training_iterations += 1
        self.logger.info(f"Completed training iteration {self.training_iterations} for agent {self.name}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get the training status of the agent.
        
        Returns:
            Dictionary with training information
        """
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "last_training": self.last_training,
            "training_iterations": self.training_iterations,
            "time_since_last_training": time.time() - self.last_training if self.last_training > 0 else None
        }
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update the agent's performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics to update
        """
        self.performance_metrics.update(metrics)
    
    @abstractmethod
    def run(self, data: Any) -> Any:
        """
        Execute the agent's logic based on provided data.
        Must be implemented by all concrete agent classes.
        
        Args:
            data: Input data for the agent to process
            
        Returns:
            Agent's output after processing the data
        """
        pass
    
    @abstractmethod
    def train(self, training_data: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Train the agent using the provided data.
        Must be implemented by all concrete agent classes.
        
        Args:
            training_data: Data to use for training
            
        Returns:
            Tuple of (success, training_metrics)
        """
        pass 