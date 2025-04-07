"""
Agent Manager Module - Manages the hierarchy of trading agents in the system.

This module provides the AgentManager class that handles agent creation,
organization, and coordination in a hierarchical structure.
"""
import logging
from typing import Dict, Any, List, Optional, Type, Tuple
import time
import json
import importlib

from agents.base_agent import BaseAgent

class AgentManager:
    """
    Manages the hierarchy of trading agents in the system.
    
    Responsible for:
    - Creating and registering agents
    - Organizing agents in a hierarchical structure
    - Distributing data to agents
    - Collecting and aggregating agent decisions
    - Coordinating agent training
    """
    
    def __init__(self, db_handler, config: Dict[str, Any]):
        """
        Initialize the agent manager.
        
        Args:
            db_handler: Database handler for persistence
            config: Configuration parameters
        """
        self.logger = logging.getLogger("agent_manager")
        self.db_handler = db_handler
        self.config = config
        
        # Root agents (top of the hierarchy)
        self.root_agents: Dict[str, BaseAgent] = {}
        
        # Registry of all agents by ID for quick lookup
        self.agent_registry: Dict[str, BaseAgent] = {}
        
        # Registry of agent types and their classes
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        self.logger.info("Agent Manager initialized")
    
    def register_agent_type(self, type_name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent type for later instantiation.
        
        Args:
            type_name: Name of the agent type
            agent_class: Class of the agent type
        """
        self.agent_types[type_name] = agent_class
        self.logger.info(f"Registered agent type: {type_name}")
    
    def register_agent_types_from_modules(self, modules: List[str]) -> None:
        """
        Register agent types from a list of module names.
        
        Args:
            modules: List of module names containing agent classes
        """
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseAgent) and attr != BaseAgent:
                        type_name = attr_name.lower()
                        self.register_agent_type(type_name, attr)
            except ImportError as e:
                self.logger.error(f"Failed to import module {module_name}: {e}")
    
    def create_agent(self, 
                     agent_type: str, 
                     name: str, 
                     description: str,
                     config: Dict[str, Any],
                     parent_id: Optional[str] = None) -> Optional[BaseAgent]:
        """
        Create a new agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            name: Name of the agent
            description: Description of the agent
            config: Configuration for the agent
            parent_id: ID of parent agent if any
            
        Returns:
            The created agent or None if type not found
        """
        if agent_type not in self.agent_types:
            self.logger.error(f"Agent type {agent_type} not registered")
            return None
            
        agent_class = self.agent_types[agent_type]
        agent = agent_class(name=name, 
                           description=description,
                           agent_type=agent_type,
                           config=config,
                           parent_id=parent_id)
        
        # Register the agent
        self.agent_registry[agent.id] = agent
        
        # Add to hierarchy
        if parent_id:
            parent = self.agent_registry.get(parent_id)
            if parent:
                parent.add_child(agent)
            else:
                self.logger.warning(f"Parent agent {parent_id} not found for {name}")
                self.root_agents[agent.id] = agent
        else:
            # No parent, it's a root agent
            self.root_agents[agent.id] = agent
            
        self.logger.info(f"Created agent: {name} ({agent.id}) of type {agent_type}")
        return agent
    
    def create_agent_tree(self, tree_config: Dict[str, Any]) -> Optional[BaseAgent]:
        """
        Create a tree of agents based on a configuration dictionary.
        
        Args:
            tree_config: Dictionary describing the agent tree
            
        Returns:
            The root agent of the created tree or None if failed
        """
        if "type" not in tree_config or "name" not in tree_config:
            self.logger.error("Agent tree config missing required fields")
            return None
            
        # Create the root agent
        root = self.create_agent(
            agent_type=tree_config["type"],
            name=tree_config["name"],
            description=tree_config.get("description", ""),
            config=tree_config.get("config", {})
        )
        
        if not root:
            return None
            
        # Create children recursively
        children = tree_config.get("children", [])
        for child_config in children:
            self._create_subtree(child_config, root.id)
            
        return root
    
    def _create_subtree(self, tree_config: Dict[str, Any], parent_id: str) -> Optional[BaseAgent]:
        """
        Create a subtree of agents recursively.
        
        Args:
            tree_config: Dictionary describing the agent subtree
            parent_id: ID of the parent agent
            
        Returns:
            The root agent of the created subtree or None if failed
        """
        if "type" not in tree_config or "name" not in tree_config:
            self.logger.error("Agent subtree config missing required fields")
            return None
            
        # Create the node agent
        node = self.create_agent(
            agent_type=tree_config["type"],
            name=tree_config["name"],
            description=tree_config.get("description", ""),
            config=tree_config.get("config", {}),
            parent_id=parent_id
        )
        
        if not node:
            return None
            
        # Create children recursively
        children = tree_config.get("children", [])
        for child_config in children:
            self._create_subtree(child_config, node.id)
            
        return node
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            The agent or None if not found
        """
        return self.agent_registry.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to get
            
        Returns:
            List of agents of the specified type
        """
        return [agent for agent in self.agent_registry.values() 
                if agent.agent_type == agent_type]
    
    def get_root_agents(self) -> List[BaseAgent]:
        """
        Get all root agents.
        
        Returns:
            List of root agents
        """
        return list(self.root_agents.values())
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent and all its children.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if agent was deleted, False if not found
        """
        agent = self.agent_registry.get(agent_id)
        if not agent:
            return False
            
        # Remove from parent if any
        if agent.parent_id:
            parent = self.agent_registry.get(agent.parent_id)
            if parent:
                parent.remove_child(agent_id)
                
        # Remove all children recursively
        for child in agent.children.copy():
            self.delete_agent(child.id)
            
        # Remove from registry
        del self.agent_registry[agent_id]
        
        # Remove from root agents if it's a root
        if agent_id in self.root_agents:
            del self.root_agents[agent_id]
            
        self.logger.info(f"Deleted agent: {agent.name} ({agent_id})")
        return True
    
    def run_agent(self, agent_id: str, data: Any) -> Any:
        """
        Run a specific agent with the given data.
        
        Args:
            agent_id: ID of the agent to run
            data: Data to process
            
        Returns:
            Agent's output or None if agent not found
        """
        agent = self.agent_registry.get(agent_id)
        if not agent or not agent.active:
            return None
            
        try:
            start_time = time.time()
            result = agent.run(data)
            execution_time = time.time() - start_time
            
            # Update agent state
            agent.last_run = time.time()
            
            self.logger.debug(f"Agent {agent.name} ({agent_id}) executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            self.logger.error(f"Error running agent {agent_id}: {e}", exc_info=True)
            return None
    
    def run_agent_tree(self, root_id: str, data: Any) -> Dict[str, Any]:
        """
        Run an agent and all its descendants with the given data.
        
        Args:
            root_id: ID of the root agent to run
            data: Data to process
            
        Returns:
            Dictionary mapping agent IDs to their outputs
        """
        results = {}
        root = self.agent_registry.get(root_id)
        if not root or not root.active:
            return results
            
        # Run the root agent
        results[root_id] = self.run_agent(root_id, data)
        
        # Run all child agents with the root's output
        for child in root.children:
            if child.active:
                child_results = self.run_agent_tree(child.id, results[root_id])
                results.update(child_results)
                
        return results
    
    def run_all_roots(self, data: Any) -> Dict[str, Any]:
        """
        Run all root agents with the given data.
        
        Args:
            data: Data to process
            
        Returns:
            Dictionary mapping agent IDs to their outputs
        """
        results = {}
        for root_id, root in self.root_agents.items():
            if root.active:
                root_results = self.run_agent_tree(root_id, data)
                results.update(root_results)
                
        return results
    
    def train_agent(self, agent_id: str, training_data: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Train a specific agent with the given data.
        
        Args:
            agent_id: ID of the agent to train
            training_data: Data to use for training
            
        Returns:
            Tuple of (success, training_metrics)
        """
        agent = self.agent_registry.get(agent_id)
        if not agent or not agent.active:
            return False, {}
            
        try:
            start_time = time.time()
            success, metrics = agent.train(training_data)
            training_time = time.time() - start_time
            
            if success:
                agent.record_training()
                
            self.logger.info(f"Agent {agent.name} ({agent_id}) trained in {training_time:.2f}s with result: {success}")
            return success, metrics
        except Exception as e:
            self.logger.error(f"Error training agent {agent_id}: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def train_agent_tree(self, root_id: str, training_data: Any) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """
        Train an agent and all its descendants with the given data.
        
        Args:
            root_id: ID of the root agent to train
            training_data: Data to use for training
            
        Returns:
            Dictionary mapping agent IDs to their training results
        """
        results = {}
        root = self.agent_registry.get(root_id)
        if not root or not root.active:
            return results
            
        # Train the root agent
        results[root_id] = self.train_agent(root_id, training_data)
        
        # Train all child agents
        for child in root.children:
            if child.active:
                child_results = self.train_agent_tree(child.id, training_data)
                results.update(child_results)
                
        return results
    
    def train_all_agents(self, training_data: Any) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """
        Train all active agents with the given data.
        
        Args:
            training_data: Data to use for training
            
        Returns:
            Dictionary mapping agent IDs to their training results
        """
        results = {}
        for agent_id, agent in self.agent_registry.items():
            if agent.active:
                results[agent_id] = self.train_agent(agent_id, training_data)
                
        return results
    
    def save_agents_state(self) -> None:
        """Save the state of all agents."""
        for agent in self.agent_registry.values():
            agent.save_state(self.db_handler)
            
        self.logger.info(f"Saved state for {len(self.agent_registry)} agents")
    
    def load_agents_state(self) -> int:
        """
        Load the state of all agents.
        
        Returns:
            Number of agents with successfully loaded state
        """
        loaded_count = 0
        for agent in self.agent_registry.values():
            if agent.load_state(self.db_handler):
                loaded_count += 1
                
        self.logger.info(f"Loaded state for {loaded_count} agents")
        return loaded_count
    
    def get_all_training_status(self) -> List[Dict[str, Any]]:
        """
        Get training status for all agents.
        
        Returns:
            List of dictionaries with training status
        """
        return [agent.get_training_status() for agent in self.agent_registry.values()]
    
    def get_agent_hierarchy(self) -> Dict[str, Any]:
        """
        Get the agent hierarchy as a nested dictionary.
        
        Returns:
            Dictionary representing the agent hierarchy
        """
        hierarchy = {}
        
        for root_id, root in self.root_agents.items():
            hierarchy[root_id] = self._build_hierarchy_dict(root)
            
        return hierarchy
    
    def _build_hierarchy_dict(self, agent: BaseAgent) -> Dict[str, Any]:
        """
        Build a dictionary representing the hierarchy under an agent.
        
        Args:
            agent: The agent to build the hierarchy for
            
        Returns:
            Dictionary representing the agent and its descendants
        """
        agent_dict = {
            "id": agent.id,
            "name": agent.name,
            "type": agent.agent_type,
            "active": agent.active,
            "children": {}
        }
        
        for child in agent.children:
            agent_dict["children"][child.id] = self._build_hierarchy_dict(child)
            
        return agent_dict
    
    def export_agents_config(self) -> Dict[str, Any]:
        """
        Export the configuration of all agents.
        
        Returns:
            Dictionary with agent configurations
        """
        config = {
            "root_agents": [],
            "agent_registry": {}
        }
        
        # Export root agents
        for root_id, root in self.root_agents.items():
            config["root_agents"].append(root_id)
            
        # Export all agents
        for agent_id, agent in self.agent_registry.items():
            config["agent_registry"][agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "type": agent.agent_type,
                "parent_id": agent.parent_id,
                "config": agent.config
            }
            
        return config
    
    def import_agents_config(self, config: Dict[str, Any]) -> int:
        """
        Import agent configurations and recreate them.
        
        Args:
            config: Dictionary with agent configurations
            
        Returns:
            Number of agents created
        """
        # Clear current agents
        self.root_agents = {}
        self.agent_registry = {}
        
        created_count = 0
        agent_registry = config.get("agent_registry", {})
        
        # First pass: create all agents without parent relationships
        for agent_id, agent_config in agent_registry.items():
            if agent_config["type"] not in self.agent_types:
                self.logger.warning(f"Agent type {agent_config['type']} not registered, skipping")
                continue
                
            agent_class = self.agent_types[agent_config["type"]]
            agent = agent_class(
                name=agent_config["name"],
                description=agent_config["description"],
                agent_type=agent_config["type"],
                config=agent_config.get("config", {})
            )
            
            # Override the ID
            agent.id = agent_id
            
            # Register the agent
            self.agent_registry[agent_id] = agent
            created_count += 1
            
        # Second pass: set up parent-child relationships
        for agent_id, agent_config in agent_registry.items():
            agent = self.agent_registry.get(agent_id)
            if not agent:
                continue
                
            parent_id = agent_config.get("parent_id")
            if parent_id:
                parent = self.agent_registry.get(parent_id)
                if parent:
                    agent.parent_id = parent_id
                    parent.children.append(agent)
                else:
                    # Parent not found, make it a root
                    self.root_agents[agent_id] = agent
            else:
                # No parent, it's a root agent
                self.root_agents[agent_id] = agent
                
        self.logger.info(f"Imported {created_count} agents")
        return created_count 