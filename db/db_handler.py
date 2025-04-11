"""
Database handler implementations for the trading system.
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

class DummyDBHandler:
    """
    A simple in-memory database handler for development/testing.
    Uses a dictionary to store data and can optionally persist to disk as JSON.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the dummy database handler.
        
        Args:
            storage_path: Optional path for persisting data to disk
        """
        self.storage = {}
        self.storage_path = storage_path
        self.logger = logging.getLogger('dummy_db')
        
        # Create storage directory if needed
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            self.logger.info(f"Created storage directory: {storage_path}")
        
        # Load existing data if available
        if storage_path and os.path.exists(os.path.join(storage_path, "agent_states.json")):
            self.load()
    
    def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """
        Save agent state to the database.
        
        Args:
            agent_id: Unique identifier for the agent
            state: State dictionary to save
            
        Returns:
            Boolean indicating success
        """
        try:
            # Add timestamp to state
            state_with_meta = {
                "timestamp": datetime.now().isoformat(),
                "data": state
            }
            
            # Save to in-memory dictionary
            self.storage[agent_id] = state_with_meta
            self.logger.debug(f"Saved state for agent {agent_id}")
            
            # Optionally persist to disk
            if self.storage_path:
                self._save_to_disk()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent state: {str(e)}")
            return False
    
    def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load agent state from the database.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Agent state dictionary or None if not found
        """
        if agent_id in self.storage:
            self.logger.debug(f"Loaded state for agent {agent_id}")
            return self.storage[agent_id]["data"]
        
        self.logger.warning(f"No state found for agent {agent_id}")
        return None
    
    def list_agent_states(self) -> List[str]:
        """
        List all agent IDs with saved states.
        
        Returns:
            List of agent IDs
        """
        return list(self.storage.keys())
    
    def load(self) -> bool:
        """
        Load all data from disk.
        
        Returns:
            Boolean indicating success
        """
        if not self.storage_path:
            self.logger.warning("No storage path specified, cannot load data")
            return False
        
        try:
            filepath = os.path.join(self.storage_path, "agent_states.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.storage = json.load(f)
                self.logger.info(f"Loaded {len(self.storage)} agent states from disk")
                return True
            else:
                self.logger.warning(f"No data file found at {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading data from disk: {str(e)}")
            return False
    
    def _save_to_disk(self) -> bool:
        """
        Save all data to disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            filepath = os.path.join(self.storage_path, "agent_states.json")
            with open(filepath, 'w') as f:
                json.dump(self.storage, f, indent=2)
            self.logger.debug(f"Saved {len(self.storage)} agent states to disk")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to disk: {str(e)}")
            return False

class PickleDBHandler:
    """
    A database handler that uses pickle for serialization.
    Better for storing complex Python objects like model weights.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the pickle database handler.
        
        Args:
            storage_path: Path for persisting data to disk
        """
        self.storage = {}
        self.storage_path = storage_path
        self.logger = logging.getLogger('pickle_db')
        
        # Create storage directory if needed
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            self.logger.info(f"Created storage directory: {storage_path}")
        
        # Load existing data if available
        if os.path.exists(os.path.join(storage_path, "agent_states.pkl")):
            self.load()
    
    def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """
        Save agent state to the database.
        
        Args:
            agent_id: Unique identifier for the agent
            state: State dictionary to save
            
        Returns:
            Boolean indicating success
        """
        try:
            # Add timestamp to state
            state_with_meta = {
                "timestamp": datetime.now().isoformat(),
                "data": state
            }
            
            # Save to in-memory dictionary
            self.storage[agent_id] = state_with_meta
            self.logger.debug(f"Saved state for agent {agent_id}")
            
            # Persist to disk
            self._save_to_disk()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent state: {str(e)}")
            return False
    
    def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load agent state from the database.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Agent state dictionary or None if not found
        """
        if agent_id in self.storage:
            self.logger.debug(f"Loaded state for agent {agent_id}")
            return self.storage[agent_id]["data"]
        
        self.logger.warning(f"No state found for agent {agent_id}")
        return None
    
    def list_agent_states(self) -> List[str]:
        """
        List all agent IDs with saved states.
        
        Returns:
            List of agent IDs
        """
        return list(self.storage.keys())
    
    def load(self) -> bool:
        """
        Load all data from disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            filepath = os.path.join(self.storage_path, "agent_states.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.storage = pickle.load(f)
                self.logger.info(f"Loaded {len(self.storage)} agent states from disk")
                return True
            else:
                self.logger.warning(f"No data file found at {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading data from disk: {str(e)}")
            return False
    
    def _save_to_disk(self) -> bool:
        """
        Save all data to disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            filepath = os.path.join(self.storage_path, "agent_states.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(self.storage, f)
            self.logger.debug(f"Saved {len(self.storage)} agent states to disk")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to disk: {str(e)}")
            return False 