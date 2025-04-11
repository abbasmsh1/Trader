class DummyDBHandler:
    """Dummy database handler for storing agent states."""

    def __init__(self):
        self.agent_states = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("DummyDBHandler initialized")

    def save_agent_state(self, agent_id, state):
        """Save agent state to database."""
        self.agent_states[agent_id] = state
        return True

    def load_agent_state(self, agent_id):
        """Load agent state from database."""
        return self.agent_states.get(agent_id, None)

    def list_agent_states(self):
        """List all agent states in database."""
        return list(self.agent_states.keys())

    def load(self, agent_id):
        """Load agent state from database. Alias for load_agent_state."""
        self.logger.debug(f"Loading agent state for {agent_id}")
        return self.load_agent_state(agent_id) 