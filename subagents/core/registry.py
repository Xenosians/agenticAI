from subagents.core.types import AgentDefinition


class AgentRegistry:
    """
    Stores and retrieves available sub-agent definitions.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        if agent.name in self._agents:
            raise ValueError(
                f"Agent '{agent.name}' is already registered."
            )

        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentDefinition:
        if name not in self._agents:
            raise KeyError(
                f"Agent '{name}' is not registered."
            )

        return self._agents[name]

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def exists(self, name: str) -> bool:
        return name in self._agents
    
    def register_many(
        self, 
        agents: list[AgentDefinition],
    ) -> None:
        for agent in agents:
            self.register(agent)