from subagents.core.registry import AgentRegistry
from subagents.core.types import AgentResult, AgentTask
from subagents.llm.base import LLMBackend

class AgentRuntime:
    """
    Executes tasks using registered specialist agents.
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        llm: LLMBackend,
    ) -> None:
        self.registry = registry
        self.llm = llm
        
    def run(self, task: AgentTask) -> AgentResult:
        """
        Executes the given task using the appropriate specialist agent.
        
        Args:
            task (AgentTask): The task to be executed.
            
        Returns:
            AgentResult: The result of the task execution.
        """
        
        try:
            agent = self.registry.get(task.agent_name)
        except KeyError as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=str(exc),
            )
            
        messages = [
        {
            "role" : "system",
            "content" : agent.system_prompt,
        },
        {
            "role" : "system",
            "content" : task.user_request,
        }
        ]
        
        if task.instructions:
            messages.append(
                {
                    "role" : "user",
                    "content" : (
                        f"Additional instructions:\n" 
                        f"{task.instructions}"
                    )
                }
            )
        
        try:
            response = self.llm.generate(messages)
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=str(exc),
            )
        return AgentResult(
            task_id=task.task_id,
            agent_name=task.agent_name,
            status="success",
            result=response,
        )