from subagents.core.registry import AgentRegistry

class Router:
    """
    Deterministically routes user requests to one or more specialist agents.
    """
    
    ACCOUNT_KEYWORDS = {
        "account",
        "locked",
        "lock",
        "unlock",
        "password",
        "enabled",
        "disabled",
    }
    
    ACCESS_KEYWORDS = {
        "access",
        "vpn",
        "permission",
        "permissions",
        "group",
        "autherization",
        "authorize",
    }
    
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        
    def route(self, user_request: str) -> list[str]:
        """
        Routes the user request to one or more specialist agents based on keywords.
        
        Args:
            user_request (str): The user's request.
        """
        text = user_request.lower()
        
        routes: list[str] = []
        
        if any(keyword in text for keyword in self.ACCOUNT_KEYWORDS):
            if self.registry.exists("account-specialist"):
                routes.append("account-specialist")
                
        if any(keyword in text for keyword in self.ACCESS_KEYWORDS):
            if self.registry.exists("access-specialist"):
                routes.append("access-specialist")
                
        return routes
    
        