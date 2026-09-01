from abc import ABC, abstractmethod

class LLMBackend(ABC):
    """
    Interface implemented by any language-model backend.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,    
    )-> str:
        raise NotImplementedError