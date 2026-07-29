from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    async def rewrite(self, prompt: str) -> dict:
        pass
