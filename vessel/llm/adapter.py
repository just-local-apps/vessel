from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    model: str

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...
