"""Base abstract class for agents in the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from multi_agent_builder.config import get_llm


class BaseAgent(ABC):
    """Abstract base agent providing common LLM interface and logging structure."""

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.name = name
        self.llm = llm or get_llm(provider=provider, optional=True)

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent task taking current graph state and returning updated state slice.

        Args:
            state: The current LangGraph AgentState dictionary.

        Returns:
            Dictionary with state field updates.
        """
        pass
