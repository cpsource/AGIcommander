#!/usr/bin/env python3
"""
llm_adapters/base.py - Base LLM adapter interface

Provides a unified interface for different LLM providers, allowing AGIcommander
to work with multiple AI models through a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging


class ModelCapability(Enum):
    """Enum for different model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    provider: str
    capabilities: List[ModelCapability]
    context_window: int
    cost_per_input_token: float
    cost_per_output_token: float
    max_output_tokens: int


@dataclass
class Message:
    """Standard message format"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CompletionResponse:
    """Standard response format"""
    content: str
    model: str
    usage: Dict[str, int]  # tokens used
    metadata: Dict[str, Any]
    reasoning_tokens: Optional[int] = None


@dataclass
class ToolCall:
    """Tool/function call representation"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    All LLM providers should implement this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_initialized = False
        self.available_models: List[ModelInfo] = []
        self.current_model: Optional[str] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM adapter"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the LLM adapter"""
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages
        
        Args:
            messages: List of messages in the conversation
            model: Specific model to use (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Provider-specific arguments
        
        Returns:
            CompletionResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion
        
        Args:
            messages: List of messages in the conversation
            model: Specific model to use (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific arguments
        
        Yields:
            Partial completion strings
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        pass
    
    def supports_capability(self, capability: ModelCapability, model: Optional[str] = None) -> bool:
        """Check if the adapter/model supports a specific capability"""
        target_model = model or self.current_model
        if not target_model:
            return False
        
        for model_info in self.available_models:
            if model_info.name == target_model:
                return capability in model_info.capabilities
        return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> float:
        """Estimate cost for a completion"""
        target_model = model or self.current_model
        if not target_model:
            return 0.0
        
        for model_info in self.available_models:
            if model_info.name == target_model:
                input_cost = input_tokens * model_info.cost_per_input_token / 1000
                output_cost = output_tokens * model_info.cost_per_output_token / 1000
                return input_cost + output_cost
        return 0.0
    
    async def validate_connection(self) -> bool:
        """Test if the connection to the LLM provider is working"""
        try:
            test_messages = [Message(role="user", content="Test connection")]
            response = await self.complete(test_messages, max_tokens=10)
            return bool(response.content)
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        return self.__class__.__name__.replace("Adapter", "").lower()


class LLMAdapterError(Exception):
    """Base exception for LLM adapter errors"""
    pass


class ModelNotFoundError(LLMAdapterError):
    """Raised when a requested model is not available"""
    pass


class RateLimitError(LLMAdapterError):
    """Raised when hitting rate limits"""
    pass


class AuthenticationError(LLMAdapterError):
    """Raised when authentication fails"""
    pass


class ContextLengthError(LLMAdapterError):
    """Raised when context exceeds model limits"""
    pass


# Utility functions for working with adapters
def count_tokens(text: str) -> int:
    """
    Simple token counting approximation
    Real implementations should use proper tokenizers
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def create_system_message(content: str) -> Message:
    """Create a system message"""
    return Message(role="system", content=content)


def create_user_message(content: str) -> Message:
    """Create a user message"""
    return Message(role="user", content=content)


def create_assistant_message(content: str) -> Message:
    """Create an assistant message"""
    return Message(role="assistant", content=content)

