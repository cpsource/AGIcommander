#!/usr/bin/env python3
"""
llm_adapters/gemini.py - Google Gemini LLM adapter

Implements the BaseLLMAdapter interface for Google's Gemini models,
integrating your existing Gemini functionality into the AGIcommander framework.
"""

import os
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from .base import (
    BaseLLMAdapter, Message, CompletionResponse, ModelInfo, ModelCapability,
    LLMAdapterError, AuthenticationError, RateLimitError, count_tokens
)


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini LLM adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv("GOOGLE_API_KEY")
        self.default_model = config.get('default_model', 'gemini-2.0-flash-exp')
        self.llm_instances: Dict[str, ChatGoogleGenerativeAI] = {}
        
        if not self.api_key:
            raise AuthenticationError("Google API key not found in config or environment")
    
    async def initialize(self) -> None:
        """Initialize the Gemini adapter"""
        self.logger.info("Initializing Gemini adapter...")
        
        try:
            # Initialize available models
            await self._setup_available_models()
            
            # Initialize default model
            self.current_model = self.default_model
            await self._initialize_model(self.default_model)
            
            # Test connection
            if await self.validate_connection():
                self.is_initialized = True
                self.logger.info("✅ Gemini adapter initialized successfully")
            else:
                raise AuthenticationError("Failed to validate Gemini connection")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini adapter: {e}")
            raise LLMAdapterError(f"Gemini initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the Gemini adapter"""
        self.logger.info("Shutting down Gemini adapter...")
        self.llm_instances.clear()
        self.is_initialized = False
    
    async def _setup_available_models(self) -> None:
        """Setup the list of available Gemini models"""
        self.available_models = [
            ModelInfo(
                name="gemini-2.0-flash-exp",
                provider="google",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.REASONING,
                    ModelCapability.LONG_CONTEXT
                ],
                context_window=2000000,  # 2M tokens
                cost_per_input_token=0.000125,  # Per 1K tokens
                cost_per_output_token=0.000375,  # Per 1K tokens
                max_output_tokens=8192
            ),
            ModelInfo(
                name="gemini-1.5-pro",
                provider="google",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.LONG_CONTEXT
                ],
                context_window=2000000,  # 2M tokens
                cost_per_input_token=0.00125,   # Per 1K tokens
                cost_per_output_token=0.00375,  # Per 1K tokens
                max_output_tokens=8192
            ),
            ModelInfo(
                name="gemini-1.5-flash",
                provider="google",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION
                ],
                context_window=1000000,  # 1M tokens
                cost_per_input_token=0.000075,  # Per 1K tokens
                cost_per_output_token=0.0003,   # Per 1K tokens
                max_output_tokens=8192
            )
        ]
    
    async def _initialize_model(self, model_name: str) -> ChatGoogleGenerativeAI:
        """Initialize a specific model instance"""
        if model_name not in self.llm_instances:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                self.llm_instances[model_name] = llm
                self.logger.info(f"Initialized {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {model_name}: {e}")
                raise
        
        return self.llm_instances[model_name]
    
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion using Gemini"""
        
        if not self.is_initialized:
            raise LLMAdapterError("Adapter not initialized")
        
        target_model = model or self.current_model
        llm = await self._initialize_model(target_model)
        
        # Update temperature if different
        if temperature != 0.1:
            llm.temperature = temperature
        
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)
            
            # Execute completion
            response = await asyncio.to_thread(llm.invoke, langchain_messages)
            
            # Calculate token usage (approximation)
            input_text = "\n".join([msg.content for msg in messages])
            input_tokens = count_tokens(input_text)
            output_tokens = count_tokens(response.content)
            
            return CompletionResponse(
                content=response.content,
                model=target_model,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                metadata={
                    "provider": "google",
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
        except Exception as e:
            self.logger.error(f"Completion failed: {e}")
            
            # Handle specific error types
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif "auth" in str(e).lower() or "key" in str(e).lower():
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise LLMAdapterError(f"Completion failed: {e}")
    
    async def stream_complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion using Gemini"""
        
        if not self.is_initialized:
            raise LLMAdapterError("Adapter not initialized")
        
        target_model = model or self.current_model
        llm = await self._initialize_model(target_model)
        
        # Update temperature if different
        if temperature != 0.1:
            llm.temperature = temperature
        
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)
            
            # For now, we'll simulate streaming by chunking the response
            # LangChain's Google integration doesn't have native streaming yet
            response = await asyncio.to_thread(llm.invoke, langchain_messages)
            
            # Simulate streaming by yielding chunks
            content = response.content
            chunk_size = 50  # characters per chunk
            
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
                
        except Exception as e:
            self.logger.error(f"Streaming completion failed: {e}")
            raise LLMAdapterError(f"Streaming failed: {e}")
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available Gemini models"""
        return self.available_models.copy()
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Gemini model"""
        for model_info in self.available_models:
            if model_info.name == model_name:
                return model_info
        return None
    
    def _convert_messages(self, messages: List[Message]) -> List:
        """Convert AGIcommander messages to LangChain format"""
        langchain_messages = []
        
        for msg in messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            else:
                # Default to human message for unknown roles
                langchain_messages.append(HumanMessage(content=msg.content))
        
        return langchain_messages
    
    async def create_code_completion(
        self,
        instructions: str,
        files_data: Dict[str, tuple],
        model: Optional[str] = None
    ) -> str:
        """
        Create a code completion using the original Commander format
        This method maintains compatibility with existing Commander functionality
        """
        
        # Build the prompt in Commander format
        prompt = f"""You are a skilled developer tasked with modifying multiple files according to specific instructions.

INSTRUCTIONS:
{instructions}

FILES TO PROCESS:
"""
        
        for filename, (content, language) in files_data.items():
            if language:
                prompt += f"\n---{filename}---\n```{language}\n{content}\n```\n"
            else:
                prompt += f"\n---{filename}---\n```\n{content}\n```\n"
        
        prompt += """

RESPONSE FORMAT:
Please return the modified files in the exact same format, with each file preceded by its filename marker and appropriate code fencing:
---filename.ext---
```language
[modified code here]
```

Only return files that need to be changed. If a file doesn't need modification, don't include it in your response.
Ensure all code is syntactically correct and follows best practices for the respective language.
"""
        
        # Create messages and get completion
        messages = [
            Message(role="system", content="You are an expert developer who carefully modifies code according to instructions."),
            Message(role="user", content=prompt)
        ]
        
        response = await self.complete(messages, model=model)
        return response.content


# Factory function for dynamic loading
def create_adapter(config: Dict[str, Any]) -> GeminiAdapter:
    """Factory function to create GeminiAdapter instance"""
    return GeminiAdapter(config)


# For testing
if __name__ == "__main__":
    async def main():
        config = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "default_model": "gemini-2.0-flash-exp"
        }
        
        adapter = GeminiAdapter(config)
        await adapter.initialize()
        
        # Test basic completion
        messages = [
            Message(role="user", content="Write a simple Python function to calculate fibonacci numbers")
        ]
        
        response = await adapter.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        
        await adapter.shutdown()
    
    asyncio.run(main())

