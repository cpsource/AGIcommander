#!/usr/bin/env python3
"""
agents/base_agent.py - Base agent class for AGIcommander

Provides the foundation for specialized AI agents that can perform
specific tasks using LLM adapters and MCP servers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class BaseAgent(ABC):
    """
    Abstract base class for AGIcommander agents.
    All specialized agents should inherit from this class.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_adapters: Dict[str, Any],
        mcp_servers: Dict[str, Any],
        safety_controller: Any
    ):
        self.config = config
        self.llm_adapters = llm_adapters
        self.mcp_servers = mcp_servers
        self.safety_controller = safety_controller
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Agent configuration
        self.primary_llm = config.get('primary_llm', 'gemini')
        self.fallback_llm = config.get('fallback_llm', 'gemini')
        self.available_tools = config.get('tools', [])
        self.max_iterations = config.get('max_iterations', 5)
        
        # State tracking
        self.current_task = None
        self.task_history = []
        self.performance_metrics = {}
    
    @abstractmethod
    async def execute_task(self, task_description: str) -> str:
        """
        Execute a specific task
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            String result of the task execution
        """
        pass
    
    @abstractmethod
    async def suggest_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements within this agent's domain
        
        Returns:
            List of improvement suggestions
        """
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info(f"Shutting down {self.__class__.__name__}")
        # Base implementation - can be overridden
        pass
    
    async def _get_llm_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """Get a response from the LLM"""
        from ..llm_adapters.base import Message
        
        llm_name = model or self.primary_llm
        
        if llm_name not in self.llm_adapters:
            self.logger.warning(f"LLM {llm_name} not available, using fallback")
            llm_name = self.fallback_llm
        
        if llm_name not in self.llm_adapters:
            raise RuntimeError(f"No available LLM adapters")
        
        try:
            adapter = self.llm_adapters[llm_name]
            messages = [Message(role="user", content=prompt)]
            
            response = await adapter.complete(
                messages=messages,
                temperature=temperature
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise
    
    async def _use_tool(self, tool_name: str, **kwargs) -> str:
        """Use a specific MCP server tool"""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available to this agent")
        
        if tool_name not in self.mcp_servers:
            raise RuntimeError(f"MCP server {tool_name} not running")
        
        try:
            server = self.mcp_servers[tool_name]
            # This is a simplified interface - real implementation would
            # use proper MCP protocol
            result = await server.execute_tool(**kwargs)
            return result
            
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    def _log_task_start(self, task_description: str):
        """Log the start of a task"""
        self.current_task = {
            "description": task_description,
            "start_time": datetime.now(),
            "iterations": 0,
            "status": "running"
        }
        
        self.logger.info(f"🎯 Starting task: {task_description}")
    
    def _log_task_end(self, result: str, success: bool = True):
        """Log the end of a task"""
        if self.current_task:
            self.current_task.update({
                "end_time": datetime.now(),
                "result": result,
                "success": success,
                "status": "completed" if success else "failed"
            })
            
            duration = (
                self.current_task["end_time"] - self.current_task["start_time"]
            ).total_seconds()
            
            self.task_history.append(self.current_task.copy())
            
            status = "✅" if success else "❌"
            self.logger.info(
                f"{status} Task completed in {duration:.1f}s: "
                f"{self.current_task['description']}"
            )
            
            self.current_task = None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        if not self.task_history:
            return {"total_tasks": 0, "success_rate": 0.0}
        
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for task in self.task_history if task["success"])
        success_rate = successful_tasks / total_tasks
        
        # Calculate average duration
        durations = []
        for task in self.task_history:
            if "end_time" in task and "start_time" in task:
                duration = (task["end_time"] - task["start_time"]).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "recent_tasks": self.task_history[-5:]  # Last 5 tasks
        }

