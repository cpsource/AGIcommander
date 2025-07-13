#!/usr/bin/env python3
"""
servers/base_server.py - Base MCP server class

Provides the foundation for all MCP servers in the AGIcommander system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseMCPServer(ABC):
    """
    Abstract base class for MCP servers.
    All MCP servers should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Server state
        self.is_running = False
        self.start_time = None
        self.request_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def start(self):
        """Start the MCP server"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the MCP server"""
        pass
    
    async def execute_tool(self, action: str, **kwargs) -> str:
        """
        Execute a tool action (simplified interface for AGIcommander)
        
        Args:
            action: The action/tool to execute
            **kwargs: Arguments for the action
            
        Returns:
            String result of the action
        """
        self.request_count += 1
        
        try:
            result = await self._execute_action(action, **kwargs)
            return result
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Tool execution failed: {e}")
            raise
    
    @abstractmethod
    async def _execute_action(self, action: str, **kwargs) -> str:
        """
        Internal method to execute actions
        Must be implemented by subclasses
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status information"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "name": self.name,
            "running": self.is_running,
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
    
    def _log_start(self):
        """Log server startup"""
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info(f"🚀 {self.name} server started")
    
    def _log_stop(self):
        """Log server shutdown"""
        self.is_running = False
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        self.logger.info(f"🛑 {self.name} server stopped (uptime: {uptime:.1f}s)")


class MockMCPServer(BaseMCPServer):
    """
    Mock MCP server for testing when real servers aren't available
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_type = config.get('type', 'mock')
    
    async def start(self):
        """Start the mock server"""
        self._log_start()
        await asyncio.sleep(0.1)  # Simulate startup time
    
    async def stop(self):
        """Stop the mock server"""
        self._log_stop()
        await asyncio.sleep(0.1)  # Simulate shutdown time
    
    async def _execute_action(self, action: str, **kwargs) -> str:
        """Execute mock actions"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        if action == "analyze_codebase":
            return self._mock_analyze_codebase()
        elif action == "modify_files":
            return self._mock_modify_files(**kwargs)
        elif action == "suggest_improvements":
            return self._mock_suggest_improvements()
        else:
            return f"Mock result for action '{action}' with args: {kwargs}"
    
    def _mock_analyze_codebase(self) -> str:
        """Mock codebase analysis"""
        return """
Codebase Analysis (Mock):
- Total files: 12
- Lines of code: 1,247
- Languages: Python (85%), YAML (10%), Markdown (5%)
- Code quality: Good
- Test coverage: 65%
- Technical debt: Low
- Security issues: None detected

Recommendations:
1. Add more unit tests to reach 80% coverage
2. Consider adding type hints for better code clarity
3. Update documentation for newer modules
        """.strip()
    
    def _mock_modify_files(self, **kwargs) -> str:
        """Mock file modification"""
        instructions = kwargs.get('instructions', 'No instructions provided')
        return f"""
File Modification Results (Mock):
Instructions: {instructions}

Modified files:
- example.py: Added docstrings and type hints
- utils.py: Refactored helper functions
- config.py: Updated configuration validation

Changes applied successfully. Backups created with .backup extension.
        """.strip()
    
    def _mock_suggest_improvements(self) -> str:
        """Mock improvement suggestions"""
        return """
Code Improvement Suggestions (Mock):

High Priority:
1. Add error handling to network requests
2. Implement proper logging throughout the application

Medium Priority:
3. Refactor large functions into smaller, more focused ones
4. Add input validation for user-facing functions

Low Priority:
5. Consider using dataclasses for configuration objects
6. Add performance monitoring for critical paths
        """.strip()

