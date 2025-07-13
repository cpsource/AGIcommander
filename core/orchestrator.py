#!/usr/bin/env python3
"""
core/orchestrator.py - Main AGI coordination logic for AGIcommander

This orchestrator manages the interaction between LLM adapters, MCP servers,
and agents to enable autonomous AI-driven development and self-improvement.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from datetime import datetime

from .config import AGIConfig
from .safety import SafetyController
from ..llm_adapters.base import BaseLLMAdapter
from ..servers.base_server import BaseMCPServer
from ..agents.base_agent import BaseAgent


class AGIOrchestrator:
    """
    Main orchestrator for AGIcommander system.
    Coordinates between LLMs, MCP servers, and agents for autonomous operation.
    """
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = AGIConfig(config_path)
        self.safety = SafetyController(self.config)
        self.logger = self._setup_logging()
        
        # Core components
        self.llm_adapters: Dict[str, BaseLLMAdapter] = {}
        self.mcp_servers: Dict[str, BaseMCPServer] = {}
        self.agents: Dict[str, BaseAgent] = {}
        
        # State tracking
        self.is_running = False
        self.current_task = None
        self.learning_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('memory/logs/orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("🚀 Initializing AGIcommander...")
        
        try:
            # Initialize safety systems first
            await self.safety.initialize()
            
            # Load and initialize LLM adapters
            await self._initialize_llm_adapters()
            
            # Start MCP servers
            await self._initialize_mcp_servers()
            
            # Initialize agents
            await self._initialize_agents()
            
            self.logger.info("✅ AGIcommander initialization complete")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _initialize_llm_adapters(self):
        """Initialize LLM adapters based on configuration"""
        self.logger.info("🧠 Initializing LLM adapters...")
        
        for provider_name, provider_config in self.config.llm_providers.items():
            if provider_config.get('enabled', False):
                try:
                    # Dynamic import based on provider name
                    module_name = f"llm_adapters.{provider_name}"
                    class_name = f"{provider_name.title()}Adapter"
                    
                    module = __import__(module_name, fromlist=[class_name])
                    adapter_class = getattr(module, class_name)
                    
                    adapter = adapter_class(provider_config)
                    await adapter.initialize()
                    
                    self.llm_adapters[provider_name] = adapter
                    self.logger.info(f"✅ Initialized {provider_name} adapter")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {provider_name}: {e}")
    
    async def _initialize_mcp_servers(self):
        """Initialize MCP servers based on configuration"""
        self.logger.info("🔧 Initializing MCP servers...")
        
        for server_name, server_config in self.config.mcp_servers.items():
            if server_config.get('enabled', False):
                try:
                    # Dynamic import based on server type
                    server_type = server_config['type']
                    module_path = f"servers.{server_type}"
                    
                    module = __import__(module_path, fromlist=['server'])
                    server_class = getattr(module, 'server')
                    
                    server = server_class(server_config)
                    await server.start()
                    
                    self.mcp_servers[server_name] = server
                    self.logger.info(f"✅ Started {server_name} server")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to start {server_name}: {e}")
    
    async def _initialize_agents(self):
        """Initialize AI agents"""
        self.logger.info("🤖 Initializing agents...")
        
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.get('enabled', False):
                try:
                    # Dynamic import based on agent type
                    agent_type = agent_config['type']
                    module_path = f"agents.{agent_type}"
                    
                    module = __import__(module_path, fromlist=[f"{agent_type.title()}Agent"])
                    agent_class = getattr(module, f"{agent_type.title()}Agent")
                    
                    agent = agent_class(
                        config=agent_config,
                        llm_adapters=self.llm_adapters,
                        mcp_servers=self.mcp_servers,
                        safety_controller=self.safety
                    )
                    
                    self.agents[agent_name] = agent
                    self.logger.info(f"✅ Initialized {agent_name} agent")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {agent_name}: {e}")
    
    async def start_autonomous_mode(self):
        """Start autonomous operation mode"""
        self.logger.info("🧠 Starting autonomous mode...")
        self.is_running = True
        
        try:
            while self.is_running:
                # Main autonomous learning loop
                await self._autonomous_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(self.config.cycle_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Autonomous mode interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in autonomous mode: {e}")
        finally:
            await self.shutdown()
    
    async def _autonomous_cycle(self):
        """Execute one autonomous learning/improvement cycle"""
        cycle_start = datetime.now()
        self.logger.info(f"🔄 Starting autonomous cycle at {cycle_start}")
        
        try:
            # 1. Self-assessment
            await self._perform_self_assessment()
            
            # 2. Identify improvement opportunities
            improvements = await self._identify_improvements()
            
            # 3. Execute improvements (with safety checks)
            if improvements:
                await self._execute_improvements(improvements)
            
            # 4. Learn from results
            await self._learn_from_cycle(cycle_start)
            
        except Exception as e:
            self.logger.error(f"❌ Error in autonomous cycle: {e}")
            # Continue to next cycle rather than crash
    
    async def _perform_self_assessment(self):
        """Perform self-assessment of current capabilities"""
        if 'introspection' in self.mcp_servers:
            introspection_server = self.mcp_servers['introspection']
            assessment = await introspection_server.assess_capabilities()
            self.logger.info(f"📊 Self-assessment: {assessment}")
    
    async def _identify_improvements(self) -> List[Dict[str, Any]]:
        """Identify potential improvements"""
        improvements = []
        
        if 'developer' in self.agents:
            developer_agent = self.agents['developer']
            code_improvements = await developer_agent.suggest_improvements()
            improvements.extend(code_improvements)
        
        if 'learner' in self.agents:
            learner_agent = self.agents['learner']
            learning_opportunities = await learner_agent.identify_learning_goals()
            improvements.extend(learning_opportunities)
        
        return improvements
    
    async def _execute_improvements(self, improvements: List[Dict[str, Any]]):
        """Execute improvements with safety checks"""
        for improvement in improvements:
            # Safety check before execution
            if await self.safety.approve_improvement(improvement):
                try:
                    await self._execute_single_improvement(improvement)
                    self.logger.info(f"✅ Executed improvement: {improvement['type']}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to execute improvement: {e}")
            else:
                self.logger.warning(f"⚠️ Safety controller blocked improvement: {improvement['type']}")
    
    async def _execute_single_improvement(self, improvement: Dict[str, Any]):
        """Execute a single improvement"""
        improvement_type = improvement['type']
        
        if improvement_type == 'code_modification':
            # Use commander server for code modifications
            if 'commander' in self.mcp_servers:
                commander = self.mcp_servers['commander']
                await commander.modify_files(improvement['instructions'], improvement['files'])
        
        elif improvement_type == 'learning':
            # Use research server for learning
            if 'research' in self.mcp_servers:
                research = self.mcp_servers['research']
                await research.learn_topic(improvement['topic'])
        
        elif improvement_type == 'capability_enhancement':
            # Add new capabilities
            await self._add_new_capability(improvement['capability'])
    
    async def _add_new_capability(self, capability: Dict[str, Any]):
        """Add new capability to the system"""
        # This is where the system could modify itself
        # For now, log the intention
        self.logger.info(f"🆕 Would add new capability: {capability}")
    
    async def _learn_from_cycle(self, cycle_start: datetime):
        """Learn from the completed cycle"""
        cycle_end = datetime.now()
        cycle_duration = (cycle_end - cycle_start).total_seconds()
        
        cycle_data = {
            'start_time': cycle_start.isoformat(),
            'end_time': cycle_end.isoformat(),
            'duration': cycle_duration,
            'improvements_attempted': len(self.learning_history),
            'status': 'completed'
        }
        
        self.learning_history.append(cycle_data)
        
        # Store learning in memory system
        if 'memory' in self.mcp_servers:
            memory_server = self.mcp_servers['memory']
            await memory_server.store_learning_cycle(cycle_data)
    
    async def execute_task(self, task_description: str) -> str:
        """Execute a specific task using appropriate agents"""
        self.logger.info(f"🎯 Executing task: {task_description}")
        self.current_task = task_description
        
        try:
            # Determine which agent(s) to use
            agent = await self._select_agent_for_task(task_description)
            
            if agent:
                result = await agent.execute_task(task_description)
                self.logger.info(f"✅ Task completed: {result}")
                return result
            else:
                error_msg = "No suitable agent found for task"
                self.logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error executing task: {e}"
            self.logger.error(error_msg)
            return error_msg
        finally:
            self.current_task = None
    
    async def _select_agent_for_task(self, task_description: str) -> Optional[BaseAgent]:
        """Select the most appropriate agent for a task"""
        # Simple heuristic-based selection
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ['code', 'modify', 'refactor', 'implement']):
            return self.agents.get('developer')
        elif any(keyword in task_lower for keyword in ['research', 'learn', 'study', 'analyze']):
            return self.agents.get('learner')
        else:
            # Default to developer agent
            return self.agents.get('developer')
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        self.logger.info("🛑 Shutting down AGIcommander...")
        self.is_running = False
        
        # Shutdown agents
        for agent in self.agents.values():
            await agent.shutdown()
        
        # Shutdown MCP servers
        for server in self.mcp_servers.values():
            await server.stop()
        
        # Shutdown LLM adapters
        for adapter in self.llm_adapters.values():
            await adapter.shutdown()
        
        self.logger.info("✅ AGIcommander shutdown complete")


# Main entry point for testing
if __name__ == "__main__":
    async def main():
        orchestrator = AGIOrchestrator()
        await orchestrator.initialize()
        
        # Example task execution
        result = await orchestrator.execute_task(
            "Analyze the current codebase and suggest improvements"
        )
        print(f"Result: {result}")
        
        await orchestrator.shutdown()
    
    asyncio.run(main())

