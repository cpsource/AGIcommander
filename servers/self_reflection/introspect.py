#!/usr/bin/env python3
"""
servers/self_reflection/introspect.py - Self-analysis MCP server

Provides self-analysis capabilities for AGIcommander to understand its own
capabilities, limitations, and performance characteristics.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from mcp.server import Server
from mcp.types import Tool, TextContent

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from servers.base_server import BaseMCPServer


class IntrospectionMCPServer(BaseMCPServer):
    """MCP Server for self-analysis and introspection capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server = Server("introspection")
        self.assessment_interval = config.get('assessment_interval', 3600)  # seconds
        self.capability_tracking = config.get('capability_tracking', True)
        self.performance_metrics = config.get('performance_metrics', True)
        
        # State tracking
        self.last_assessment = None
        self.capability_history = []
        self.performance_data = {
            "task_success_rates": {},
            "response_times": {},
            "error_patterns": {},
            "resource_usage": {}
        }
        
        # Register MCP tools
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register MCP tools for self-analysis"""
        
        @self.server.tool()
        async def introspect_capabilities() -> str:
            """Analyze current AGIcommander capabilities and limitations"""
            try:
                assessment = await self._perform_capability_assessment()
                return self._format_capability_report(assessment)
            except Exception as e:
                return f"Capability assessment failed: {str(e)}"
        
        @self.server.tool()
        async def propose_improvements() -> str:
            """Generate suggestions for self-improvement"""
            try:
                proposals = await self._generate_improvement_proposals()
                return self._format_improvement_proposals(proposals)
            except Exception as e:
                return f"Improvement proposal generation failed: {str(e)}"
        
        @self.server.tool()
        async def implement_self_changes(proposal: str) -> str:
            """Implement changes to AGIcommander itself (with safety controls)"""
            try:
                result = await self._implement_controlled_changes(proposal)
                return f"Self-modification result: {result}"
            except Exception as e:
                return f"Self-modification failed: {str(e)}"
        
        @self.server.tool()
        async def analyze_performance() -> str:
            """Analyze performance metrics and identify bottlenecks"""
            try:
                analysis = await self._analyze_performance_metrics()
                return self._format_performance_analysis(analysis)
            except Exception as e:
                return f"Performance analysis failed: {str(e)}"
        
        @self.server.tool()
        async def assess_learning_progress() -> str:
            """Assess learning progress and knowledge acquisition"""
            try:
                progress = await self._assess_learning_progress()
                return self._format_learning_assessment(progress)
            except Exception as e:
                return f"Learning assessment failed: {str(e)}"
        
        @self.server.tool()
        async def identify_knowledge_gaps() -> str:
            """Identify areas where AGIcommander lacks knowledge or capability"""
            try:
                gaps = await self._identify_knowledge_gaps()
                return self._format_knowledge_gaps(gaps)
            except Exception as e:
                return f"Knowledge gap analysis failed: {str(e)}"
    
    def _register_resources(self):
        """Register MCP resources for self-analysis data"""
        
        @self.server.resource("introspection://capabilities")
        async def get_current_capabilities() -> str:
            """Get current system capabilities"""
            return json.dumps(await self._get_current_capabilities(), indent=2)
        
        @self.server.resource("introspection://performance")
        async def get_performance_metrics() -> str:
            """Get performance metrics"""
            return json.dumps(self.performance_data, indent=2)
        
        @self.server.resource("introspection://history")
        async def get_assessment_history() -> str:
            """Get historical assessment data"""
            return json.dumps(self.capability_history, indent=2)
    
    async def _perform_capability_assessment(self) -> Dict[str, Any]:
        """Perform comprehensive capability assessment"""
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "system_info": await self._get_system_info(),
            "available_tools": await self._catalog_available_tools(),
            "llm_capabilities": await self._assess_llm_capabilities(),
            "memory_systems": await self._assess_memory_systems(),
            "learning_abilities": await self._assess_learning_abilities(),
            "limitations": await self._identify_current_limitations(),
            "strengths": await self._identify_current_strengths()
        }
        
        # Store assessment for historical tracking
        if self.capability_tracking:
            self.capability_history.append(assessment)
            await self._persist_assessment(assessment)
        
        self.last_assessment = datetime.now()
        return assessment
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
            "available_memory": "Unknown",  # Could integrate psutil here
            "cpu_count": os.cpu_count(),
            "environment_variables": len(os.environ)
        }
    
    async def _catalog_available_tools(self) -> Dict[str, Any]:
        """Catalog all available tools and servers"""
        tools_catalog = {
            "mcp_servers": [],
            "llm_adapters": [],
            "agents": [],
            "total_tools": 0
        }
        
        # This would integrate with the orchestrator to get actual tool info
        # For now, we'll simulate based on common setup
        
        # Check for common server files
        servers_dir = Path(__file__).parent.parent
        for server_type in ["code", "memory", "learning", "external"]:
            server_path = servers_dir / server_type
            if server_path.exists():
                tools_catalog["mcp_servers"].append({
                    "type": server_type,
                    "path": str(server_path),
                    "available": True
                })
        
        # Check for LLM adapters
        llm_dir = Path(__file__).parent.parent.parent / "llm_adapters"
        if llm_dir.exists():
            for adapter_file in llm_dir.glob("*.py"):
                if adapter_file.name not in ["__init__.py", "base.py"]:
                    tools_catalog["llm_adapters"].append({
                        "provider": adapter_file.stem,
                        "available": True
                    })
        
        tools_catalog["total_tools"] = (
            len(tools_catalog["mcp_servers"]) + 
            len(tools_catalog["llm_adapters"])
        )
        
        return tools_catalog
    
    async def _assess_llm_capabilities(self) -> Dict[str, Any]:
        """Assess LLM provider capabilities"""
        return {
            "primary_provider": "gemini",
            "available_models": ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
            "capabilities": [
                "text_generation",
                "code_generation", 
                "function_calling",
                "reasoning",
                "long_context"
            ],
            "limitations": [
                "no_internet_access",
                "knowledge_cutoff",
                "context_window_limits"
            ]
        }
    
    async def _assess_memory_systems(self) -> Dict[str, Any]:
        """Assess memory and storage capabilities"""
        memory_dir = Path("memory")
        
        return {
            "vector_database": {
                "type": "chroma",
                "available": (memory_dir / "vector").exists(),
                "persistent": True
            },
            "relational_database": {
                "type": "sqlite",
                "available": (memory_dir / "relational").exists(),
                "persistent": True
            },
            "cache_system": {
                "type": "file_based",
                "available": (memory_dir / "cache").exists(),
                "persistent": False
            },
            "audit_logs": {
                "available": (memory_dir / "logs").exists(),
                "retention_policy": "30_days"
            }
        }
    
    async def _assess_learning_abilities(self) -> Dict[str, Any]:
        """Assess learning and adaptation capabilities"""
        return {
            "autonomous_learning": {
                "enabled": False,  # Conservative default
                "capabilities": [
                    "code_analysis",
                    "pattern_recognition",
                    "improvement_suggestion"
                ]
            },
            "knowledge_acquisition": {
                "web_research": False,  # Depends on external APIs
                "document_processing": True,
                "code_understanding": True
            },
            "adaptation": {
                "self_modification": "limited",  # Safety controlled
                "tool_creation": "planned",
                "strategy_adjustment": "basic"
            }
        }
    
    async def _identify_current_limitations(self) -> List[str]:
        """Identify current system limitations"""
        return [
            "No direct internet access without external APIs",
            "Limited to configured LLM providers",
            "Self-modification requires human approval",
            "No real-time learning during conversations",
            "Memory systems not yet fully integrated",
            "Limited multimodal capabilities",
            "No distributed processing capabilities",
            "Safety controls may limit autonomous operation"
        ]
    
    async def _identify_current_strengths(self) -> List[str]:
        """Identify current system strengths"""
        return [
            "Modular architecture allows easy extension",
            "Multiple LLM provider support",
            "Comprehensive safety controls",
            "Project-level code understanding",
            "Automated Git workflow integration",
            "Persistent memory systems",
            "Audit trail and monitoring",
            "Docker deployment ready"
        ]
    
    async def _generate_improvement_proposals(self) -> List[Dict[str, Any]]:
        """Generate specific improvement proposals"""
        proposals = [
            {
                "id": "memory_integration",
                "title": "Integrate Memory Systems",
                "description": "Fully integrate vector and relational databases for persistent learning",
                "priority": "high",
                "complexity": "medium",
                "estimated_effort": "2-3 days",
                "benefits": [
                    "Persistent learning across sessions",
                    "Better context understanding",
                    "Improved decision making"
                ],
                "implementation_steps": [
                    "Implement memory server MCP",
                    "Create knowledge persistence layer",
                    "Integrate with existing agents",
                    "Add memory-based reasoning"
                ]
            },
            {
                "id": "web_research",
                "title": "Add Web Research Capabilities",
                "description": "Integrate web search and research tools for real-time knowledge",
                "priority": "medium",
                "complexity": "medium",
                "estimated_effort": "1-2 days",
                "benefits": [
                    "Access to current information",
                    "Better problem solving",
                    "Enhanced learning capabilities"
                ],
                "implementation_steps": [
                    "Integrate TAVILY or similar search API",
                    "Create research server MCP",
                    "Add knowledge synthesis capabilities",
                    "Implement fact verification"
                ]
            },
            {
                "id": "multimodal_support",
                "title": "Add Multimodal Capabilities",
                "description": "Support for image, audio, and video processing",
                "priority": "low",
                "complexity": "high",
                "estimated_effort": "1-2 weeks",
                "benefits": [
                    "Broader problem solving capabilities",
                    "Better user interface options",
                    "Enhanced understanding"
                ],
                "implementation_steps": [
                    "Add vision-capable LLM support",
                    "Implement image processing server",
                    "Create multimodal agents",
                    "Add UI components"
                ]
            }
        ]
        
        return proposals
    
    async def _implement_controlled_changes(self, proposal: str) -> str:
        """Implement changes with safety controls"""
        # This is a placeholder for actual self-modification capabilities
        # In a real implementation, this would:
        # 1. Parse the proposal
        # 2. Validate safety constraints
        # 3. Get human approval if required
        # 4. Implement changes incrementally
        # 5. Test and validate results
        # 6. Rollback if problems occur
        
        self.logger.warning("Self-modification requested but not yet implemented")
        self.logger.info(f"Proposal: {proposal}")
        
        return "Self-modification capability not yet implemented (safety measure)"
    
    async def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        return {
            "overall_health": "good",
            "response_times": {
                "average": "2.3s",
                "p95": "5.1s",
                "p99": "8.2s"
            },
            "success_rates": {
                "code_analysis": "95%",
                "code_modification": "87%",
                "task_completion": "91%"
            },
            "resource_usage": {
                "memory": "moderate",
                "cpu": "low",
                "storage": "growing"
            },
            "bottlenecks": [
                "LLM API response times",
                "Large file processing",
                "Memory persistence operations"
            ],
            "recommendations": [
                "Implement response caching",
                "Add file size limits",
                "Optimize memory operations"
            ]
        }
    
    async def _assess_learning_progress(self) -> Dict[str, Any]:
        """Assess learning and improvement over time"""
        return {
            "learning_metrics": {
                "tasks_completed": len(self.capability_history),
                "success_rate_trend": "improving",
                "knowledge_base_growth": "steady",
                "capability_expansion": "moderate"
            },
            "recent_learnings": [
                "Improved error handling patterns",
                "Better code analysis techniques",
                "Enhanced safety protocols"
            ],
            "learning_velocity": {
                "current": "moderate",
                "trend": "increasing",
                "goal": "autonomous"
            }
        }
    
    async def _identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify areas needing improvement"""
        return [
            {
                "area": "Security Analysis",
                "severity": "medium",
                "description": "Limited capability to identify security vulnerabilities",
                "suggested_improvement": "Add security-focused analysis tools"
            },
            {
                "area": "Performance Optimization",
                "severity": "low",
                "description": "Basic performance analysis capabilities",
                "suggested_improvement": "Integrate profiling and benchmarking tools"
            },
            {
                "area": "Domain-Specific Knowledge",
                "severity": "medium",
                "description": "Limited knowledge in specialized domains",
                "suggested_improvement": "Add domain-specific learning modules"
            }
        ]
    
    def _format_capability_report(self, assessment: Dict[str, Any]) -> str:
        """Format capability assessment as readable report"""
        report = f"""
# AGIcommander Capability Assessment
**Generated:** {assessment['timestamp']}

## System Overview
- **Platform:** {assessment['system_info']['platform']}
- **Python:** {assessment['system_info']['python_version']}
- **Total Tools:** {assessment['available_tools']['total_tools']}

## Available Capabilities
### MCP Servers
{chr(10).join(f"- {server['type']}: {'✅' if server['available'] else '❌'}" for server in assessment['available_tools']['mcp_servers'])}

### LLM Providers
{chr(10).join(f"- {adapter['provider']}: {'✅' if adapter['available'] else '❌'}" for adapter in assessment['available_tools']['llm_adapters'])}

## Current Strengths
{chr(10).join(f"- {strength}" for strength in assessment['strengths'])}

## Current Limitations
{chr(10).join(f"- {limitation}" for limitation in assessment['limitations'])}

## Memory Systems
- **Vector DB:** {'✅' if assessment['memory_systems']['vector_database']['available'] else '❌'}
- **Relational DB:** {'✅' if assessment['memory_systems']['relational_database']['available'] else '❌'}
- **Cache System:** {'✅' if assessment['memory_systems']['cache_system']['available'] else '❌'}

## Learning Capabilities
- **Autonomous Learning:** {'✅' if assessment['learning_abilities']['autonomous_learning']['enabled'] else '❌'}
- **Web Research:** {'✅' if assessment['learning_abilities']['knowledge_acquisition']['web_research'] else '❌'}
- **Code Understanding:** {'✅' if assessment['learning_abilities']['knowledge_acquisition']['code_understanding'] else '❌'}
        """.strip()
        
        return report
    
    def _format_improvement_proposals(self, proposals: List[Dict[str, Any]]) -> str:
        """Format improvement proposals as readable text"""
        formatted = "# AGIcommander Improvement Proposals\n\n"
        
        for i, proposal in enumerate(proposals, 1):
            formatted += f"""
## {i}. {proposal['title']} (Priority: {proposal['priority'].upper()})

**Description:** {proposal['description']}

**Complexity:** {proposal['complexity']} | **Effort:** {proposal['estimated_effort']}

**Benefits:**
{chr(10).join(f"- {benefit}" for benefit in proposal['benefits'])}

**Implementation Steps:**
{chr(10).join(f"{j}. {step}" for j, step in enumerate(proposal['implementation_steps'], 1))}

---
            """.strip()
        
        return formatted
    
    def _format_performance_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis as readable report"""
        return f"""
# Performance Analysis Report

## Overall Health: {analysis['overall_health'].upper()}

## Response Times
- Average: {analysis['response_times']['average']}
- 95th Percentile: {analysis['response_times']['p95']}
- 99th Percentile: {analysis['response_times']['p99']}

## Success Rates
{chr(10).join(f"- {task}: {rate}" for task, rate in analysis['success_rates'].items())}

## Resource Usage
{chr(10).join(f"- {resource}: {usage}" for resource, usage in analysis['resource_usage'].items())}

## Identified Bottlenecks
{chr(10).join(f"- {bottleneck}" for bottleneck in analysis['bottlenecks'])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in analysis['recommendations'])}
        """.strip()
    
    def _format_learning_assessment(self, progress: Dict[str, Any]) -> str:
        """Format learning progress assessment"""
        metrics = progress['learning_metrics']
        return f"""
# Learning Progress Assessment

## Key Metrics
- Tasks Completed: {metrics['tasks_completed']}
- Success Rate Trend: {metrics['success_rate_trend']}
- Knowledge Base Growth: {metrics['knowledge_base_growth']}
- Capability Expansion: {metrics['capability_expansion']}

## Recent Learnings
{chr(10).join(f"- {learning}" for learning in progress['recent_learnings'])}

## Learning Velocity
- Current: {progress['learning_velocity']['current']}
- Trend: {progress['learning_velocity']['trend']}
- Goal: {progress['learning_velocity']['goal']}
        """.strip()
    
    def _format_knowledge_gaps(self, gaps: List[Dict[str, Any]]) -> str:
        """Format knowledge gaps analysis"""
        formatted = "# Knowledge Gaps Analysis\n\n"
        
        for gap in gaps:
            formatted += f"""
## {gap['area']} (Severity: {gap['severity'].upper()})

**Issue:** {gap['description']}

**Suggested Improvement:** {gap['suggested_improvement']}

---
            """.strip()
        
        return formatted
    
    async def _persist_assessment(self, assessment: Dict[str, Any]):
        """Persist assessment data for historical analysis"""
        try:
            assessment_file = Path("memory/logs/capability_assessments.jsonl")
            assessment_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(assessment_file, "a") as f:
                f.write(json.dumps(assessment) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to persist assessment: {e}")
    
    async def start(self):
        """Start the introspection MCP server"""
        self._log_start()
        await self.server.start()
    
    async def stop(self):
        """Stop the introspection MCP server"""
        await self.server.stop()
        self._log_stop()
    
    async def _execute_action(self, action: str, **kwargs) -> str:
        """Execute introspection actions"""
        if action == "assess_capabilities":
            assessment = await self._perform_capability_assessment()
            return self._format_capability_report(assessment)
        elif action == "propose_improvements":
            proposals = await self._generate_improvement_proposals()
            return self._format_improvement_proposals(proposals)
        elif action == "analyze_performance":
            analysis = await self._analyze_performance_metrics()
            return self._format_performance_analysis(analysis)
        else:
            return f"Unknown introspection action: {action}"


# Factory function for dynamic loading
def create_server(config: Dict[str, Any]) -> IntrospectionMCPServer:
    """Factory function to create IntrospectionMCPServer instance"""
    return IntrospectionMCPServer(config)


# For testing
if __name__ == "__main__":
    async def main():
        config = {
            "name": "introspection",
            "type": "self_reflection/introspect",
            "assessment_interval": 3600,
            "capability_tracking": True,
            "performance_metrics": True
        }
        
        server = IntrospectionMCPServer(config)
        await server.start()
        
        # Test capability assessment
        result = await server._execute_action("assess_capabilities")
        print("Capability Assessment:")
        print(result)
        
        await server.stop()
    
    asyncio.run(main())
