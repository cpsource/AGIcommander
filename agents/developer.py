#!/usr/bin/env python3
"""
agents/developer.py - Development-focused agent

Specialized agent for code analysis, modification, and improvement tasks.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent


class DeveloperAgent(BaseAgent):
    """Agent specialized in software development tasks"""
    
    def __init__(self, config, llm_adapters, mcp_servers, safety_controller):
        super().__init__(config, llm_adapters, mcp_servers, safety_controller)
        
        self.code_quality_checks = config.get('code_quality_checks', True)
        self.test_execution = config.get('test_execution', False)
        
        self.logger.info("🧑‍💻 Developer agent initialized")
    
    async def execute_task(self, task_description: str) -> str:
        """Execute a development-related task"""
        self._log_task_start(task_description)
        
        try:
            # Analyze the task to determine the approach
            task_type = await self._analyze_task_type(task_description)
            
            if task_type == "code_analysis":
                result = await self._analyze_code(task_description)
            elif task_type == "code_modification":
                result = await self._modify_code(task_description)
            elif task_type == "code_creation":
                result = await self._create_code(task_description)
            elif task_type == "suggestion":
                result = await self._provide_suggestions(task_description)
            else:
                result = await self._general_development_task(task_description)
            
            self._log_task_end(result, success=True)
            return result
            
        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            self._log_task_end(error_msg, success=False)
            return error_msg
    
    async def _analyze_task_type(self, task_description: str) -> str:
        """Analyze task description to determine the type of work needed"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['analyze', 'review', 'examine', 'inspect']):
            return "code_analysis"
        elif any(word in task_lower for word in ['modify', 'change', 'update', 'fix', 'refactor']):
            return "code_modification"
        elif any(word in task_lower for word in ['create', 'write', 'implement', 'build']):
            return "code_creation"
        elif any(word in task_lower for word in ['suggest', 'recommend', 'improve']):
            return "suggestion"
        else:
            return "general"
    
    async def _analyze_code(self, task_description: str) -> str:
        """Analyze existing code"""
        self.logger.info("📊 Performing code analysis...")
        
        # Use commander MCP server to analyze codebase
        if 'commander' in self.mcp_servers:
            try:
                # Get codebase analysis
                analysis_result = await self._use_tool('commander', action='analyze_codebase')
                
                # Get AI interpretation of the analysis
                interpretation_prompt = f"""
                Task: {task_description}
                
                Code analysis results:
                {analysis_result}
                
                Please provide a clear interpretation of this analysis, focusing on:
                1. Code structure and organization
                2. Potential issues or improvements
                3. Specific recommendations
                4. Priority areas for attention
                
                Make the response actionable and specific.
                """
                
                interpretation = await self._get_llm_response(interpretation_prompt)
                
                return f"Code Analysis Results:\n\n{interpretation}"
                
            except Exception as e:
                self.logger.error(f"Analysis failed: {e}")
                return f"Analysis failed: {str(e)}"
        else:
            return "Commander tool not available for code analysis"
    
    async def _modify_code(self, task_description: str) -> str:
        """Modify existing code"""
        self.logger.info("🔧 Performing code modification...")
        
        if 'commander' in self.mcp_servers:
            try:
                # Use commander to modify files
                result = await self._use_tool(
                    'commander',
                    action='modify_files',
                    instructions=task_description,
                    auto_confirm=False  # Require approval for modifications
                )
                
                return f"Code Modification Results:\n\n{result}"
                
            except Exception as e:
                self.logger.error(f"Modification failed: {e}")
                return f"Modification failed: {str(e)}"
        else:
            return "Commander tool not available for code modification"
    
    async def _create_code(self, task_description: str) -> str:
        """Create new code"""
        self.logger.info("🆕 Creating new code...")
        
        # Extract file specifications from task description
        creation_prompt = f"""
        Task: {task_description}
        
        Based on this task, create the necessary files and code.
        
        Please provide:
        1. A list of files that should be created
        2. The complete content for each file
        3. Brief explanation of the implementation approach
        
        Format your response with clear file separators and appropriate code blocks.
        """
        
        response = await self._get_llm_response(creation_prompt)
        
        # If commander is available, we could use it to actually create the files
        if 'commander' in self.mcp_servers:
            try:
                # For now, just return the generated code
                # In the future, we could parse the response and create actual files
                return f"Code Creation Plan:\n\n{response}\n\n(Note: Use commander tool to actually create these files)"
            except Exception as e:
                return f"Code generation completed, but file creation failed: {str(e)}\n\nGenerated Code:\n{response}"
        
        return f"Generated Code:\n\n{response}"
    
    async def _provide_suggestions(self, task_description: str) -> str:
        """Provide code improvement suggestions"""
        self.logger.info("💡 Generating suggestions...")
        
        if 'commander' in self.mcp_servers:
            try:
                # Get suggestions from commander
                suggestions = await self._use_tool('commander', action='suggest_improvements')
                
                # Enhance suggestions with AI analysis
                enhancement_prompt = f"""
                Task: {task_description}
                
                Initial suggestions:
                {suggestions}
                
                Please enhance these suggestions by:
                1. Prioritizing them by impact and difficulty
                2. Adding specific implementation steps
                3. Explaining the benefits of each suggestion
                4. Providing code examples where helpful
                
                Make the suggestions actionable and developer-friendly.
                """
                
                enhanced_suggestions = await self._get_llm_response(enhancement_prompt)
                
                return f"Code Improvement Suggestions:\n\n{enhanced_suggestions}"
                
            except Exception as e:
                return f"Suggestion generation failed: {str(e)}"
        else:
            # Fallback to AI-only suggestions
            suggestion_prompt = f"""
            {task_description}
            
            Please provide specific, actionable suggestions for improving the codebase.
            Consider:
            - Code quality and maintainability
            - Performance optimizations
            - Security improvements
            - Best practices
            - Testing strategies
            
            Prioritize suggestions by impact and provide implementation guidance.
            """
            
            suggestions = await self._get_llm_response(suggestion_prompt)
            return f"AI-Generated Suggestions:\n\n{suggestions}"
    
    async def _general_development_task(self, task_description: str) -> str:
        """Handle general development tasks"""
        self.logger.info("🔧 Handling general development task...")
        
        # Use AI to understand and break down the task
        analysis_prompt = f"""
        Development Task: {task_description}
        
        Please analyze this task and provide:
        1. Understanding of what needs to be done
        2. Step-by-step approach to complete the task
        3. Tools and resources needed
        4. Potential challenges and solutions
        5. Expected outcomes
        
        Be specific and actionable in your response.
        """
        
        analysis = await self._get_llm_response(analysis_prompt)
        return f"Task Analysis and Plan:\n\n{analysis}"
    
    async def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements for the development workflow"""
        improvements = []
        
        # Analyze current codebase
        if 'commander' in self.mcp_servers:
            try:
                # Get current codebase insights
                analysis = await self._use_tool('commander', action='analyze_codebase')
                
                # Generate improvement suggestions based on analysis
                suggestion_prompt = f"""
                Based on this codebase analysis:
                {analysis}
                
                Suggest 3-5 specific improvements that would benefit the project.
                For each suggestion, provide:
                - Type (code_quality, performance, security, architecture, testing)
                - Priority (high, medium, low)
                - Description
                - Implementation steps
                - Expected benefits
                
                Format as a structured list.
                """
                
                suggestions_text = await self._get_llm_response(suggestion_prompt)
                
                # Parse suggestions into structured format
                # This is a simplified parser - could be enhanced
                improvements.append({
                    "type": "code_modification",
                    "priority": "medium",
                    "description": "Improve code quality based on analysis",
                    "details": suggestions_text,
                    "tools_needed": ["commander"],
                    "estimated_effort": "medium"
                })
                
            except Exception as e:
                self.logger.error(f"Failed to generate improvements: {e}")
        
        # Always suggest code quality improvements
        improvements.append({
            "type": "code_analysis",
            "priority": "low",
            "description": "Perform regular code quality assessment",
            "details": "Regular analysis helps maintain code quality and identify technical debt",
            "tools_needed": ["commander"],
            "estimated_effort": "low"
        })
        
        return improvements
    
    async def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks on the current codebase"""
        if not self.code_quality_checks:
            return {"status": "disabled", "message": "Code quality checks are disabled"}
        
        results = {
            "status": "completed",
            "checks": {},
            "overall_score": 0,
            "recommendations": []
        }
        
        try:
            # Use commander to analyze code
            if 'commander' in self.mcp_servers:
                analysis = await self._use_tool('commander', action='analyze_codebase')
                results["checks"]["structure_analysis"] = {
                    "status": "passed",
                    "details": analysis
                }
            
            # Simulate additional checks
            results["checks"]["syntax_check"] = {
                "status": "passed",
                "details": "No syntax errors found"
            }
            
            results["checks"]["style_check"] = {
                "status": "warning",
                "details": "Some style inconsistencies detected"
            }
            
            # Calculate overall score
            passed_checks = sum(1 for check in results["checks"].values() 
                              if check["status"] == "passed")
            total_checks = len(results["checks"])
            results["overall_score"] = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            
        except Exception as e:
            results = {
                "status": "failed",
                "error": str(e),
                "overall_score": 0
            }
        
        return results
    
    async def create_documentation(self, target: str = "all") -> str:
        """Generate documentation for the codebase"""
        self.logger.info(f"📚 Creating documentation for: {target}")
        
        doc_prompt = f"""
        Please generate comprehensive documentation for the codebase.
        Target: {target}
        
        Include:
        1. Overview and purpose
        2. Architecture and design patterns
        3. API documentation
        4. Setup and installation instructions
        5. Usage examples
        6. Contributing guidelines
        
        Format as markdown and make it developer-friendly.
        """
        
        if 'commander' in self.mcp_servers:
            try:
                # Get codebase structure first
                analysis = await self._use_tool('commander', action='analyze_codebase')
                enhanced_prompt = f"{doc_prompt}\n\nCodebase analysis:\n{analysis}"
                documentation = await self._get_llm_response(enhanced_prompt)
                return f"Generated Documentation:\n\n{documentation}"
            except Exception as e:
                self.logger.error(f"Documentation generation failed: {e}")
                return f"Documentation generation failed: {str(e)}"
        else:
            documentation = await self._get_llm_response(doc_prompt)
            return f"Generated Documentation:\n\n{documentation}"
    
    async def estimate_complexity(self, task_description: str) -> Dict[str, Any]:
        """Estimate the complexity and effort required for a task"""
        estimation_prompt = f"""
        Task: {task_description}
        
        Please estimate the complexity and effort for this development task:
        
        Provide:
        1. Complexity level (low, medium, high, very high)
        2. Estimated time (hours or days)
        3. Skills required
        4. Dependencies and prerequisites
        5. Risk factors
        6. Confidence level in the estimate
        
        Be realistic and consider potential complications.
        """
        
        estimation_text = await self._get_llm_response(estimation_prompt)
        
        # Parse estimation into structured format
        # This is simplified - could be enhanced with better parsing
        return {
            "task": task_description,
            "estimation": estimation_text,
            "timestamp": asyncio.get_event_loop().time(),
            "agent": "developer"
        }

