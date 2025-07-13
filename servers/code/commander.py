#!/usr/bin/env python3
"""
servers/code/commander.py - Commander functionality as MCP server

Converts the original commander.py into an MCP server that can be used
by AGIcommander agents for code modification tasks.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent

# Import original commander classes (modified for async)
from ...original.commander_classes import (
    FileProcessor, 
    CommanderInstructions, 
    GeminiProcessor, 
    ResponseParser
)


class CommanderMCPServer:
    """MCP Server wrapper for Commander functionality"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.server = Server("commander")
        self.api_key = config.get('api_key') or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not found in config or environment")
        
        # Initialize components
        self.gemini_processor = GeminiProcessor(self.api_key)
        self.response_parser = ResponseParser()
        
        # Register MCP tools
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register MCP tools for code modification"""
        
        @self.server.tool()
        async def modify_files(
            instructions: str,
            extensions: str = "py",
            recursive: bool = False,
            directory: str = ".",
            auto_confirm: bool = True
        ) -> str:
            """
            Modify files based on AI instructions (original Commander functionality)
            
            Args:
                instructions: What modifications to make
                extensions: File extensions to process (comma-separated)
                recursive: Process subdirectories
                directory: Base directory to search
                auto_confirm: Skip confirmation prompts
            """
            try:
                result = await self._execute_commander_logic(
                    instructions, extensions, recursive, directory, auto_confirm
                )
                return f"✅ Modified {result['files_modified']} files successfully"
                
            except Exception as e:
                return f"❌ Error modifying files: {str(e)}"
        
        @self.server.tool()
        async def analyze_codebase(
            directory: str = ".",
            extensions: str = "py",
            recursive: bool = True
        ) -> str:
            """
            Analyze codebase structure and provide insights
            
            Args:
                directory: Directory to analyze
                extensions: File extensions to include
                recursive: Include subdirectories
            """
            try:
                analysis = await self._analyze_codebase(directory, extensions, recursive)
                return analysis
                
            except Exception as e:
                return f"❌ Error analyzing codebase: {str(e)}"
        
        @self.server.tool()
        async def suggest_improvements(
            directory: str = ".",
            focus_areas: str = "all"
        ) -> str:
            """
            Suggest code improvements based on analysis
            
            Args:
                directory: Directory to analyze
                focus_areas: Areas to focus on (security, performance, style, all)
            """
            try:
                suggestions = await self._suggest_improvements(directory, focus_areas)
                return suggestions
                
            except Exception as e:
                return f"❌ Error generating suggestions: {str(e)}"
        
        @self.server.tool()
        async def create_files(
            file_specs: str,
            base_directory: str = "."
        ) -> str:
            """
            Create new files based on specifications
            
            Args:
                file_specs: JSON string with file specifications
                base_directory: Base directory for new files
            """
            try:
                result = await self._create_files(file_specs, base_directory)
                return f"✅ Created {result['files_created']} files"
                
            except Exception as e:
                return f"❌ Error creating files: {str(e)}"
    
    def _register_resources(self):
        """Register MCP resources for file access"""
        
        @self.server.resource("file://{path}")
        async def read_file(path: str) -> str:
            """Read file contents"""
            try:
                file_path = Path(path)
                if file_path.exists() and file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"File not found: {path}"
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        @self.server.resource("codebase://structure")
        async def get_codebase_structure() -> str:
            """Get current codebase structure"""
            try:
                return await self._get_directory_structure()
            except Exception as e:
                return f"Error getting structure: {str(e)}"
    
    async def _execute_commander_logic(
        self, 
        instructions: str, 
        extensions: str, 
        recursive: bool, 
        directory: str, 
        auto_confirm: bool
    ) -> Dict:
        """Execute the core commander logic asynchronously"""
        
        # Parse extensions
        ext_list = [ext.strip().lstrip('.') for ext in extensions.split(',')]
        ext_list = [ext for ext in ext_list if ext]
        
        # Find files
        file_processor = FileProcessor(recursive, ext_list)
        found_files = file_processor.find_files(directory)
        
        if not found_files:
            raise Exception(f"No files found with extensions: {extensions}")
        
        # Read file contents
        files_data = {}
        for file_path in found_files:
            content = file_processor.read_file_content(file_path)
            if content:
                language = file_processor.get_language_for_extension(file_path)
                files_data[file_path] = (content, language)
        
        if not files_data:
            raise Exception("No files could be read")
        
        # Process with Gemini
        response = await asyncio.to_thread(
            self.gemini_processor.process_files, 
            instructions, 
            files_data
        )
        
        if not response:
            raise Exception("No response from Gemini")
        
        # Parse and apply changes
        modified_files = self.response_parser.parse_response(response)
        
        if not modified_files:
            return {"files_modified": 0, "message": "No files needed modification"}
        
        # Write modified files (if auto_confirm is True)
        if auto_confirm:
            self.response_parser.write_modified_files(modified_files)
            return {
                "files_modified": len(modified_files),
                "files": list(modified_files.keys()),
                "message": "Files modified successfully"
            }
        else:
            return {
                "files_modified": 0,
                "pending_files": list(modified_files.keys()),
                "message": "Files ready for modification (pending confirmation)"
            }
    
    async def _analyze_codebase(self, directory: str, extensions: str, recursive: bool) -> str:
        """Analyze codebase and return insights"""
        
        # Use file processor to gather files
        ext_list = [ext.strip().lstrip('.') for ext in extensions.split(',')]
        file_processor = FileProcessor(recursive, ext_list)
        found_files = file_processor.find_files(directory)
        
        # Gather file statistics
        total_files = len(found_files)
        total_lines = 0
        file_sizes = []
        
        for file_path in found_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    file_sizes.append(lines)
            except:
                pass
        
        # Generate analysis using AI
        analysis_prompt = f"""
        Analyze this codebase structure:
        - Total files: {total_files}
        - Total lines: {total_lines}
        - Average file size: {total_lines // total_files if total_files > 0 else 0} lines
        - File extensions: {extensions}
        - Files: {found_files[:10]}{'...' if len(found_files) > 10 else ''}
        
        Provide insights about code organization, potential issues, and recommendations.
        """
        
        # Use Gemini for analysis
        response = await asyncio.to_thread(
            self.gemini_processor.process_files,
            analysis_prompt,
            {}  # No files to modify, just analysis
        )
        
        return response or "Unable to analyze codebase"
    
    async def _suggest_improvements(self, directory: str, focus_areas: str) -> str:
        """Generate improvement suggestions"""
        
        improvement_prompt = f"""
        Suggest improvements for the codebase in {directory}.
        Focus areas: {focus_areas}
        
        Provide specific, actionable suggestions for:
        - Code quality improvements
        - Performance optimizations
        - Security enhancements
        - Architecture improvements
        - Testing strategies
        """
        
        # Use Gemini for suggestions
        response = await asyncio.to_thread(
            self.gemini_processor.process_files,
            improvement_prompt,
            {}
        )
        
        return response or "Unable to generate suggestions"
    
    async def _create_files(self, file_specs: str, base_directory: str) -> Dict:
        """Create new files based on specifications"""
        
        creation_prompt = f"""
        Create files based on these specifications:
        {file_specs}
        
        Base directory: {base_directory}
        
        Generate complete, functional file contents for each specified file.
        """
        
        # Use Gemini to generate file contents
        response = await asyncio.to_thread(
            self.gemini_processor.process_files,
            creation_prompt,
            {}
        )
        
        if response:
            # Parse generated files and create them
            modified_files = self.response_parser.parse_response(response)
            
            # Create files in base directory
            for filename, content in modified_files.items():
                full_path = Path(base_directory) / filename
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return {
                "files_created": len(modified_files),
                "files": list(modified_files.keys())
            }
        
        return {"files_created": 0, "files": []}
    
    async def _get_directory_structure(self, max_depth: int = 3) -> str:
        """Get directory structure as a tree"""
        
        def build_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
            if current_depth >= max_depth:
                return ""
            
            items = []
            try:
                for item in sorted(directory.iterdir()):
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        items.append(f"{prefix}📁 {item.name}/")
                        if current_depth < max_depth - 1:
                            subtree = build_tree(item, prefix + "  ", max_depth, current_depth + 1)
                            if subtree:
                                items.append(subtree)
                    else:
                        items.append(f"{prefix}📄 {item.name}")
            except PermissionError:
                items.append(f"{prefix}❌ Permission denied")
            
            return "\n".join(items)
        
        return build_tree(Path("."), max_depth=max_depth)
    
    async def start(self):
        """Start the MCP server"""
        await self.server.start()
    
    async def stop(self):
        """Stop the MCP server"""
        await self.server.stop()


# Server factory function for dynamic loading
def create_server(config: Dict) -> CommanderMCPServer:
    """Factory function to create CommanderMCPServer instance"""
    return CommanderMCPServer(config)


# For testing
if __name__ == "__main__":
    async def main():
        config = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "name": "commander",
            "type": "code"
        }
        
        server = CommanderMCPServer(config)
        await server.start()
        
        # Test the modify_files tool
        result = await server.server.tools["modify_files"](
            instructions="Add docstrings to all functions",
            extensions="py",
            recursive=False
        )
        print(f"Result: {result}")
        
        await server.stop()
    
    asyncio.run(main())
