#!/usr/bin/env python3
"""
test_basic.py - Basic test script for AGIcommander

Tests the core functionality of the AGIcommander system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import AGIOrchestrator


async def main():
    """Main test function"""
    print("🧪 Starting AGIcommander basic test...")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        print("🔧 Initializing orchestrator...")
        orchestrator = AGIOrchestrator("config/default.yaml")
        await orchestrator.initialize()
        
        print("✅ Initialization complete!")
        print()
        
        # Test basic task execution
        print("🎯 Testing task execution...")
        result = await orchestrator.execute_task(
            "Analyze this file and suggest one improvement"
        )
        
        print("📋 Task Result:")
        print("-" * 30)
        print(result)
        print("-" * 30)
        print()
        
        # Test system status
        print("📊 Getting system status...")
        
        # Get orchestrator status
        print("Orchestrator status:")
        print(f"  - Running: {orchestrator.is_running}")
        print(f"  - LLM adapters: {len(orchestrator.llm_adapters)}")
        print(f"  - MCP servers: {len(orchestrator.mcp_servers)}")
        print(f"  - Agents: {len(orchestrator.agents)}")
        print()
        
        # Get MCP server status
        if orchestrator.mcp_servers:
            print("MCP Server status:")
            for name, server in orchestrator.mcp_servers.items():
                status = server.get_status()
                print(f"  - {name}: {'🟢' if status['running'] else '🔴'} "
                      f"(requests: {status['request_count']}, "
                      f"errors: {status['error_count']})")
        
        # Get agent performance
        if orchestrator.agents:
            print("\nAgent performance:")
            for name, agent in orchestrator.agents.items():
                perf = agent.get_performance_summary()
                print(f"  - {name}: {perf['total_tasks']} tasks, "
                      f"{perf['success_rate']:.1%} success rate")
        
        print()
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Always try to shutdown gracefully
        try:
            print("\n🛑 Shutting down...")
            await orchestrator.shutdown()
            print("✅ Shutdown complete!")
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")
    
    return True


if __name__ == "__main__":
    # Check environment setup
    print("🔍 Checking environment...")
    
    # Check for required environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)
    else:
        print("✅ GOOGLE_API_KEY found")
    
    # Check for config file
    config_file = Path("config/default.yaml")
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Please ensure the config file exists in the correct location")
        sys.exit(1)
    else:
        print("✅ Configuration file found")
    
    # Check for required directories
    required_dirs = ["memory/logs", "memory/vector", "memory/relational"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Required directories created")
    
    print()
    
    # Run the test
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 All tests passed! AGIcommander is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed! Check the error messages above.")
        sys.exit(1)

