#!/usr/bin/env python3
"""
AGIcommander Startup Script

This script orchestrates the startup of all MCP servers and core components.
Think of it as the "conductor" that brings the entire AI orchestra to life.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from core.orchestrator import AGIOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agicommander-startup")

class AGICommanderStartup:
    """
    Startup orchestrator for AGIcommander.
    
    Like a conductor preparing an orchestra, this ensures all components
    are ready and working together before the performance begins.
    """
    
    def __init__(self, config_path: str = "config/startup.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.processes = {}
        self.orchestrator = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._shutdown())
    
    async def load_config(self):
        """Load startup configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Use default configuration
                self.config = self._get_default_config()
                logger.info("Using default configuration")
                
                # Save default config for future reference
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, indent=2)
                logger.info(f"Saved default configuration to {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_default_config(self) -> Dict:
        """Get default startup configuration"""
        return {
            "agicommander": {
                "name": "AGIcommander",
                "version": "0.1.0",
                "mode": "development",  # development, production, autonomous
                "safety_mode": True,
                "log_level": "INFO"
            },
            "servers": {
                "memory": {
                    "vector_db": {
                        "enabled": True,
                        "command": "python servers/memory/vector_db/vector_db_mcp_server.py",
                        "startup_timeout": 30,
                        "health_check_interval": 60,
                        "env": {
                            "VECTOR_DB_PATH": "./memory/chromadb",
                            "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
                        }
                    },
                    "s3": {
                        "enabled": True,
                        "command": "python servers/memory/S3/s3_mcp_server.py",
                        "startup_timeout": 15,
                        "health_check_interval": 60,
                        "env": {
                            "AWS_DEFAULT_REGION": "us-east-1"
                        }
                    }
                },
                "code": {
                    "commander": {
                        "enabled": True,
                        "command": "python servers/code/commander.py",
                        "startup_timeout": 20,
                        "health_check_interval": 60
                    }
                },
                "learning": {
                    "research": {
                        "enabled": True,
                        "command": "python servers/learning/research.py", 
                        "startup_timeout": 25,
                        "health_check_interval": 60
                    }
                },
                "self_reflection": {
                    "analyzer": {
                        "enabled": False,  # Experimental feature
                        "command": "python servers/self_reflection/analyzer.py",
                        "startup_timeout": 30,
                        "health_check_interval": 120
                    }
                }
            },
            "orchestrator": {
                "startup_timeout": 60,
                "max_concurrent_tasks": 10,
                "task_timeout": 300,
                "enable_monitoring": True
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 30,
                "log_performance": True,
                "save_metrics_to_file": True,
                "metrics_file": "./logs/performance_metrics.json"
            }
        }
    
    async def check_prerequisites(self):
        """Check system prerequisites before startup"""
        logger.info("Checking system prerequisites...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check required directories
        required_dirs = [
            "servers/memory/vector_db",
            "servers/memory/S3", 
            "memory",
            "logs",
            "config"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Check environment variables
        required_env = []
        if self.config["servers"]["memory"]["s3"]["enabled"]:
            if not os.getenv("AWS_ACCESS_KEY_ID") and not os.getenv("AWS_PROFILE"):
                issues.append("AWS credentials not configured (set AWS_ACCESS_KEY_ID or AWS_PROFILE)")
        
        # Check for Google API key if using Gemini
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not set - some LLM features may not work")
        
        if issues:
            logger.error("Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise RuntimeError("Prerequisites not met")
        
        logger.info("✓ Prerequisites check passed")
    
    async def start_mcp_servers(self):
        """Start all configured MCP servers"""
        logger.info("Starting MCP servers...")
        
        for category, servers in self.config["servers"].items():
            for server_name, server_config in servers.items():
                if not server_config.get("enabled", False):
                    logger.info(f"Skipping disabled server: {category}.{server_name}")
                    continue
                
                await self._start_server(category, server_name, server_config)
        
        logger.info("✓ All MCP servers started")
    
    async def _start_server(self, category: str, name: str, config: Dict):
        """Start an individual MCP server"""
        server_id = f"{category}.{name}"
        command = config["command"]
        timeout = config.get("startup_timeout", 30)
        env = config.get("env", {})
        
        logger.info(f"Starting server: {server_id}")
        
        try:
            # Prepare environment
            server_env = os.environ.copy()
            server_env.update(env)
            
            # Start the process
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=server_env
            )
            
            # Wait for startup with timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_server_ready(process, server_id),
                    timeout=timeout
                )
                
                self.processes[server_id] = {
                    "process": process,
                    "config": config,
                    "status": "running",
                    "started_at": time.time()
                }
                
                logger.info(f"✓ Server started: {server_id}")
                
            except asyncio.TimeoutError:
                logger.error(f"✗ Server startup timeout: {server_id}")
                process.terminate()
                await process.wait()
                raise
                
        except Exception as e:
            logger.error(f"✗ Failed to start server {server_id}: {e}")
            raise
    
    async def _wait_for_server_ready(self, process: asyncio.subprocess.Process, server_id: str):
        """Wait for server to be ready (basic implementation)"""
        # For now, just wait a short time and check if process is still running
        # In a more sophisticated implementation, you'd check for specific readiness signals
        await asyncio.sleep(2)
        
        if process.returncode is not None:
            stderr = await process.stderr.read()
            raise RuntimeError(f"Server {server_id} exited during startup: {stderr.decode()}")
    
    async def start_orchestrator(self):
        """Start the main AGI orchestrator"""
        logger.info("Starting AGI Orchestrator...")
        
        try:
            self.orchestrator = AGIOrchestrator()
            await self.orchestrator.initialize()
            
            # Configure orchestrator based on startup config
            orchestrator_config = self.config.get("orchestrator", {})
            if hasattr(self.orchestrator, 'configure'):
                await self.orchestrator.configure(orchestrator_config)
            
            logger.info("✓ AGI Orchestrator started")
            
        except Exception as e:
            logger.error(f"✗ Failed to start orchestrator: {e}")
            raise
    
    async def run_health_checks(self):
        """Run health checks on all components"""
        logger.info("Running health checks...")
        
        unhealthy_servers = []
        
        for server_id, server_info in self.processes.items():
            process = server_info["process"]
            
            if process.returncode is not None:
                unhealthy_servers.append(f"{server_id} (exited with code {process.returncode})")
            else:
                logger.debug(f"✓ {server_id} is running")
        
        if unhealthy_servers:
            logger.warning(f"Unhealthy servers detected: {', '.join(unhealthy_servers)}")
            # In production, you might want to restart failed servers here
        else:
            logger.info("✓ All servers healthy")
        
        return len(unhealthy_servers) == 0
    
    async def start_monitoring(self):
        """Start system monitoring if enabled"""
        if not self.config.get("monitoring", {}).get("enabled", False):
            return
        
        logger.info("Starting monitoring...")
        
        monitoring_config = self.config["monitoring"]
        interval = monitoring_config.get("metrics_interval", 30)
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop(interval))
        
        logger.info("✓ Monitoring started")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = {
                    "timestamp": time.time(),
                    "servers": {},
                    "system": {}
                }
                
                # Server metrics
                for server_id, server_info in self.processes.items():
                    process = server_info["process"]
                    metrics["servers"][server_id] = {
                        "status": "running" if process.returncode is None else "stopped",
                        "uptime": time.time() - server_info["started_at"],
                        "pid": process.pid
                    }
                
                # System metrics (basic)
                try:
                    import psutil
                    metrics["system"] = {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/').percent
                    }
                except ImportError:
                    metrics["system"] = {"note": "psutil not available for system metrics"}
                
                # Save metrics if configured
                if self.config["monitoring"].get("save_metrics_to_file"):
                    metrics_file = Path(self.config["monitoring"]["metrics_file"])
                    metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Append to metrics file
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(metrics) + '\n')
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Starting graceful shutdown...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Shutdown orchestrator
        if self.orchestrator:
            try:
                await self.orchestrator.shutdown()
                logger.info("✓ Orchestrator shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down orchestrator: {e}")
        
        # Shutdown MCP servers
        for server_id, server_info in self.processes.items():
            try:
                process = server_info["process"]
                logger.info(f"Shutting down server: {server_id}")
                
                # Send SIGTERM first
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                    logger.info(f"✓ Server {server_id} shutdown gracefully")
                except asyncio.TimeoutError:
                    # Force kill if not responsive
                    logger.warning(f"Force killing server: {server_id}")
                    process.kill()
                    await process.wait()
                    
            except Exception as e:
                logger.error(f"Error shutting down server {server_id}: {e}")
        
        logger.info("✓ Graceful shutdown complete")
    
    async def run(self):
        """Main startup sequence"""
        try:
            logger.info("🚀 Starting AGIcommander...")
            
            # Load configuration
            await self.load_config()
            
            # Check prerequisites  
            await self.check_prerequisites()
            
            # Start MCP servers
            await self.start_mcp_servers()
            
            # Start orchestrator
            await self.start_orchestrator()
            
            # Start monitoring
            await self.start_monitoring()
            
            # Initial health check
            await self.run_health_checks()
            
            logger.info("🎯 AGIcommander startup complete!")
            logger.info("System is ready for autonomous development tasks.")
            
            # Keep running until shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self._shutdown()
            sys.exit(1)

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGIcommander Startup Script")
    parser.add_argument("--config", "-c", default="config/startup.yaml",
                       help="Path to startup configuration file")
    parser.add_argument("--mode", "-m", choices=["development", "production", "autonomous"],
                       help="Override startup mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    startup = AGICommanderStartup(config_path=args.config)
    
    # Override mode if specified
    if args.mode:
        await startup.load_config()
        startup.config["agicommander"]["mode"] = args.mode
    
    await startup.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

