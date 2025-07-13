#!/usr/bin/env python3
"""
servers/self_reflection/metrics.py - Performance metrics MCP server

Provides performance tracking and metrics collection for AGIcommander
to monitor its own performance and identify optimization opportunities.
"""

import asyncio
import json
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from mcp.server import Server
from mcp.types import Tool, TextContent

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from servers.base_server import BaseMCPServer


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: str
    metric_type: str
    value: float
    unit: str
    context: Dict[str, Any]


@dataclass
class TaskMetric:
    """Task execution metric"""
    task_id: str
    task_type: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    error_message: Optional[str]
    resource_usage: Dict[str, float]


class MetricsMCPServer(BaseMCPServer):
    """MCP Server for performance metrics and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server = Server("metrics")
        
        # Configuration
        self.collection_interval = config.get('collection_interval', 60)  # seconds
        self.retention_days = config.get('retention_days', 30)
        self.enable_system_metrics = config.get('enable_system_metrics', True)
        
        # Metrics storage
        self.performance_metrics: List[PerformanceMetric] = []
        self.task_metrics: List[TaskMetric] = []
        self.system_metrics: List[Dict[str, Any]] = []
        
        # Metric collection state
        self.collection_active = False
        self.collection_task = None
        
        # Register MCP tools
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register MCP tools for metrics collection and analysis"""
        
        @self.server.tool()
        async def collect_performance_metrics() -> str:
            """Collect current performance metrics"""
            try:
                metrics = await self._collect_current_metrics()
                return self._format_metrics_summary(metrics)
            except Exception as e:
                return f"Metrics collection failed: {str(e)}"
        
        @self.server.tool()
        async def analyze_performance_trends() -> str:
            """Analyze performance trends over time"""
            try:
                analysis = await self._analyze_trends()
                return self._format_trend_analysis(analysis)
            except Exception as e:
                return f"Trend analysis failed: {str(e)}"
        
        @self.server.tool()
        async def get_system_health() -> str:
            """Get current system health status"""
            try:
                health = await self._assess_system_health()
                return self._format_health_report(health)
            except Exception as e:
                return f"Health assessment failed: {str(e)}"
        
        @self.server.tool()
        async def track_task_performance(task_data: str) -> str:
            """Track performance of a specific task"""
            try:
                task_info = json.loads(task_data)
                await self._record_task_metric(task_info)
                return "Task performance recorded"
            except Exception as e:
                return f"Task tracking failed: {str(e)}"
        
        @self.server.tool()
        async def generate_performance_report() -> str:
            """Generate comprehensive performance report"""
            try:
                report = await self._generate_comprehensive_report()
                return report
            except Exception as e:
                return f"Report generation failed: {str(e)}"
        
        @self.server.tool()
        async def identify_bottlenecks() -> str:
            """Identify performance bottlenecks"""
            try:
                bottlenecks = await self._identify_performance_bottlenecks()
                return self._format_bottleneck_analysis(bottlenecks)
            except Exception as e:
                return f"Bottleneck analysis failed: {str(e)}"
    
    def _register_resources(self):
        """Register MCP resources for metrics data"""
        
        @self.server.resource("metrics://performance")
        async def get_performance_data() -> str:
            """Get performance metrics data"""
            return json.dumps([asdict(m) for m in self.performance_metrics[-100:]], indent=2)
        
        @self.server.resource("metrics://tasks")
        async def get_task_data() -> str:
            """Get task metrics data"""
            return json.dumps([asdict(m) for m in self.task_metrics[-50:]], indent=2)
        
        @self.server.resource("metrics://system")
        async def get_system_data() -> str:
            """Get system metrics data"""
            return json.dumps(self.system_metrics[-50:], indent=2)
    
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "system": await self._collect_system_metrics(),
            "application": await self._collect_application_metrics(),
            "ai_performance": await self._collect_ai_metrics()
        }
        
        # Store metrics
        for metric_type, data in metrics.items():
            if metric_type != "timestamp" and isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        metric = PerformanceMetric(
                            timestamp=timestamp,
                            metric_type=f"{metric_type}.{key}",
                            value=float(value),
                            unit=self._get_metric_unit(key),
                            context={"category": metric_type}
                        )
                        self.performance_metrics.append(metric)
        
        return metrics
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        if not self.enable_system_metrics:
            return {}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        
        # Calculate recent task success rate
        recent_tasks = [t for t in self.task_metrics 
                       if datetime.fromisoformat(t.start_time) > datetime.now() - timedelta(hours=1)]
        
        success_rate = (sum(1 for t in recent_tasks if t.success) / len(recent_tasks)) if recent_tasks else 0
        
        # Calculate average response time
        avg_duration = sum(t.duration for t in recent_tasks) / len(recent_tasks) if recent_tasks else 0
        
        return {
            "active_servers": self.request_count,  # From base class
            "total_requests": len(self.task_metrics),
            "success_rate_1h": success_rate,
            "avg_response_time": avg_duration,
            "error_rate": self.error_count / max(self.request_count, 1),
            "uptime_seconds": time.time() - (self.start_time.timestamp() if self.start_time else time.time())
        }
    
    async def _collect_ai_metrics(self) -> Dict[str, Any]:
        """Collect AI-specific performance metrics"""
        
        # Analyze recent AI tasks
        ai_tasks = [t for t in self.task_metrics 
                   if t.task_type in ['code_analysis', 'code_modification', 'ai_completion']]
        
        if not ai_tasks:
            return {}
        
        # Group by task type
        task_performance = {}
        for task_type in ['code_analysis', 'code_modification', 'ai_completion']:
            type_tasks = [t for t in ai_tasks if t.task_type == task_type]
            if type_tasks:
                task_performance[f"{task_type}_success_rate"] = sum(1 for t in type_tasks if t.success) / len(type_tasks)
                task_performance[f"{task_type}_avg_duration"] = sum(t.duration for t in type_tasks) / len(type_tasks)
        
        return task_performance
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get the unit for a metric"""
        unit_map = {
            "cpu_usage_percent": "%",
            "memory_usage_percent": "%", 
            "memory_available_gb": "GB",
            "disk_usage_percent": "%",
            "disk_free_gb": "GB",
            "success_rate": "%",
            "avg_response_time": "seconds",
            "uptime_seconds": "seconds",
            "error_rate": "%"
        }
        return unit_map.get(metric_name, "count")
    
    async def _record_task_metric(self, task_info: Dict[str, Any]):
        """Record metrics for a task execution"""
        
        task_metric = TaskMetric(
            task_id=task_info.get('task_id', f"task_{len(self.task_metrics)}"),
            task_type=task_info.get('task_type', 'unknown'),
            start_time=task_info.get('start_time', datetime.now().isoformat()),
            end_time=task_info.get('end_time', datetime.now().isoformat()),
            duration=task_info.get('duration', 0.0),
            success=task_info.get('success', True),
            error_message=task_info.get('error_message'),
            resource_usage=task_info.get('resource_usage', {})
        )
        
        self.task_metrics.append(task_metric)
        
        # Persist metric
        await self._persist_task_metric(task_metric)
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        # Group metrics by type and time period
        trend_analysis = {}
        
        # Analyze last 24 hours in hourly buckets
        now = datetime.now()
        for hours_back in range(24):
            bucket_start = now - timedelta(hours=hours_back+1)
            bucket_end = now - timedelta(hours=hours_back)
            
            bucket_metrics = [
                m for m in self.performance_metrics
                if bucket_start <= datetime.fromisoformat(m.timestamp) < bucket_end
            ]
            
            if bucket_metrics:
                bucket_key = f"{hours_back}h_ago"
                trend_analysis[bucket_key] = {}
                
                # Group by metric type
                by_type = {}
                for metric in bucket_metrics:
                    if metric.metric_type not in by_type:
                        by_type[metric.metric_type] = []
                    by_type[metric.metric_type].append(metric.value)
                
                # Calculate averages
                for metric_type, values in by_type.items():
                    trend_analysis[bucket_key][metric_type] = sum(values) / len(values)
        
        return trend_analysis
    
    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        
        current_metrics = await self._collect_current_metrics()
        
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "alerts": [],
            "recommendations": []
        }
        
        # CPU health check
        cpu_usage = current_metrics.get("system", {}).get("cpu_usage_percent", 0)
        if cpu_usage > 80:
            health_status["checks"]["cpu"] = "warning"
            health_status["alerts"].append(f"High CPU usage: {cpu_usage:.1f}%")
        else:
            health_status["checks"]["cpu"] = "healthy"
        
        # Memory health check
        memory_usage = current_metrics.get("system", {}).get("memory_usage_percent", 0)
        if memory_usage > 85:
            health_status["checks"]["memory"] = "critical"
            health_status["alerts"].append(f"High memory usage: {memory_usage:.1f}%")
        elif memory_usage > 70:
            health_status["checks"]["memory"] = "warning"
            health_status["alerts"].append(f"Elevated memory usage: {memory_usage:.1f}%")
        else:
            health_status["checks"]["memory"] = "healthy"
        
        # Application health check
        success_rate = current_metrics.get("application", {}).get("success_rate_1h", 1.0)
        if success_rate < 0.8:
            health_status["checks"]["application"] = "warning"
            health_status["alerts"].append(f"Low success rate: {success_rate:.1%}")
        else:
            health_status["checks"]["application"] = "healthy"
        
        # Determine overall status
        if any(status == "critical" for status in health_status["checks"].values()):
            health_status["overall_status"] = "critical"
        elif any(status == "warning" for status in health_status["checks"].values()):
            health_status["overall_status"] = "warning"
        
        # Add recommendations
        if health_status["overall_status"] != "healthy":
            health_status["recommendations"] = [
                "Monitor resource usage closely",
                "Consider scaling if issues persist",
                "Review recent changes for performance impact"
            ]
        
        return health_status
    
    async def _identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # Analyze slow tasks
        slow_tasks = [t for t in self.task_metrics if t.duration > 10.0]  # Tasks > 10 seconds
        if slow_tasks:
            avg_slow_duration = sum(t.duration for t in slow_tasks) / len(slow_tasks)
            bottlenecks.append({
                "type": "slow_tasks",
                "severity": "medium",
                "description": f"{len(slow_tasks)} tasks took longer than 10 seconds",
                "impact": f"Average duration: {avg_slow_duration:.1f}s",
                "recommendation": "Optimize slow task processing or add timeout handling"
            })
        
        # Analyze high failure rate tasks
        failed_tasks = [t for t in self.task_metrics if not t.success]
        if len(failed_tasks) / max(len(self.task_metrics), 1) > 0.1:  # >10% failure rate
            bottlenecks.append({
                "type": "high_failure_rate",
                "severity": "high",
                "description": f"{len(failed_tasks)} out of {len(self.task_metrics)} tasks failed",
                "impact": f"Failure rate: {len(failed_tasks) / len(self.task_metrics):.1%}",
                "recommendation": "Investigate common failure patterns and add error handling"
            })
        
        # Analyze system resource bottlenecks
        recent_metrics = [m for m in self.performance_metrics 
                         if datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(hours=1)]
        
        cpu_metrics = [m for m in recent_metrics if m.metric_type == "system.cpu_usage_percent"]
        if cpu_metrics and sum(m.value for m in cpu_metrics) / len(cpu_metrics) > 70:
            bottlenecks.append({
                "type": "cpu_bottleneck",
                "severity": "medium",
                "description": "Consistently high CPU usage",
                "impact": f"Average CPU: {sum(m.value for m in cpu_metrics) / len(cpu_metrics):.1f}%",
                "recommendation": "Consider optimizing CPU-intensive operations or scaling"
            })
        
        return bottlenecks
    
    async def _generate_comprehensive_report(self) -> str:
        """Generate a comprehensive performance report"""
        
        current_metrics = await self._collect_current_metrics()
        trends = await self._analyze_trends()
        health = await self._assess_system_health()
        bottlenecks = await self._identify_performance_bottlenecks()
        
        report = f"""
# AGIcommander Performance Report
**Generated:** {datetime.now().isoformat()}

## Executive Summary
- **Overall Health:** {health['overall_status'].upper()}
- **Total Tasks Processed:** {len(self.task_metrics)}
- **Current Success Rate:** {current_metrics.get('application', {}).get('success_rate_1h', 0):.1%}
- **Average Response Time:** {current_metrics.get('application', {}).get('avg_response_time', 0):.2f}s

## Current System Metrics
### Resource Usage
- **CPU:** {current_metrics.get('system', {}).get('cpu_usage_percent', 0):.1f}%
- **Memory:** {current_metrics.get('system', {}).get('memory_usage_percent', 0):.1f}%
- **Disk:** {current_metrics.get('system', {}).get('disk_usage_percent', 0):.1f}%

### Application Performance
- **Uptime:** {current_metrics.get('application', {}).get('uptime_seconds', 0) / 3600:.1f} hours
- **Error Rate:** {current_metrics.get('application', {}).get('error_rate', 0):.1%}
- **Total Requests:** {current_metrics.get('application', {}).get('total_requests', 0)}

## Health Alerts
{chr(10).join(f"- {alert}" for alert in health.get('alerts', ['No alerts']))}

## Performance Bottlenecks
{chr(10).join(f"- **{b['type']}** ({b['severity']}): {b['description']}" for b in bottlenecks) if bottlenecks else "No significant bottlenecks detected"}

## Recommendations
{chr(10).join(f"- {rec}" for rec in health.get('recommendations', ['System performing well']))}

## AI Performance Breakdown
"""
        
        ai_metrics = current_metrics.get('ai_performance', {})
        for metric_name, value in ai_metrics.items():
            if 'success_rate' in metric_name:
                report += f"- **{metric_name.replace('_', ' ').title()}:** {value:.1%}\n"
            elif 'duration' in metric_name:
                report += f"- **{metric_name.replace('_', ' ').title()}:** {value:.2f}s\n"
        
        return report.strip()
    
    def _format_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a summary"""
        
        summary = f"""
# Current Performance Metrics
**Timestamp:** {metrics['timestamp']}

## System Resources
"""
        
        system_metrics = metrics.get('system', {})
        for key, value in system_metrics.items():
            unit = self._get_metric_unit(key)
            summary += f"- **{key.replace('_', ' ').title()}:** {value:.1f}{unit}\n"
        
        summary += "\n## Application Performance\n"
        app_metrics = metrics.get('application', {})
        for key, value in app_metrics.items():
            unit = self._get_metric_unit(key)
            if 'rate' in key:
                summary += f"- **{key.replace('_', ' ').title()}:** {value:.1%}\n"
            else:
                summary += f"- **{key.replace('_', ' ').title()}:** {value:.2f}{unit}\n"
        
        return summary.strip()
    
    def _format_trend_analysis(self, trends: Dict[str, Any]) -> str:
        """Format trend analysis"""
        
        analysis = "# Performance Trend Analysis\n\n"
        
        # Show trends for key metrics
        key_metrics = ['system.cpu_usage_percent', 'application.success_rate_1h', 'application.avg_response_time']
        
        for metric in key_metrics:
            analysis += f"## {metric.replace('_', ' ').replace('.', ' - ').title()}\n"
            
            metric_trends = []
            for time_key in sorted(trends.keys(), key=lambda x: int(x.replace('h_ago', ''))):
                value = trends[time_key].get(metric)
                if value is not None:
                    metric_trends.append(f"- {time_key}: {value:.2f}")
            
            if metric_trends:
                analysis += "\n".join(metric_trends) + "\n\n"
            else:
                analysis += "No data available\n\n"
        
        return analysis.strip()
    
    def _format_health_report(self, health: Dict[str, Any]) -> str:
        """Format health report"""
        
        status_icons = {
            "healthy": "✅",
            "warning": "⚠️", 
            "critical": "🚨"
        }
        
        report = f"""
# System Health Report
**Overall Status:** {status_icons.get(health['overall_status'], '❓')} {health['overall_status'].upper()}

## Health Checks
"""
        
        for check, status in health['checks'].items():
            icon = status_icons.get(status, '❓')
            report += f"- **{check.title()}:** {icon} {status}\n"
        
        if health['alerts']:
            report += "\n## Active Alerts\n"
            for alert in health['alerts']:
                report += f"- 🚨 {alert}\n"
        
        if health['recommendations']:
            report += "\n## Recommendations\n"
            for rec in health['recommendations']:
                report += f"- 💡 {rec}\n"
        
        return report.strip()
    
    def _format_bottleneck_analysis(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """Format bottleneck analysis"""
        
        if not bottlenecks:
            return "# Performance Analysis\n\n✅ No significant performance bottlenecks detected."
        
        analysis = "# Performance Bottleneck Analysis\n\n"
        
        severity_icons = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🔴",
            "critical": "🚨"
        }
        
        for bottleneck in bottlenecks:
            icon = severity_icons.get(bottleneck['severity'], '❓')
            analysis += f"""
## {icon} {bottleneck['type'].replace('_', ' ').title()}
**Severity:** {bottleneck['severity']}
**Description:** {bottleneck['description']}
**Impact:** {bottleneck['impact']}
**Recommendation:** {bottleneck['recommendation']}

---
            """.strip()
        
        return analysis
    
    async def _persist_task_metric(self, metric: TaskMetric):
        """Persist task metric for historical analysis"""
        
        try:
            metrics_file = Path("memory/logs/task_metrics.jsonl")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, "a") as f:
                f.write(json.dumps(asdict(metric)) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to persist task metric: {e}")
    
    async def start_metrics_collection(self):
        """Start automatic metrics collection"""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._metrics_collection_loop())
        self.logger.info("Started metrics collection")
    
    async def stop_metrics_collection(self):
        """Stop automatic metrics collection"""
        self.collection_active = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped metrics collection")
    
    async def _metrics_collection_loop(self):
        """Main metrics collection loop"""
        while self.collection_active:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def start(self):
        """Start the metrics MCP server"""
        self._log_start()
        await self.server.start()
        await self.start_metrics_collection()
    
    async def stop(self):
        """Stop the metrics MCP server"""
        await self.stop_metrics_collection()
        await self.server.stop()
        self._log_stop()
    
    async def _execute_action(self, action: str, **kwargs) -> str:
        """Execute metrics actions"""
        if action == "collect_metrics":
            metrics = await self._collect_current_metrics()
            return self._format_metrics_summary(metrics)
        elif action == "health_check":
            health = await self._assess_system_health()
            return self._format_health_report(health)
        elif action == "performance_report":
            return await self._generate_comprehensive_report()
        else:
            return f"Unknown metrics action: {action}"


# Factory function for dynamic loading
def create_server(config: Dict[str, Any]) -> MetricsMCPServer:
    """Factory function to create MetricsMCPServer instance"""
    return MetricsMCPServer(config)


# For testing
if __name__ == "__main__":
    async def main():
        config = {
            "name": "metrics",
            "type": "self_reflection/metrics",
            "collection_interval": 30,
            "enable_system_metrics": True
        }
        
        server = MetricsMCPServer(config)
        await server.start()
        
        # Test metrics collection
        result = await server._execute_action("collect_metrics")
        print("Metrics Collection:")
        print(result)
        
        await server.stop()
    
    asyncio.run(main())

