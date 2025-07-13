#!/usr/bin/env python3
"""
servers/self_reflection/evolution.py - Capability evolution tracking MCP server

Tracks the evolution of AGIcommander's capabilities over time, monitoring
learning progress, skill acquisition, and autonomous development.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from mcp.server import Server
from mcp.types import Tool, TextContent

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from servers.base_server import BaseMCPServer


@dataclass
class CapabilitySnapshot:
    """Snapshot of system capabilities at a point in time"""
    timestamp: str
    version: str
    capabilities: Dict[str, Any]
    performance_metrics: Dict[str, float]
    knowledge_areas: List[str]
    skill_levels: Dict[str, float]
    limitations: List[str]
    checksum: str


@dataclass
class LearningEvent:
    """Record of a learning event or capability change"""
    timestamp: str
    event_type: str  # 'skill_acquired', 'capability_enhanced', 'knowledge_gained'
    description: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    impact_score: float
    confidence: float


@dataclass
class EvolutionMilestone:
    """Significant milestone in AGI development"""
    timestamp: str
    milestone_type: str
    title: str
    description: str
    significance: str  # 'minor', 'major', 'breakthrough'
    evidence: List[str]
    metrics_improvement: Dict[str, float]


class EvolutionMCPServer(BaseMCPServer):
    """MCP Server for tracking capability evolution and learning progress"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server = Server("evolution")
        
        # Configuration
        self.snapshot_interval = config.get('snapshot_interval', 86400)  # 24 hours
        self.milestone_threshold = config.get('milestone_threshold', 0.15)  # 15% improvement
        self.track_code_changes = config.get('track_code_changes', True)
        
        # Evolution tracking data
        self.capability_snapshots: List[CapabilitySnapshot] = []
        self.learning_events: List[LearningEvent] = []
        self.evolution_milestones: List[EvolutionMilestone] = []
        
        # Current state tracking
        self.current_capabilities: Dict[str, Any] = {}
        self.skill_progression: Dict[str, List[float]] = {}
        self.knowledge_graph: Dict[str, Set[str]] = {}
        
        # Register MCP tools
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register MCP tools for evolution tracking"""
        
        @self.server.tool()
        async def track_capability_evolution() -> str:
            """Track the evolution of system capabilities"""
            try:
                evolution_data = await self._analyze_capability_evolution()
                return self._format_evolution_analysis(evolution_data)
            except Exception as e:
                return f"Evolution tracking failed: {str(e)}"
        
        @self.server.tool()
        async def record_learning_event(event_data: str) -> str:
            """Record a significant learning or capability change event"""
            try:
                event_info = json.loads(event_data)
                await self._record_learning_event(event_info)
                return "Learning event recorded successfully"
            except Exception as e:
                return f"Learning event recording failed: {str(e)}"
        
        @self.server.tool()
        async def create_capability_snapshot() -> str:
            """Create a snapshot of current system capabilities"""
            try:
                snapshot = await self._create_capability_snapshot()
                return self._format_capability_snapshot(snapshot)
            except Exception as e:
                return f"Snapshot creation failed: {str(e)}"
        
        @self.server.tool()
        async def analyze_learning_progress() -> str:
            """Analyze learning progress and skill development over time"""
            try:
                progress = await self._analyze_learning_progress()
                return self._format_learning_progress(progress)
            except Exception as e:
                return f"Learning progress analysis failed: {str(e)}"
        
        @self.server.tool()
        async def detect_evolution_milestones() -> str:
            """Detect and record significant evolution milestones"""
            try:
                milestones = await self._detect_milestones()
                return self._format_milestones(milestones)
            except Exception as e:
                return f"Milestone detection failed: {str(e)}"
        
        @self.server.tool()
        async def predict_capability_growth() -> str:
            """Predict future capability development based on current trends"""
            try:
                predictions = await self._predict_growth_trajectory()
                return self._format_growth_predictions(predictions)
            except Exception as e:
                return f"Growth prediction failed: {str(e)}"
        
        @self.server.tool()
        async def assess_agi_progress() -> str:
            """Assess progress toward AGI capabilities"""
            try:
                assessment = await self._assess_agi_progress()
                return self._format_agi_assessment(assessment)
            except Exception as e:
                return f"AGI assessment failed: {str(e)}"
    
    def _register_resources(self):
        """Register MCP resources for evolution data"""
        
        @self.server.resource("evolution://snapshots")
        async def get_capability_snapshots() -> str:
            """Get capability snapshots history"""
            return json.dumps([asdict(s) for s in self.capability_snapshots[-10:]], indent=2)
        
        @self.server.resource("evolution://events")
        async def get_learning_events() -> str:
            """Get learning events history"""
            return json.dumps([asdict(e) for e in self.learning_events[-20:]], indent=2)
        
        @self.server.resource("evolution://milestones")
        async def get_evolution_milestones() -> str:
            """Get evolution milestones"""
            return json.dumps([asdict(m) for m in self.evolution_milestones], indent=2)
    
    async def _create_capability_snapshot(self) -> CapabilitySnapshot:
        """Create a comprehensive snapshot of current capabilities"""
        
        timestamp = datetime.now().isoformat()
        
        # Assess current capabilities
        capabilities = await self._assess_current_capabilities()
        performance_metrics = await self._collect_performance_metrics()
        knowledge_areas = await self._catalog_knowledge_areas()
        skill_levels = await self._assess_skill_levels()
        limitations = await self._identify_current_limitations()
        
        # Create checksum for integrity
        snapshot_data = {
            "capabilities": capabilities,
            "performance": performance_metrics,
            "knowledge": knowledge_areas,
            "skills": skill_levels
        }
        checksum = hashlib.sha256(json.dumps(snapshot_data, sort_keys=True).encode()).hexdigest()[:16]
        
        snapshot = CapabilitySnapshot(
            timestamp=timestamp,
            version=f"v{len(self.capability_snapshots) + 1}",
            capabilities=capabilities,
            performance_metrics=performance_metrics,
            knowledge_areas=knowledge_areas,
            skill_levels=skill_levels,
            limitations=limitations,
            checksum=checksum
        )
        
        self.capability_snapshots.append(snapshot)
        await self._persist_snapshot(snapshot)
        
        return snapshot
    
    async def _assess_current_capabilities(self) -> Dict[str, Any]:
        """Assess current system capabilities"""
        
        capabilities = {
            "core_functions": {
                "code_analysis": await self._assess_capability_level("code_analysis"),
                "code_modification": await self._assess_capability_level("code_modification"),
                "problem_solving": await self._assess_capability_level("problem_solving"),
                "learning": await self._assess_capability_level("learning"),
                "self_reflection": await self._assess_capability_level("self_reflection")
            },
            "technical_skills": {
                "python_proficiency": await self._assess_skill("python"),
                "javascript_proficiency": await self._assess_skill("javascript"),
                "architecture_design": await self._assess_skill("architecture"),
                "debugging": await self._assess_skill("debugging"),
                "testing": await self._assess_skill("testing")
            },
            "cognitive_abilities": {
                "pattern_recognition": await self._assess_cognitive_ability("pattern_recognition"),
                "abstract_reasoning": await self._assess_cognitive_ability("abstract_reasoning"),
                "creative_problem_solving": await self._assess_cognitive_ability("creativity"),
                "knowledge_synthesis": await self._assess_cognitive_ability("synthesis"),
                "meta_learning": await self._assess_cognitive_ability("meta_learning")
            },
            "integration_capabilities": {
                "multi_tool_coordination": await self._assess_integration("multi_tool"),
                "context_preservation": await self._assess_integration("context"),
                "error_recovery": await self._assess_integration("error_recovery"),
                "adaptive_behavior": await self._assess_integration("adaptation")
            }
        }
        
        return capabilities
    
    async def _assess_capability_level(self, capability: str) -> float:
        """Assess the level of a specific capability (0.0 to 1.0)"""
        capability_scores = {
            "code_analysis": 0.75,
            "code_modification": 0.68,
            "problem_solving": 0.72,
            "learning": 0.45,
            "self_reflection": 0.58
        }
        return capability_scores.get(capability, 0.5)
    
    async def _assess_skill(self, skill: str) -> float:
        """Assess proficiency level in a specific skill"""
        skill_scores = {
            "python": 0.82,
            "javascript": 0.65,
            "architecture": 0.55,
            "debugging": 0.70,
            "testing": 0.48
        }
        return skill_scores.get(skill, 0.5)
    
    async def _assess_cognitive_ability(self, ability: str) -> float:
        """Assess cognitive ability level"""
        cognitive_scores = {
            "pattern_recognition": 0.78,
            "abstract_reasoning": 0.62,
            "creativity": 0.55,
            "synthesis": 0.68,
            "meta_learning": 0.35
        }
        return cognitive_scores.get(ability, 0.5)
    
    async def _assess_integration(self, integration_type: str) -> float:
        """Assess integration capability level"""
        integration_scores = {
            "multi_tool": 0.72,
            "context": 0.65,
            "error_recovery": 0.58,
            "adaptation": 0.42
        }
        return integration_scores.get(integration_type, 0.5)
    
    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        return {
            "task_success_rate": 0.87,
            "average_response_time": 3.2,
            "error_recovery_rate": 0.75,
            "learning_efficiency": 0.62,
            "adaptation_speed": 0.55,
            "consistency_score": 0.78
        }
    
    async def _catalog_knowledge_areas(self) -> List[str]:
        """Catalog current knowledge areas"""
        return [
            "software_development",
            "python_programming",
            "web_technologies",
            "database_systems",
            "machine_learning_basics",
            "system_architecture",
            "debugging_techniques",
            "code_quality_practices",
            "version_control",
            "testing_methodologies"
        ]
    
    async def _assess_skill_levels(self) -> Dict[str, float]:
        """Assess current skill levels across all areas"""
        return {
            "programming": 0.78,
            "system_design": 0.65,
            "problem_solving": 0.72,
            "communication": 0.68,
            "learning": 0.58,
            "adaptation": 0.52,
            "creativity": 0.48,
            "collaboration": 0.35
        }
    
    async def _identify_current_limitations(self) -> List[str]:
        """Identify current system limitations"""
        return [
            "Limited autonomous learning capability",
            "No persistent memory across sessions",
            "Requires human approval for self-modification",
            "Limited understanding of user intent",
            "No real-time learning from interactions",
            "Constrained by current model knowledge cutoff",
            "Limited multimodal capabilities",
            "No distributed processing abilities"
        ]
    
    async def _analyze_capability_evolution(self) -> Dict[str, Any]:
        """Analyze the evolution of system capabilities over time"""
        
        if len(self.capability_snapshots) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 snapshots for comparison"}
        
        # Compare latest snapshot with previous ones
        latest = self.capability_snapshots[-1]
        previous = self.capability_snapshots[-2]
        baseline = self.capability_snapshots[0] if len(self.capability_snapshots) > 2 else previous
        
        evolution_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "snapshots_analyzed": len(self.capability_snapshots),
            "time_span_days": self._calculate_time_span(),
            "recent_changes": await self._compare_snapshots(previous, latest),
            "overall_progress": await self._compare_snapshots(baseline, latest),
            "capability_trends": await self._analyze_capability_trends(),
            "learning_velocity": await self._calculate_learning_velocity(),
            "emerging_capabilities": await self._identify_emerging_capabilities()
        }
        
        return evolution_data
    
    async def _compare_snapshots(self, snapshot1: CapabilitySnapshot, snapshot2: CapabilitySnapshot) -> Dict[str, Any]:
        """Compare two capability snapshots"""
        
        comparison = {
            "time_difference": self._calculate_time_difference(snapshot1.timestamp, snapshot2.timestamp),
            "capability_changes": {},
            "performance_changes": {},
            "skill_changes": {},
            "new_knowledge_areas": [],
            "resolved_limitations": [],
            "overall_improvement_score": 0.0
        }
        
        # Compare capabilities
        for category, capabilities in snapshot2.capabilities.items():
            if category in snapshot1.capabilities:
                category_changes = {}
                for capability, level in capabilities.items():
                    if capability in snapshot1.capabilities[category]:
                        old_level = snapshot1.capabilities[category][capability]
                        change = level - old_level
                        if abs(change) > 0.01:
                            category_changes[capability] = {
                                "old": old_level,
                                "new": level,
                                "change": change,
                                "improvement": change > 0
                            }
                if category_changes:
                    comparison["capability_changes"][category] = category_changes
        
        # Compare performance metrics
        for metric, value in snapshot2.performance_metrics.items():
            if metric in snapshot1.performance_metrics:
                old_value = snapshot1.performance_metrics[metric]
                change = value - old_value
                if abs(change) > 0.01:
                    comparison["performance_changes"][metric] = {
                        "old": old_value,
                        "new": value,
                        "change": change,
                        "improvement": change > 0
                    }
        
        # Find new knowledge areas
        old_knowledge = set(snapshot1.knowledge_areas)
        new_knowledge = set(snapshot2.knowledge_areas)
        comparison["new_knowledge_areas"] = list(new_knowledge - old_knowledge)
        
        # Find resolved limitations
        old_limitations = set(snapshot1.limitations)
        new_limitations = set(snapshot2.limitations)
        comparison["resolved_limitations"] = list(old_limitations - new_limitations)
        
        # Calculate overall improvement score
        comparison["overall_improvement_score"] = await self._calculate_improvement_score(comparison)
        
        return comparison
    
    async def _record_learning_event(self, event_info: Dict[str, Any]):
        """Record a significant learning event"""
        
        event = LearningEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_info.get('type', 'knowledge_gained'),
            description=event_info.get('description', ''),
            before_state=event_info.get('before_state', {}),
            after_state=event_info.get('after_state', {}),
            impact_score=event_info.get('impact_score', 0.5),
            confidence=event_info.get('confidence', 0.8)
        )
        
        self.learning_events.append(event)
        await self._persist_learning_event(event)
        
        # Check if this constitutes a milestone
        if event.impact_score > self.milestone_threshold:
            await self._consider_milestone(event)
    
    async def _analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress over time"""
        
        recent_events = [e for e in self.learning_events 
                        if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(days=30)]
        
        progress = {
            "recent_events_count": len(recent_events),
            "average_impact": sum(e.impact_score for e in recent_events) / len(recent_events) if recent_events else 0,
            "skill_progression": await self._analyze_skill_progression(),
            "knowledge_growth": await self._analyze_knowledge_growth(),
            "efficiency_trends": await self._analyze_efficiency_trends()
        }
        
        return progress
    
    async def _analyze_skill_progression(self) -> Dict[str, Any]:
        """Analyze skill progression over time"""
        
        progression = {}
        if len(self.capability_snapshots) >= 2:
            latest = self.capability_snapshots[-1]
            previous = self.capability_snapshots[-2]
            
            for skill, current_level in latest.skill_levels.items():
                if skill in previous.skill_levels:
                    previous_level = previous.skill_levels[skill]
                    trend = current_level - previous_level
                    progression[skill] = {
                        "current": current_level,
                        "previous": previous_level,
                        "trend": trend
                    }
        
        return progression
    
    async def _analyze_knowledge_growth(self) -> Dict[str, Any]:
        """Analyze knowledge growth"""
        
        if len(self.capability_snapshots) >= 2:
            latest = self.capability_snapshots[-1]
            previous = self.capability_snapshots[-2]
            
            new_areas = len(set(latest.knowledge_areas) - set(previous.knowledge_areas))
            
            return {
                "new_areas": new_areas,
                "depth_increase": 0.05,  # Simulated
                "connections": 3  # Simulated cross-domain connections
            }
        
        return {"new_areas": 0, "depth_increase": 0, "connections": 0}
    
    async def _analyze_efficiency_trends(self) -> Dict[str, Any]:
        """Analyze learning efficiency trends"""
        
        return {
            "speed_trend": "improving",
            "retention": 0.8,
            "transfer": 0.6
        }
    
    async def _detect_milestones(self) -> List[EvolutionMilestone]:
        """Detect significant evolution milestones"""
        
        new_milestones = []
        
        # Check for capability breakthroughs
        if len(self.capability_snapshots) >= 2:
            recent_comparison = await self._compare_snapshots(
                self.capability_snapshots[-2], 
                self.capability_snapshots[-1]
            )
            
            if recent_comparison["overall_improvement_score"] > 0.2:
                milestone = EvolutionMilestone(
                    timestamp=datetime.now().isoformat(),
                    milestone_type="capability_breakthrough",
                    title="Significant Capability Improvement",
                    description=f"Overall improvement score: {recent_comparison['overall_improvement_score']:.2f}",
                    significance="major",
                    evidence=[
                        f"Improved {len(recent_comparison['capability_changes'])} capability categories",
                        f"Gained {len(recent_comparison['new_knowledge_areas'])} new knowledge areas"
                    ],
                    metrics_improvement=recent_comparison["performance_changes"]
                )
                new_milestones.append(milestone)
        
        # Add to milestones list
        for milestone in new_milestones:
            self.evolution_milestones.append(milestone)
            await self._persist_milestone(milestone)
        
        return new_milestones
    
    async def _predict_growth_trajectory(self) -> Dict[str, Any]:
        """Predict future capability development"""
        
        predictions = {
            "prediction_horizon_days": 90,
            "confidence_level": 0.7,
            "capability_predictions": {},
            "milestone_predictions": [],
            "agi_progress_estimate": {}
        }
        
        # Predict capability growth based on trends
        if len(self.capability_snapshots) >= 2:
            latest = self.capability_snapshots[-1]
            previous = self.capability_snapshots[-2]
            
            for category, capabilities in latest.capabilities.items():
                if category in previous.capabilities:
                    for capability, current_level in capabilities.items():
                        if capability in previous.capabilities[category]:
                            previous_level = previous.capabilities[category][capability]
                            trend = current_level - previous_level
                            
                            if abs(trend) > 0.01:
                                predicted_level = current_level + (trend * 3)  # 3 intervals ahead
                                predicted_level = max(0.0, min(1.0, predicted_level))
                                
                                predictions["capability_predictions"][f"{category}.{capability}"] = {
                                    "current": current_level,
                                    "predicted": predicted_level,
                                    "trend": trend,
                                    "confidence": 0.7 if abs(trend) > 0.05 else 0.4
                                }
        
        return predictions
    
    async def _assess_agi_progress(self) -> Dict[str, Any]:
        """Assess progress toward AGI capabilities"""
        
        # Define AGI capability requirements
        agi_requirements = {
            "autonomous_learning": 0.8,
            "general_problem_solving": 0.85,
            "creative_reasoning": 0.75,
            "self_improvement": 0.7,
            "adaptive_behavior": 0.8,
            "knowledge_synthesis": 0.8,
            "meta_cognition": 0.7
        }
        
        # Assess current state against requirements
        current_state = await self._assess_current_capabilities()
        
        agi_assessment = {
            "overall_agi_progress": 0.0,
            "requirement_status": {},
            "achieved_requirements": [],
            "critical_gaps": [],
            "estimated_agi_timeline": None
        }
        
        total_progress = 0.0
        
        for requirement, threshold in agi_requirements.items():
            current_level = await self._map_to_agi_requirement(requirement, current_state)
            progress = current_level / threshold
            
            agi_assessment["requirement_status"][requirement] = {
                "current_level": current_level,
                "required_level": threshold,
                "progress": min(1.0, progress),
                "gap": max(0.0, threshold - current_level),
                "achieved": current_level >= threshold
            }
            
            if current_level >= threshold:
                agi_assessment["achieved_requirements"].append(requirement)
            else:
                agi_assessment["critical_gaps"].append({
                    "requirement": requirement,
                    "gap": threshold - current_level
                })
            
            total_progress += min(1.0, progress)
        
        agi_assessment["overall_agi_progress"] = total_progress / len(agi_requirements)
        
        return agi_assessment
    
    async def _map_to_agi_requirement(self, requirement: str, current_state: Dict[str, Any]) -> float:
        """Map current capabilities to AGI requirements"""
        
        mapping = {
            "autonomous_learning": current_state.get("cognitive_abilities", {}).get("meta_learning", 0.3),
            "general_problem_solving": current_state.get("core_functions", {}).get("problem_solving", 0.5),
            "creative_reasoning": current_state.get("cognitive_abilities", {}).get("creative_problem_solving", 0.4),
            "self_improvement": current_state.get("core_functions", {}).get("self_reflection", 0.4),
            "adaptive_behavior": current_state.get("integration_capabilities", {}).get("adaptive_behavior", 0.3),
            "knowledge_synthesis": current_state.get("cognitive_abilities", {}).get("knowledge_synthesis", 0.5),
            "meta_cognition": current_state.get("cognitive_abilities", {}).get("abstract_reasoning", 0.4)
        }
        
        return mapping.get(requirement, 0.3)
    
    # Utility methods
    def _calculate_time_span(self) -> float:
        """Calculate time span of snapshots in days"""
        if len(self.capability_snapshots) < 2:
            return 0.0
        
        first = datetime.fromisoformat(self.capability_snapshots[0].timestamp)
        last = datetime.fromisoformat(self.capability_snapshots[-1].timestamp)
        return (last - first).total_seconds() / 86400
    
    def _calculate_time_difference(self, timestamp1: str, timestamp2: str) -> float:
        """Calculate time difference in hours"""
        t1 = datetime.fromisoformat(timestamp1)
        t2 = datetime.fromisoformat(timestamp2)
        return abs((t2 - t1).total_seconds() / 3600)
    
    async def _calculate_improvement_score(self, comparison: Dict[str, Any]) -> float:
        """Calculate overall improvement score from comparison"""
        score = 0.0
        
        # Score capability improvements
        for category, changes in comparison["capability_changes"].items():
            for capability, change_info in changes.items():
                if change_info["improvement"]:
                    score += change_info["change"] * 0.5
        
        # Score performance improvements
        for metric, change_info in comparison["performance_changes"].items():
            if change_info["improvement"]:
                score += change_info["change"] * 0.3
        
        # Score new knowledge and resolved limitations
        score += len(comparison["new_knowledge_areas"]) * 0.05
        score += len(comparison["resolved_limitations"]) * 0.1
        
        return score
    
    async def _calculate_learning_velocity(self) -> Dict[str, float]:
        """Calculate the velocity of learning across different areas"""
        
        if len(self.learning_events) < 2:
            return {"overall": 0.0}
        
        recent_events = [e for e in self.learning_events 
                        if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(days=30)]
        
        velocity_by_type = {}
        for event_type in ["skill_acquired", "capability_enhanced", "knowledge_gained"]:
            type_events = [e for e in recent_events if e.event_type == event_type]
            if type_events:
                total_impact = sum(e.impact_score for e in type_events)
                velocity_by_type[event_type] = total_impact / 30
        
        overall_velocity = sum(velocity_by_type.values())
        velocity_by_type["overall"] = overall_velocity
        
        return velocity_by_type
    
    async def _analyze_capability_trends(self) -> Dict[str, Any]:
        """Analyze trends in capability development"""
        
        if len(self.capability_snapshots) < 3:
            return {"status": "insufficient_data"}
        
        trends = {
            "capability_trajectories": {},
            "learning_acceleration": {},
            "stagnation_areas": [],
            "breakthrough_candidates": []
        }
        
        return trends
    
    async def _identify_emerging_capabilities(self) -> List[Dict[str, Any]]:
        """Identify capabilities that are emerging or developing"""
        
        emerging = []
        
        if len(self.capability_snapshots) >= 2:
            latest = self.capability_snapshots[-1]
            previous = self.capability_snapshots[-2]
            
            for category, capabilities in latest.capabilities.items():
                if category in previous.capabilities:
                    for capability, level in capabilities.items():
                        if capability in previous.capabilities[category]:
                            old_level = previous.capabilities[category][capability]
                            improvement = level - old_level
                            
                            if improvement > 0.1 and level < 0.7:
                                emerging.append({
                                    "capability": f"{category}.{capability}",
                                    "current_level": level,
                                    "recent_improvement": improvement,
                                    "emergence_stage": "developing" if level < 0.5 else "maturing"
                                })
        
        return emerging
    
    # Formatting methods
    def _format_evolution_analysis(self, evolution_data: Dict[str, Any]) -> str:
        """Format evolution analysis as readable text"""
        
        if evolution_data.get("status") == "insufficient_data":
            return "Evolution Analysis: Insufficient data for analysis (need at least 2 capability snapshots)"
        
        analysis = f"""
# Capability Evolution Analysis
**Analysis Date:** {evolution_data['analysis_timestamp']}
**Snapshots Analyzed:** {evolution_data['snapshots_analyzed']}
**Time Span:** {evolution_data['time_span_days']:.1f} days

## Recent Changes
**Overall Improvement Score:** {evolution_data['recent_changes']['overall_improvement_score']:.3f}

### New Knowledge Areas
{', '.join(evolution_data['recent_changes']['new_knowledge_areas']) if evolution_data['recent_changes']['new_knowledge_areas'] else 'None'}

### Resolved Limitations
{', '.join(evolution_data['recent_changes']['resolved_limitations']) if evolution_data['recent_changes']['resolved_limitations'] else 'None'}

## Learning Velocity
**Overall:** {evolution_data['learning_velocity'].get('overall', 0):.3f} impact/day

## Emerging Capabilities
{self._format_emerging_capabilities(evolution_data['emerging_capabilities'])}
        """.strip()
        
        return analysis
    
    def _format_emerging_capabilities(self, emerging: List[Dict[str, Any]]) -> str:
        """Format emerging capabilities"""
        if not emerging:
            return "No emerging capabilities detected"
        
        formatted = ""
        for capability in emerging:
            stage_icon = "🌱" if capability["emergence_stage"] == "developing" else "🌿"
            formatted += f"- {stage_icon} **{capability['capability']}** ({capability['emergence_stage']})\n"
            formatted += f"  Current: {capability['current_level']:.2f}, Recent improvement: +{capability['recent_improvement']:.2f}\n"
        
        return formatted.strip()
    
    def _format_capability_snapshot(self, snapshot: CapabilitySnapshot) -> str:
        """Format capability snapshot"""
        
        formatted = f"""
# Capability Snapshot {snapshot.version}
**Timestamp:** {snapshot.timestamp}
**Checksum:** {snapshot.checksum}

## Core Capabilities
{self._format_capability_category(snapshot.capabilities.get('core_functions', {}))}

## Technical Skills
{self._format_capability_category(snapshot.capabilities.get('technical_skills', {}))}

## Cognitive Abilities
{self._format_capability_category(snapshot.capabilities.get('cognitive_abilities', {}))}

## Performance Metrics
{self._format_metrics(snapshot.performance_metrics)}

## Knowledge Areas ({len(snapshot.knowledge_areas)} total)
{', '.join(snapshot.knowledge_areas)}

## Current Limitations ({len(snapshot.limitations)} total)
{chr(10).join(f"- {limitation}" for limitation in snapshot.limitations)}
        """.strip()
        
        return formatted
    
    def _format_capability_category(self, category: Dict[str, float]) -> str:
        """Format a capability category"""
        if not category:
            return "No data available"
        
        formatted = ""
        for capability, level in category.items():
            bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
            formatted += f"- **{capability.replace('_', ' ').title()}:** {level:.2f} [{bar}]\n"
        
        return formatted.strip()
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics"""
        formatted = ""
        for metric, value in metrics.items():
            formatted += f"- **{metric.replace('_', ' ').title()}:** {value:.2f}\n"
        
        return formatted.strip()
    
    def _format_learning_progress(self, progress: Dict[str, Any]) -> str:
        """Format learning progress analysis"""
        
        return f"""
# Learning Progress Analysis

## Learning Events Summary
- **Recent Events (30 days):** {progress.get('recent_events_count', 0)}
- **Average Impact Score:** {progress.get('average_impact', 0):.2f}

## Skill Progression
{self._format_skill_progression(progress.get('skill_progression', {}))}

## Knowledge Growth
- **New Knowledge Areas:** {progress.get('knowledge_growth', {}).get('new_areas', 0)}
- **Knowledge Depth Increase:** {progress.get('knowledge_growth', {}).get('depth_increase', 0):.2f}
- **Cross-Domain Connections:** {progress.get('knowledge_growth', {}).get('connections', 0)}

## Learning Efficiency Trends
- **Learning Speed:** {progress.get('efficiency_trends', {}).get('speed_trend', 'stable')}
- **Retention Rate:** {progress.get('efficiency_trends', {}).get('retention', 0.8):.1%}
- **Transfer Learning:** {progress.get('efficiency_trends', {}).get('transfer', 0.6):.1%}
        """.strip()
    
    def _format_skill_progression(self, progression: Dict[str, Any]) -> str:
        """Format skill progression"""
        if not progression:
            return "No skill progression data available"
        
        formatted = ""
        for skill, data in progression.items():
            trend = "↗️" if data.get('trend', 0) > 0 else "↘️" if data.get('trend', 0) < 0 else "→"
            formatted += f"- **{skill.replace('_', ' ').title()}:** {data.get('current', 0):.2f} {trend}\n"
        
        return formatted.strip()
    
    def _format_milestones(self, milestones: List[EvolutionMilestone]) -> str:
        """Format evolution milestones"""
        if not milestones:
            return "No new milestones detected"
        
        formatted = "# Evolution Milestones\n\n"
        
        significance_icons = {
            "minor": "🟢",
            "major": "🟡", 
            "breakthrough": "🔥"
        }
        
        for milestone in milestones:
            icon = significance_icons.get(milestone.significance, "📍")
            formatted += f"""
## {icon} {milestone.title}
**Type:** {milestone.milestone_type}
**Significance:** {milestone.significance}
**Date:** {milestone.timestamp}

**Description:** {milestone.description}

**Evidence:**
{chr(10).join(f"- {evidence}" for evidence in milestone.evidence)}

---
            """.strip()
        
        return formatted
    
    def _format_growth_predictions(self, predictions: Dict[str, Any]) -> str:
        """Format growth predictions"""
        
        formatted = f"""
# Capability Growth Predictions
**Prediction Horizon:** {predictions['prediction_horizon_days']} days
**Confidence Level:** {predictions['confidence_level']:.1%}

## Capability Predictions
"""
        
        for capability, pred in predictions.get('capability_predictions', {}).items():
            change = pred['predicted'] - pred['current']
            trend_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            formatted += f"""
- **{capability.replace('_', ' ').replace('.', ' - ').title()}** {trend_icon}
  Current: {pred['current']:.2f} → Predicted: {pred['predicted']:.2f}
  Confidence: {pred['confidence']:.1%}
"""
        
        if predictions.get('milestone_predictions'):
            formatted += "\n## Predicted Milestones\n"
            for milestone in predictions['milestone_predictions']:
                formatted += f"- **{milestone['type']}** in ~{milestone['estimated_days']} days (confidence: {milestone['confidence']:.1%})\n"
        
        return formatted.strip()
    
    def _format_agi_assessment(self, assessment: Dict[str, Any]) -> str:
        """Format AGI progress assessment"""
        
        progress_bar = "█" * int(assessment['overall_agi_progress'] * 20) + "░" * (20 - int(assessment['overall_agi_progress'] * 20))
        
        formatted = f"""
# AGI Progress Assessment

## Overall Progress
**AGI Progress:** {assessment['overall_agi_progress']:.1%} [{progress_bar}]

## Requirement Status
"""
        
        for requirement, status in assessment['requirement_status'].items():
            status_icon = "✅" if status['achieved'] else "❌"
            req_bar = "█" * int(status['progress'] * 10) + "░" * (10 - int(status['progress'] * 10))
            formatted += f"- {status_icon} **{requirement.replace('_', ' ').title()}:** {status['current_level']:.2f}/{status['required_level']:.2f} [{req_bar}]\n"
        
        if assessment['achieved_requirements']:
            formatted += f"\n## ✅ Achieved Requirements ({len(assessment['achieved_requirements'])})\n"
            for req in assessment['achieved_requirements']:
                formatted += f"- {req.replace('_', ' ').title()}\n"
        
        if assessment['critical_gaps']:
            formatted += f"\n## ❌ Critical Gaps ({len(assessment['critical_gaps'])})\n"
            for gap in assessment['critical_gaps']:
                formatted += f"- **{gap['requirement'].replace('_', ' ').title()}:** Gap of {gap['gap']:.2f}\n"
        
        if assessment.get('estimated_agi_timeline'):
            timeline = assessment['estimated_agi_timeline']
            formatted += f"\n## 🎯 Estimated Timeline\n"
            formatted += f"**Time to AGI:** ~{timeline['days']} days ({timeline['days']/365:.1f} years)\n"
            formatted += f"**Confidence:** {timeline['confidence']:.1%}\n"
        
        return formatted.strip()
    
    # Persistence methods
    async def _persist_snapshot(self, snapshot: CapabilitySnapshot):
        """Persist capability snapshot"""
        try:
            snapshots_file = Path("memory/logs/capability_snapshots.jsonl")
            snapshots_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(snapshots_file, "a") as f:
                f.write(json.dumps(asdict(snapshot)) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to persist snapshot: {e}")
    
    async def _persist_learning_event(self, event: LearningEvent):
        """Persist learning event"""
        try:
            events_file = Path("memory/logs/learning_events.jsonl")
            events_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(events_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to persist learning event: {e}")
    
    async def _persist_milestone(self, milestone: EvolutionMilestone):
        """Persist evolution milestone"""
        try:
            milestones_file = Path("memory/logs/evolution_milestones.jsonl")
            milestones_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(milestones_file, "a") as f:
                f.write(json.dumps(asdict(milestone)) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to persist milestone: {e}")
    
    async def _consider_milestone(self, event: LearningEvent):
        """Consider if a learning event constitutes a milestone"""
        if event.impact_score > self.milestone_threshold:
            milestone = EvolutionMilestone(
                timestamp=event.timestamp,
                milestone_type="learning_breakthrough",
                title=f"Significant Learning: {event.event_type}",
                description=event.description,
                significance="major" if event.impact_score > 0.3 else "minor",
                evidence=[f"Impact score: {event.impact_score}", f"Confidence: {event.confidence}"],
                metrics_improvement={"learning_impact": event.impact_score}
            )
            
            self.evolution_milestones.append(milestone)
            await self._persist_milestone(milestone)
    
    async def start(self):
        """Start the evolution tracking MCP server"""
        self._log_start()
        await self.server.start()
    
    async def stop(self):
        """Stop the evolution tracking MCP server"""
        await self.server.stop()
        self._log_stop()
    
    async def _execute_action(self, action: str, **kwargs) -> str:
        """Execute evolution tracking actions"""
        if action == "track_evolution":
            evolution_data = await self._analyze_capability_evolution()
            return self._format_evolution_analysis(evolution_data)
        elif action == "create_snapshot":
            snapshot = await self._create_capability_snapshot()
            return self._format_capability_snapshot(snapshot)
        elif action == "assess_agi_progress":
            assessment = await self._assess_agi_progress()
            return self._format_agi_assessment(assessment)
        else:
            return f"Unknown evolution action: {action}"


# Factory function for dynamic loading
def create_server(config: Dict[str, Any]) -> EvolutionMCPServer:
    """Factory function to create EvolutionMCPServer instance"""
    return EvolutionMCPServer(config)


# For testing
if __name__ == "__main__":
    async def main():
        config = {
            "name": "evolution",
            "type": "self_reflection/evolution",
            "snapshot_interval": 86400,
            "milestone_threshold": 0.15,
            "track_code_changes": True
        }
        
        server = EvolutionMCPServer(config)
        await server.start()
        
        # Test capability snapshot
        snapshot_result = await server._execute_action("create_snapshot")
        print("Capability Snapshot:")
        print(snapshot_result)
        
        # Test AGI assessment
        agi_result = await server._execute_action("assess_agi_progress")
        print("\nAGI Progress Assessment:")
        print(agi_result)
        
        await server.stop()
    
    asyncio.run(main())

