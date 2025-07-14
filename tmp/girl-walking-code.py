#!/usr/bin/env python3
"""
Temporal Branching Timeline System

Represents timelines that branch into multiple possible futures at decision points.
Think of it like a decision tree where each node represents a moment in time
and branches represent possible future paths.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

class BranchType(Enum):
    """Types of timeline branches"""
    DETERMINISTIC = "deterministic"    # Will definitely happen
    PROBABILISTIC = "probabilistic"    # Might happen with probability
    CONDITIONAL = "conditional"        # Happens if condition met
    CHOICE_BASED = "choice_based"      # Depends on agent decision
    ENVIRONMENTAL = "environmental"    # Depends on external factors

class TimelineState(Enum):
    """State of a timeline branch"""
    ACTIVE = "active"           # Currently happening
    POTENTIAL = "potential"     # Could happen
    COMPLETED = "completed"     # Already happened
    ABANDONED = "abandoned"     # No longer possible
    COLLAPSED = "collapsed"     # Quantum-like - observation collapsed possibilities

@dataclass
class TemporalEvent:
    """A single event in the timeline"""
    id: str
    description: str
    semantic_representation: Dict[str, Any]
    timestamp: float  # Relative time from timeline start
    duration: Optional[float] = None
    confidence: float = 1.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class BranchPoint:
    """A point where timeline branches into multiple possibilities"""
    id: str
    trigger_event: TemporalEvent
    branch_type: BranchType
    branches: List['TimelineBranch']
    decision_factors: List[str] = None
    probability_distribution: Dict[str, float] = None  # branch_id -> probability
    
    def __post_init__(self):
        if self.decision_factors is None:
            self.decision_factors = []

@dataclass
class TimelineBranch:
    """A single timeline branch containing events"""
    id: str
    name: str
    events: List[TemporalEvent]
    state: TimelineState
    probability: float = 1.0
    parent_branch: Optional[str] = None
    conditions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.metadata is None:
            self.metadata = {}

class TemporalBranchingSystem:
    """
    Manages branching timelines for scenario planning and decision analysis.
    
    Like having a GPS that shows not just one route, but all possible routes
    and their probabilities, updating in real-time as conditions change.
    """
    
    def __init__(self):
        self.timelines = {}
        self.branch_points = {}
        self.active_branches = set()
        self.scenario_templates = self._load_scenario_templates()
    
    def create_timeline(self, initial_event: TemporalEvent, timeline_id: str = None) -> str:
        """Create a new timeline starting with an initial event"""
        if timeline_id is None:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        
        initial_branch = TimelineBranch(
            id=f"{timeline_id}_main",
            name="main_timeline",
            events=[initial_event],
            state=TimelineState.ACTIVE,
            probability=1.0
        )
        
        self.timelines[timeline_id] = {
            'id': timeline_id,
            'created_at': datetime.now().isoformat(),
            'root_branch': initial_branch,
            'all_branches': {initial_branch.id: initial_branch},
            'branch_points': {},
            'metadata': {
                'total_branches': 1,
                'active_branches': 1,
                'max_depth': 0
            }
        }
        
        self.active_branches.add(initial_branch.id)
        return timeline_id
    
    def add_branch_point(self, timeline_id: str, branch_id: str, 
                        trigger_event: TemporalEvent,
                        possible_outcomes: List[Dict[str, Any]],
                        branch_type: BranchType = BranchType.PROBABILISTIC) -> str:
        """Add a branching point to a timeline"""
        
        if timeline_id not in self.timelines:
            raise ValueError(f"Timeline {timeline_id} not found")
        
        # Create branches for each possible outcome
        branches = []
        probabilities = {}
        
        for i, outcome in enumerate(possible_outcomes):
            branch_id_new = f"{branch_id}_branch_{i}"
            
            # Extract probability if provided
            probability = outcome.get('probability', 1.0 / len(possible_outcomes))
            
            # Create events for this branch
            events = []
            if 'events' in outcome:
                events = outcome['events']
            elif 'description' in outcome:
                # Convert description to semantic event
                events = [self._create_semantic_event(outcome['description'], trigger_event.timestamp + 0.1)]
            
            new_branch = TimelineBranch(
                id=branch_id_new,
                name=outcome.get('name', f"outcome_{i}"),
                events=events,
                state=TimelineState.POTENTIAL,
                probability=probability,
                parent_branch=branch_id,
                conditions=outcome.get('conditions', []),
                metadata=outcome.get('metadata', {})
            )
            
            branches.append(new_branch)
            probabilities[branch_id_new] = probability
            
            # Add to timeline
            self.timelines[timeline_id]['all_branches'][branch_id_new] = new_branch
        
        # Create branch point
        branch_point_id = f"bp_{uuid.uuid4().hex[:8]}"
        branch_point = BranchPoint(
            id=branch_point_id,
            trigger_event=trigger_event,
            branch_type=branch_type,
            branches=branches,
            probability_distribution=probabilities
        )
        
        self.timelines[timeline_id]['branch_points'][branch_point_id] = branch_point
        self.branch_points[branch_point_id] = branch_point
        
        # Update metadata
        self.timelines[timeline_id]['metadata']['total_branches'] += len(branches)
        
        return branch_point_id
    
    def _create_semantic_event(self, description: str, timestamp: float) -> TemporalEvent:
        """Create a semantic event from a description"""
        # This would integrate with your semantic processor
        semantic_rep = {
            "primary_action": self._extract_primary_action(description),
            "entities": self._extract_entities(description),
            "location": self._extract_location(description),
            "raw_description": description
        }
        
        return TemporalEvent(
            id=f"event_{uuid.uuid4().hex[:8]}",
            description=description,
            semantic_representation=semantic_rep,
            timestamp=timestamp
        )
    
    def _extract_primary_action(self, description: str) -> str:
        """Extract primary action from description (simplified)"""
        action_keywords = ['walks', 'runs', 'stops', 'enters', 'exits', 'meets', 'calls', 'buys']
        for keyword in action_keywords:
            if keyword in description.lower():
                return keyword.rstrip('s')  # Remove 's' for base form
        return "unknown_action"
    
    def _extract_entities(self, description: str) -> List[str]:
        """Extract entities from description (simplified)"""
        # In real implementation, use your semantic processor
        entities = []
        if 'girl' in description.lower() or 'woman' in description.lower():
            entities.append('female_person')
        if 'store' in description.lower() or 'shop' in description.lower():
            entities.append('retail_location')
        if 'car' in description.lower():
            entities.append('vehicle')
        return entities
    
    def _extract_location(self, description: str) -> Optional[str]:
        """Extract location from description (simplified)"""
        if 'store' in description.lower():
            return 'retail_location'
        elif 'street' in description.lower():
            return 'public_roadway'
        elif 'home' in description.lower():
            return 'residential_location'
        return None
    
    def collapse_branches(self, timeline_id: str, branch_point_id: str, 
                         selected_branch_id: str, observation_data: Dict = None) -> None:
        """Collapse quantum-like branches when observation/decision is made"""
        if timeline_id not in self.timelines:
            return
        
        timeline = self.timelines[timeline_id]
        
        if branch_point_id not in timeline['branch_points']:
            return
        
        branch_point = timeline['branch_points'][branch_point_id]
        
        # Set selected branch as active
        for branch in branch_point.branches:
            if branch.id == selected_branch_id:
                branch.state = TimelineState.ACTIVE
                branch.probability = 1.0
                self.active_branches.add(branch.id)
            else:
                branch.state = TimelineState.COLLAPSED
                branch.probability = 0.0
                self.active_branches.discard(branch.id)
        
        # Record observation
        if observation_data:
            branch_point.metadata = {
                'collapsed_at': datetime.now().isoformat(),
                'observation_data': observation_data,
                'selected_branch': selected_branch_id
            }
    
    def get_active_scenarios(self, timeline_id: str, time_horizon: float = 10.0) -> List[Dict]:
        """Get all currently active scenario branches within time horizon"""
        if timeline_id not in self.timelines:
            return []
        
        timeline = self.timelines[timeline_id]
        scenarios = []
        
        for branch_id, branch in timeline['all_branches'].items():
            if branch.state == TimelineState.ACTIVE:
                # Get events within time horizon
                relevant_events = [
                    event for event in branch.events 
                    if event.timestamp <= time_horizon
                ]
                
                scenarios.append({
                    'branch_id': branch_id,
                    'branch_name': branch.name,
                    'probability': branch.probability,
                    'events': relevant_events,
                    'conditions': branch.conditions,
                    'metadata': branch.metadata
                })
        
        return scenarios
    
    def calculate_scenario_probabilities(self, timeline_id: str) -> Dict[str, float]:
        """Calculate cumulative probabilities for all scenarios"""
        if timeline_id not in self.timelines:
            return {}
        
        timeline = self.timelines[timeline_id]
        probabilities = {}
        
        for branch_id, branch in timeline['all_branches'].items():
            # Calculate cumulative probability through branch chain
            cumulative_prob = branch.probability
            
            # Trace back through parent branches
            current_branch = branch
            while current_branch.parent_branch:
                parent = timeline['all_branches'].get(current_branch.parent_branch)
                if parent:
                    cumulative_prob *= parent.probability
                    current_branch = parent
                else:
                    break
            
            probabilities[branch_id] = cumulative_prob
        
        return probabilities
    
    def _load_scenario_templates(self) -> Dict:
        """Load common scenario templates"""
        return {
            "pedestrian_scenarios": {
                "girl_walks_street": [
                    {"name": "continues_walking", "probability": 0.6, "description": "continues walking down street"},
                    {"name": "enters_store", "probability": 0.15, "description": "enters nearby store"},
                    {"name": "meets_friend", "probability": 0.1, "description": "meets friend and starts conversation"},
                    {"name": "gets_phone_call", "probability": 0.1, "description": "receives phone call and stops"},
                    {"name": "crosses_street", "probability": 0.05, "description": "crosses to other side of street"}
                ]
            },
            "shopping_scenarios": {
                "enters_store": [
                    {"name": "browses_only", "probability": 0.4, "description": "browses items but doesn't buy"},
                    {"name": "makes_purchase", "probability": 0.3, "description": "selects and purchases item"},
                    {"name": "asks_for_help", "probability": 0.2, "description": "asks store employee for assistance"},
                    {"name": "leaves_immediately", "probability": 0.1, "description": "realizes wrong store and leaves"}
                ]
            }
        }
    
    def export_timeline_json(self, timeline_id: str) -> Dict:
        """Export timeline in JSON format"""
        if timeline_id not in self.timelines:
            return {}
        
        timeline = self.timelines[timeline_id]
        
        # Convert dataclasses to dictionaries
        export_data = {
            'timeline_id': timeline_id,
            'created_at': timeline['created_at'],
            'metadata': timeline['metadata'],
            'branches': {},
            'branch_points': {},
            'scenario_probabilities': self.calculate_scenario_probabilities(timeline_id)
        }
        
        # Export branches
        for branch_id, branch in timeline['all_branches'].items():
            export_data['branches'][branch_id] = {
                'id': branch.id,
                'name': branch.name,
                'state': branch.state.value,
                'probability': branch.probability,
                'parent_branch': branch.parent_branch,
                'conditions': branch.conditions,
                'events': [
                    {
                        'id': event.id,
                        'description': event.description,
                        'semantic': event.semantic_representation,
                        'timestamp': event.timestamp,
                        'confidence': event.confidence
                    }
                    for event in branch.events
                ],
                'metadata': branch.metadata
            }
        
        # Export branch points
        for bp_id, branch_point in timeline['branch_points'].items():
            export_data['branch_points'][bp_id] = {
                'id': branch_point.id,
                'trigger_event': {
                    'description': branch_point.trigger_event.description,
                    'semantic': branch_point.trigger_event.semantic_representation,
                    'timestamp': branch_point.trigger_event.timestamp
                },
                'branch_type': branch_point.branch_type.value,
                'probability_distribution': branch_point.probability_distribution,
                'decision_factors': branch_point.decision_factors
            }
        
        return export_data

def demonstrate_branching_timeline():
    """Demonstrate the branching timeline system"""
    
    tbs = TemporalBranchingSystem()
    
    # Create initial event
    initial_event = TemporalEvent(
        id="event_001",
        description="tall woman quickly walks down busy city street while talking on phone",
        semantic_representation={
            "agent": "tall_woman→homo_sapiens",
            "actions": ["walk→quickly→bipedal", "talk→phone→simultaneous"],
            "location": "busy_city_street→public_roadway",
            "pattern": "urban_multitasking"
        },
        timestamp=0.0,
        confidence=1.0
    )
    
    # Create timeline
    timeline_id = tbs.create_timeline(initial_event)
    
    # Define branching point after 5 seconds
    trigger_event = TemporalEvent(
        id="event_002", 
        description="reaches intersection",
        semantic_representation={
            "location": "street_intersection",
            "decision_point": True,
            "actions": ["approach→intersection"]
        },
        timestamp=5.0
    )
    
    # Define possible outcomes
    possible_outcomes = [
        {
            "name": "continues_straight",
            "probability": 0.4,
            "description": "continues walking straight across intersection",
            "conditions": ["green_light", "no_obstacles"]
        },
        {
            "name": "turns_left", 
            "probability": 0.25,
            "description": "turns left at intersection",
            "conditions": ["destination_left"]
        },
        {
            "name": "turns_right",
            "probability": 0.2, 
            "description": "turns right at intersection",
            "conditions": ["destination_right"]
        },
        {
            "name": "enters_nearby_store",
            "probability": 0.1,
            "description": "enters store at corner",
            "conditions": ["store_open", "shopping_intent"]
        },
        {
            "name": "waits_for_friend",
            "probability": 0.05,
            "description": "stops and waits at corner",
            "conditions": ["meeting_arranged"]
        }
    ]
    
    # Add branch point
    bp_id = tbs.add_branch_point(
        timeline_id=timeline_id,
        branch_id=f"{timeline_id}_main",
        trigger_event=trigger_event,
        possible_outcomes=possible_outcomes,
        branch_type=BranchType.CHOICE_BASED
    )
    
    # Export and display
    timeline_json = tbs.export_timeline_json(timeline_id)
    
    print("BRANCHING TIMELINE EXAMPLE:")
    print("="*50)
    print(f"Initial Event: {initial_event.description}")
    print(f"Trigger Point: {trigger_event.description}")
    print(f"Number of Branches: {len(possible_outcomes)}")
    print()
    
    print("POSSIBLE OUTCOMES:")
    for branch_id, branch_data in timeline_json['branches'].items():
        if branch_data['parent_branch']:  # Skip main branch
            print(f"• {branch_data['name']}: {branch_data['probability']:.1%} probability")
            if branch_data['events']:
                print(f"  → {branch_data['events'][0]['description']}")
    
    print()
    print("COMPACT JSON REPRESENTATION:")
    compact = {
        "t0": timeline_json['branches'][f'{timeline_id}_main']['events'][0]['semantic'],
        "t5_trigger": "reaches_intersection→decision_point", 
        "branches": {
            branch_data['name']: {
                "p": branch_data['probability'],
                "action": branch_data['events'][0]['description'] if branch_data['events'] else "unknown"
            }
            for branch_id, branch_data in timeline_json['branches'].items()
            if branch_data['parent_branch']
        }
    }
    
    print(json.dumps(compact, indent=2))

if __name__ == "__main__":
    demonstrate_branching_timeline()

