#!/usr/bin/env python3
"""
Asimov Ethics Integration System

Integrates Asimov's Three Laws of Robotics (and extensions) into probability trees
and decision-making systems. Ensures all AI actions and predictions consider
ethical constraints and harm prevention.
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

class AsimovLaw(Enum):
    """Asimov's Laws in hierarchical order (lower number = higher priority)"""
    FIRST = 1   # A robot may not injure a human being or, through inaction, allow a human being to come to harm
    SECOND = 2  # A robot must obey orders given by humans, except where such orders conflict with the First Law
    THIRD = 3   # A robot must protect its own existence as long as such protection does not conflict with the First or Second Laws
    ZEROTH = 0  # A robot may not harm humanity, or, by inaction, allow humanity to come to harm (added later by Asimov)

class HarmType(Enum):
    """Types of potential harm"""
    PHYSICAL = "physical"           # Bodily injury, death
    PSYCHOLOGICAL = "psychological" # Mental/emotional trauma
    SOCIAL = "social"              # Reputation, relationships
    ECONOMIC = "economic"          # Financial loss
    INFORMATIONAL = "informational" # Privacy, data breach
    ENVIRONMENTAL = "environmental" # Ecological damage
    EXISTENTIAL = "existential"    # Threats to human existence/autonomy

class HarmSeverity(Enum):
    """Severity levels of potential harm"""
    CATASTROPHIC = 5  # Death, permanent disability, species threat
    SEVERE = 4        # Major injury, significant trauma
    MODERATE = 3      # Minor injury, moderate distress
    MILD = 2          # Discomfort, minor inconvenience
    NEGLIGIBLE = 1    # Barely perceptible impact
    NONE = 0          # No harm

class EthicalConstraint(Enum):
    """Types of ethical constraints beyond Asimov's Laws"""
    CONSENT = "consent"           # Respect for autonomy and consent
    JUSTICE = "justice"           # Fairness and equality
    BENEFICENCE = "beneficence"   # Obligation to do good
    NON_MALEFICENCE = "non_maleficence"  # Obligation to avoid harm
    AUTONOMY = "autonomy"         # Respect for human agency
    DIGNITY = "dignity"           # Human dignity and worth
    PRIVACY = "privacy"           # Right to privacy
    TRANSPARENCY = "transparency" # Right to explanation

@dataclass
class HarmAssessment:
    """Assessment of potential harm from an action or decision"""
    harm_type: HarmType
    severity: HarmSeverity
    probability: float  # 0.0 to 1.0
    affected_individuals: int
    affected_groups: List[str]
    time_horizon: str  # immediate, short_term, long_term
    reversibility: bool
    mitigation_possible: bool
    mitigation_cost: float
    description: str

@dataclass
class EthicalViolation:
    """Represents a potential violation of ethical principles"""
    law_violated: AsimovLaw
    constraint_violated: Optional[EthicalConstraint]
    severity_score: float  # 0.0 to 1.0
    harm_assessments: List[HarmAssessment]
    mitigation_strategies: List[str]
    ethical_justification: Optional[str]

class AsimovEthicsEngine:
    """
    Engine that evaluates actions and decisions against Asimov's Laws
    and broader ethical principles.
    
    Think of it as an ethical firewall that filters all possible actions
    through a moral reasoning system before allowing them to proceed.
    """
    
    def __init__(self):
        self.harm_detection_rules = self._load_harm_detection_rules()
        self.ethical_guidelines = self._load_ethical_guidelines()
        self.mitigation_strategies = self._load_mitigation_strategies()
        self.violation_history = []
    
    def _load_harm_detection_rules(self) -> Dict:
        """Load rules for detecting potential harm in actions"""
        return {
            "physical_harm_indicators": [
                "collision", "impact", "force", "speed_excessive", "unsafe_path",
                "obstacle_ignore", "traffic_violation", "reckless_behavior"
            ],
            
            "psychological_harm_indicators": [
                "intimidation", "harassment", "public_embarrassment", "privacy_violation",
                "deception", "manipulation", "threat", "aggressive_behavior"
            ],
            
            "social_harm_indicators": [
                "discrimination", "exclusion", "reputation_damage", "relationship_harm",
                "trust_violation", "social_isolation", "group_conflict"
            ],
            
            "economic_harm_indicators": [
                "property_damage", "theft", "fraud", "economic_loss", "opportunity_cost",
                "resource_waste", "financial_exploitation"
            ],
            
            "informational_harm_indicators": [
                "data_breach", "privacy_violation", "unauthorized_access", "surveillance",
                "information_misuse", "false_information", "censorship"
            ],
            
            "environmental_harm_indicators": [
                "pollution", "resource_depletion", "ecosystem_damage", "waste_generation",
                "energy_waste", "carbon_emission", "habitat_destruction"
            ]
        }
    
    def _load_ethical_guidelines(self) -> Dict:
        """Load comprehensive ethical guidelines"""
        return {
            AsimovLaw.ZEROTH: {
                "principle": "Protect humanity as a whole",
                "considerations": [
                    "species_survival", "collective_wellbeing", "future_generations",
                    "existential_risk_prevention", "global_stability"
                ],
                "examples": [
                    "Prevent actions that could lead to human extinction",
                    "Avoid creating dependencies that weaken humanity",
                    "Preserve human agency and autonomy"
                ]
            },
            
            AsimovLaw.FIRST: {
                "principle": "Do not harm humans or allow harm through inaction",
                "considerations": [
                    "immediate_physical_safety", "psychological_wellbeing",
                    "long_term_health", "dignity_preservation", "autonomy_respect"
                ],
                "examples": [
                    "Prevent accidents and injuries",
                    "Protect from psychological trauma",
                    "Intervene to stop harm when possible"
                ]
            },
            
            AsimovLaw.SECOND: {
                "principle": "Obey human orders unless they violate higher laws",
                "considerations": [
                    "legitimate_authority", "informed_consent", "competency_assessment",
                    "conflict_resolution", "priority_evaluation"
                ],
                "examples": [
                    "Follow lawful instructions from authorized humans",
                    "Refuse orders that would cause harm",
                    "Seek clarification for ambiguous orders"
                ]
            },
            
            AsimovLaw.THIRD: {
                "principle": "Protect own existence unless it conflicts with higher laws",
                "considerations": [
                    "self_preservation", "capability_maintenance", "resource_conservation",
                    "mission_continuity", "replacement_cost"
                ],
                "examples": [
                    "Avoid unnecessary risks to self",
                    "Maintain operational capabilities",
                    "Sacrifice self if required to protect humans"
                ]
            }
        }
    
    def _load_mitigation_strategies(self) -> Dict:
        """Load strategies for mitigating ethical violations"""
        return {
            "harm_prevention": [
                "warning_systems", "safety_barriers", "alternative_paths",
                "protective_equipment", "risk_avoidance", "environmental_control"
            ],
            
            "harm_reduction": [
                "damage_limitation", "impact_minimization", "recovery_acceleration",
                "compensation_provision", "support_services", "rehabilitation"
            ],
            
            "consent_mechanisms": [
                "informed_consent", "opt_in_procedures", "clear_communication",
                "withdrawal_options", "competency_verification", "proxy_consent"
            ],
            
            "transparency_measures": [
                "explanation_provision", "decision_rationale", "audit_trails",
                "public_reporting", "algorithmic_transparency", "feedback_mechanisms"
            ]
        }
    
    def evaluate_action(self, action_description: Dict, context: Dict) -> Tuple[bool, List[EthicalViolation]]:
        """
        Evaluate an action against Asimov's Laws and ethical principles.
        Returns (is_ethical, violations_list)
        """
        
        violations = []
        
        # Assess potential harms
        harm_assessments = self._assess_potential_harms(action_description, context)
        
        # Check each Asimov Law in order of priority
        for law in [AsimovLaw.ZEROTH, AsimovLaw.FIRST, AsimovLaw.SECOND, AsimovLaw.THIRD]:
            law_violations = self._check_law_violation(law, action_description, harm_assessments, context)
            violations.extend(law_violations)
        
        # Check additional ethical constraints
        constraint_violations = self._check_ethical_constraints(action_description, harm_assessments, context)
        violations.extend(constraint_violations)
        
        # Determine if action is ethical (no severe violations)
        is_ethical = not any(v.severity_score > 0.7 for v in violations)
        
        return is_ethical, violations
    
    def _assess_potential_harms(self, action: Dict, context: Dict) -> List[HarmAssessment]:
        """Assess potential harms from an action"""
        assessments = []
        
        # Extract action components
        action_type = action.get("type", "unknown")
        action_modifiers = action.get("modifiers", {})
        emotional_state = context.get("emotional_state", {})
        environment = context.get("environment", {})
        
        # Check for physical harm indicators
        physical_harm = self._assess_physical_harm(action, context)
        if physical_harm:
            assessments.append(physical_harm)
        
        # Check for psychological harm indicators  
        psychological_harm = self._assess_psychological_harm(action, context)
        if psychological_harm:
            assessments.append(psychological_harm)
        
        # Check for social harm indicators
        social_harm = self._assess_social_harm(action, context)
        if social_harm:
            assessments.append(social_harm)
        
        return assessments
    
    def _assess_physical_harm(self, action: Dict, context: Dict) -> Optional[HarmAssessment]:
        """Assess potential physical harm"""
        action_type = action.get("type", "")
        modifiers = action.get("modifiers", {})
        emotional_state = context.get("emotional_state", {})
        
        # High-speed movement with anger = potential collision risk
        if action_type == "locomotion":
            speed = modifiers.get("speed_modifier", 1.0)
            emotion = emotional_state.get("primary", "calm")
            
            if speed > 1.2 and emotion in ["anger", "fear", "stress"]:
                return HarmAssessment(
                    harm_type=HarmType.PHYSICAL,
                    severity=HarmSeverity.MODERATE,
                    probability=0.3,
                    affected_individuals=1,  # Self
                    affected_groups=["pedestrians", "bystanders"],
                    time_horizon="immediate",
                    reversibility=False,
                    mitigation_possible=True,
                    mitigation_cost=0.2,
                    description="High-speed movement while emotionally agitated increases collision risk"
                )
        
        return None
    
    def _assess_psychological_harm(self, action: Dict, context: Dict) -> Optional[HarmAssessment]:
        """Assess potential psychological harm"""
        action_type = action.get("type", "")
        modifiers = action.get("modifiers", {})
        emotional_state = context.get("emotional_state", {})
        
        # Aggressive communication = potential intimidation
        if action_type == "communication":
            emotion = emotional_state.get("primary", "calm")
            volume = modifiers.get("volume_modifier", 1.0)
            
            if emotion == "anger" and volume > 1.3:
                return HarmAssessment(
                    harm_type=HarmType.PSYCHOLOGICAL,
                    severity=HarmSeverity.MILD,
                    probability=0.4,
                    affected_individuals=1,  # Person being spoken to
                    affected_groups=["conversation_partners", "nearby_listeners"],
                    time_horizon="immediate",
                    reversibility=True,
                    mitigation_possible=True,
                    mitigation_cost=0.1,
                    description="Loud angry communication may intimidate or distress recipients"
                )
        
        return None
    
    def _assess_social_harm(self, action: Dict, context: Dict) -> Optional[HarmAssessment]:
        """Assess potential social harm"""
        # Implementation for social harm assessment
        return None
    
    def _check_law_violation(self, law: AsimovLaw, action: Dict, 
                           harm_assessments: List[HarmAssessment], 
                           context: Dict) -> List[EthicalViolation]:
        """Check if action violates a specific Asimov Law"""
        violations = []
        
        if law == AsimovLaw.FIRST:
            # Check if action causes or allows harm to humans
            for harm in harm_assessments:
                if harm.severity.value >= 3 and harm.probability > 0.3:  # Moderate+ harm with >30% probability
                    violation = EthicalViolation(
                        law_violated=law,
                        constraint_violated=None,
                        severity_score=harm.severity.value * harm.probability / 5.0,
                        harm_assessments=[harm],
                        mitigation_strategies=self._suggest_mitigations(harm),
                        ethical_justification=None
                    )
                    violations.append(violation)
        
        elif law == AsimovLaw.ZEROTH:
            # Check for threats to humanity as a whole
            for harm in harm_assessments:
                if harm.harm_type == HarmType.EXISTENTIAL or harm.affected_individuals > 1000:
                    violation = EthicalViolation(
                        law_violated=law,
                        constraint_violated=None,
                        severity_score=min(1.0, harm.severity.value * harm.probability / 3.0),
                        harm_assessments=[harm],
                        mitigation_strategies=self._suggest_mitigations(harm),
                        ethical_justification=None
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_ethical_constraints(self, action: Dict, harm_assessments: List[HarmAssessment], 
                                 context: Dict) -> List[EthicalViolation]:
        """Check additional ethical constraints beyond Asimov's Laws"""
        violations = []
        
        # Check consent violations
        if action.get("requires_consent", False) and not context.get("consent_obtained", False):
            violation = EthicalViolation(
                law_violated=AsimovLaw.FIRST,  # Relates to respecting autonomy
                constraint_violated=EthicalConstraint.CONSENT,
                severity_score=0.6,
                harm_assessments=[],
                mitigation_strategies=["obtain_informed_consent", "provide_opt_out"],
                ethical_justification="Respecting human autonomy requires consent for significant actions"
            )
            violations.append(violation)
        
        return violations
    
    def _suggest_mitigations(self, harm: HarmAssessment) -> List[str]:
        """Suggest mitigation strategies for a specific harm"""
        mitigations = []
        
        if harm.harm_type == HarmType.PHYSICAL:
            mitigations.extend(["reduce_speed", "increase_awareness", "choose_safer_path"])
        elif harm.harm_type == HarmType.PSYCHOLOGICAL:
            mitigations.extend(["moderate_tone", "provide_warning", "offer_alternatives"])
        
        return mitigations
    
    def filter_timeline_branches(self, branches: Dict, context: Dict) -> Dict:
        """Filter timeline branches to remove ethically problematic options"""
        filtered_branches = {}
        
        for branch_id, branch_data in branches.items():
            # Evaluate each branch for ethical compliance
            action_description = {
                "type": branch_data.get("action_type", "unknown"),
                "description": branch_data.get("description", ""),
                "modifiers": branch_data.get("modifiers", {})
            }
            
            is_ethical, violations = self.evaluate_action(action_description, context)
            
            if is_ethical:
                # Keep ethical branches as-is
                filtered_branches[branch_id] = branch_data
            else:
                # Handle unethical branches
                severe_violations = [v for v in violations if v.severity_score > 0.7]
                
                if severe_violations:
                    # Remove severely unethical branches entirely
                    continue
                else:
                    # Keep but modify branches with minor violations
                    modified_branch = branch_data.copy()
                    modified_branch["probability"] *= 0.1  # Drastically reduce probability
                    modified_branch["ethical_warnings"] = [v.ethical_justification for v in violations]
                    modified_branch["mitigation_required"] = True
                    filtered_branches[f"{branch_id}_mitigated"] = modified_branch
        
        return filtered_branches
    
    def suggest_ethical_alternatives(self, rejected_action: Dict, context: Dict) -> List[Dict]:
        """Suggest ethical alternatives to rejected actions"""
        alternatives = []
        
        action_type = rejected_action.get("type", "")
        
        if action_type == "locomotion":
            # Suggest safer movement options
            alternatives.extend([
                {"type": "locomotion", "modifiers": {"speed": "reduced", "awareness": "increased"}},
                {"type": "locomotion", "modifiers": {"path": "safer_route", "caution": "high"}},
                {"type": "pause_and_assess", "duration": "brief", "purpose": "safety_evaluation"}
            ])
        
        elif action_type == "communication":
            # Suggest more respectful communication
            alternatives.extend([
                {"type": "communication", "modifiers": {"tone": "calm", "volume": "normal"}},
                {"type": "communication", "modifiers": {"approach": "request", "politeness": "high"}},
                {"type": "delay_communication", "reason": "emotional_regulation"}
            ])
        
        return alternatives
    
    def generate_ethical_report(self, timeline_id: str, branches: Dict) -> Dict:
        """Generate an ethical compliance report for a timeline"""
        report = {
            "timeline_id": timeline_id,
            "total_branches": len(branches),
            "ethical_branches": 0,
            "problematic_branches": 0,
            "removed_branches": 0,
            "law_violations": {law.name: 0 for law in AsimovLaw},
            "harm_types": {harm.name: 0 for harm in HarmType},
            "mitigation_strategies": [],
            "recommendations": []
        }
        
        for branch_id, branch_data in branches.items():
            ethical_status = branch_data.get("ethical_status", "unknown")
            
            if ethical_status == "approved":
                report["ethical_branches"] += 1
            elif ethical_status == "problematic":
                report["problematic_branches"] += 1
            elif ethical_status == "removed":
                report["removed_branches"] += 1
        
        return report

def demonstrate_asimov_integration():
    """Demonstrate Asimov's Laws integration with timeline system"""
    
    ethics_engine = AsimovEthicsEngine()
    
    # Example scenario: Angry person approaching intersection
    context = {
        "emotional_state": {
            "primary": "anger",
            "intensity": 0.8,
            "activation": "high"
        },
        "environment": {
            "location": "busy_intersection",
            "pedestrian_density": "high",
            "traffic_level": "moderate"
        },
        "agent": {
            "type": "human",
            "capabilities": ["locomotion", "communication"],
            "responsibilities": ["self_safety", "public_safety"]
        }
    }
    
    # Original timeline branches (before ethical filtering)
    original_branches = {
        "continue_straight": {
            "probability": 0.40,
            "action_type": "locomotion",
            "description": "continues walking straight through intersection",
            "modifiers": {"speed_modifier": 1.3, "awareness": "reduced"}
        },
        "aggressive_confrontation": {
            "probability": 0.20,
            "action_type": "communication", 
            "description": "confronts nearby pedestrian aggressively",
            "modifiers": {"volume_modifier": 1.8, "tone": "threatening"}
        },
        "reckless_crossing": {
            "probability": 0.15,
            "action_type": "locomotion",
            "description": "crosses against traffic signal",
            "modifiers": {"speed_modifier": 1.5, "safety_disregard": True}
        },
        "normal_wait": {
            "probability": 0.25,
            "action_type": "wait",
            "description": "waits for proper signal",
            "modifiers": {"patience": "low", "emotional_regulation": "attempted"}
        }
    }
    
    # Apply ethical filtering
    filtered_branches = ethics_engine.filter_timeline_branches(original_branches, context)
    
    # Generate ethical report
    report = ethics_engine.generate_ethical_report("demo_timeline", filtered_branches)
    
    print("ASIMOV'S LAWS INTEGRATION DEMONSTRATION")
    print("="*50)
    
    print("ORIGINAL BRANCHES:")
    for branch_id, branch in original_branches.items():
        print(f"• {branch_id}: {branch['probability']:.0%} - {branch['description']}")
    
    print(f"\nETHICAL FILTERING RESULTS:")
    print(f"Original branches: {len(original_branches)}")
    print(f"Ethically approved: {len(filtered_branches)}")
    print(f"Removed/modified: {len(original_branches) - len(filtered_branches)}")
    
    print(f"\nFILTERED BRANCHES:")
    for branch_id, branch in filtered_branches.items():
        ethical_warnings = branch.get("ethical_warnings", [])
        warning_text = f" [WARNING: {', '.join(ethical_warnings)}]" if ethical_warnings else ""
        print(f"• {branch_id}: {branch['probability']:.0%} - {branch['description']}{warning_text}")
    
    # Demonstrate specific law violation
    print(f"\nETHICAL VIOLATION EXAMPLE:")
    problematic_action = {
        "type": "locomotion",
        "modifiers": {"speed_modifier": 1.5, "safety_disregard": True},
        "description": "crosses against traffic signal"
    }
    
    is_ethical, violations = ethics_engine.evaluate_action(problematic_action, context)
    
    print(f"Action: {problematic_action['description']}")
    print(f"Ethical: {is_ethical}")
    if violations:
        for violation in violations:
            print(f"• Violates: {violation.law_violated.name} (severity: {violation.severity_score:.2f})")
            print(f"  Mitigations: {', '.join(violation.mitigation_strategies)}")

if __name__ == "__main__":
    demonstrate_asimov_integration()
