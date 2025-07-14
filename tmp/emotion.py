#!/usr/bin/env python3
"""
Emotional State Semantic Integration System

Integrates emotional/mood states into semantic representations for enhanced
behavioral understanding and prediction. Think of it as adding the "why"
behind actions, not just the "what".
"""

import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

class EmotionalValence(Enum):
    """Core emotional valence categories"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    MIXED = "mixed"

class ActivationLevel(Enum):
    """Emotional activation/arousal levels"""
    HIGH = "high"        # Intense, energetic emotions
    MEDIUM = "medium"    # Moderate intensity
    LOW = "low"         # Calm, subdued emotions
    DORMANT = "dormant" # Minimal emotional activation

@dataclass
class EmotionalState:
    """Represents an emotional state with multiple dimensions"""
    primary_emotion: str
    intensity: float  # 0.0 to 1.0
    valence: EmotionalValence
    activation: ActivationLevel
    secondary_emotions: List[str] = None
    duration_estimate: Optional[float] = None  # In minutes
    triggers: List[str] = None
    behavioral_tendencies: List[str] = None
    
    def __post_init__(self):
        if self.secondary_emotions is None:
            self.secondary_emotions = []
        if self.triggers is None:
            self.triggers = []
        if self.behavioral_tendencies is None:
            self.behavioral_tendencies = []

class EmotionalSemanticProcessor:
    """
    Integrates emotional states into semantic representations.
    
    Like having an emotional intelligence layer that understands not just
    WHAT someone is doing, but HOW they're feeling while doing it.
    """
    
    def __init__(self):
        self.emotion_taxonomy = self._build_emotion_taxonomy()
        self.behavioral_patterns = self._load_behavioral_patterns()
        self.contextual_modifiers = self._load_contextual_modifiers()
    
    def _build_emotion_taxonomy(self) -> Dict:
        """Build comprehensive emotion taxonomy with semantic properties"""
        return {
            # PRIMARY EMOTIONS (Plutchik's Wheel + extensions)
            "joy": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.MEDIUM,
                "variations": ["happiness", "delight", "bliss", "contentment", "elation"],
                "intensity_scale": ["contentment", "joy", "ecstasy"],
                "behavioral_tendencies": ["approach", "social_engagement", "exploration", "optimistic_planning"],
                "physiological": ["increased_energy", "relaxed_muscles", "smile"],
                "cognitive_effects": ["positive_bias", "creative_thinking", "optimistic_predictions"]
            },
            
            "anger": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.HIGH,
                "variations": ["irritation", "fury", "rage", "annoyance", "frustration"],
                "intensity_scale": ["annoyance", "anger", "rage"],
                "behavioral_tendencies": ["confrontation", "aggressive_action", "boundary_setting", "problem_solving"],
                "physiological": ["increased_heart_rate", "muscle_tension", "frown"],
                "cognitive_effects": ["narrow_focus", "action_oriented", "justice_seeking"]
            },
            
            "sadness": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.LOW,
                "variations": ["melancholy", "grief", "sorrow", "despair", "disappointment"],
                "intensity_scale": ["disappointment", "sadness", "despair"],
                "behavioral_tendencies": ["withdrawal", "seeking_comfort", "reflection", "reduced_activity"],
                "physiological": ["low_energy", "tears", "slumped_posture"],
                "cognitive_effects": ["introspection", "realistic_assessment", "memory_focus"]
            },
            
            "fear": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.HIGH,
                "variations": ["anxiety", "worry", "terror", "nervousness", "apprehension"],
                "intensity_scale": ["concern", "fear", "terror"],
                "behavioral_tendencies": ["avoidance", "caution", "seeking_safety", "hypervigilance"],
                "physiological": ["increased_alertness", "muscle_readiness", "rapid_breathing"],
                "cognitive_effects": ["threat_focus", "risk_assessment", "safety_planning"]
            },
            
            "surprise": {
                "valence": EmotionalValence.NEUTRAL,
                "activation": ActivationLevel.HIGH,
                "variations": ["astonishment", "amazement", "shock", "wonder"],
                "intensity_scale": ["mild_surprise", "surprise", "shock"],
                "behavioral_tendencies": ["attention_focusing", "information_seeking", "pause_action"],
                "physiological": ["raised_eyebrows", "open_mouth", "increased_alertness"],
                "cognitive_effects": ["cognitive_reset", "learning_readiness", "memory_enhancement"]
            },
            
            "disgust": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.MEDIUM,
                "variations": ["revulsion", "contempt", "aversion", "distaste"],
                "intensity_scale": ["distaste", "disgust", "revulsion"],
                "behavioral_tendencies": ["avoidance", "rejection", "distancing", "purification"],
                "physiological": ["nausea", "facial_contortion", "recoil"],
                "cognitive_effects": ["moral_judgment", "boundary_setting", "purification_seeking"]
            },
            
            "trust": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.LOW,
                "variations": ["confidence", "faith", "reliance", "acceptance"],
                "intensity_scale": ["acceptance", "trust", "admiration"],
                "behavioral_tendencies": ["cooperation", "openness", "vulnerability", "collaboration"],
                "physiological": ["relaxed_state", "open_posture", "steady_breathing"],
                "cognitive_effects": ["positive_expectations", "reduced_vigilance", "cooperation_bias"]
            },
            
            "anticipation": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.MEDIUM,
                "variations": ["expectation", "hope", "eagerness", "excitement"],
                "intensity_scale": ["interest", "anticipation", "vigilance"],
                "behavioral_tendencies": ["forward_planning", "preparation", "goal_pursuit"],
                "physiological": ["increased_energy", "forward_lean", "alert_posture"],
                "cognitive_effects": ["future_focus", "planning_enhancement", "motivation_boost"]
            },
            
            # COMPLEX EMOTIONS (combinations)
            "love": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.MEDIUM,
                "primary_components": ["joy", "trust"],
                "variations": ["affection", "adoration", "passion", "attachment"],
                "behavioral_tendencies": ["bonding", "care_giving", "protection", "intimacy_seeking"],
                "cognitive_effects": ["positive_bias_toward_object", "protective_instincts", "empathy_enhancement"]
            },
            
            "guilt": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.MEDIUM,
                "primary_components": ["fear", "disgust"],
                "behavioral_tendencies": ["apology", "repair_action", "self_punishment", "confession"],
                "cognitive_effects": ["moral_self_evaluation", "behavior_correction", "empathy_activation"]
            },
            
            "pride": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.MEDIUM,
                "primary_components": ["joy", "anger"],
                "behavioral_tendencies": ["display", "confidence", "leadership", "achievement_seeking"],
                "cognitive_effects": ["self_efficacy", "status_awareness", "goal_reinforcement"]
            },
            
            # MOOD STATES (longer duration, less specific triggers)
            "optimistic": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.MEDIUM,
                "duration_typical": "hours_to_days",
                "behavioral_tendencies": ["risk_taking", "social_engagement", "goal_pursuit"],
                "cognitive_effects": ["positive_expectation_bias", "solution_focus"]
            },
            
            "pessimistic": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.LOW,
                "duration_typical": "hours_to_days",
                "behavioral_tendencies": ["risk_avoidance", "withdrawal", "defensive_planning"],
                "cognitive_effects": ["negative_expectation_bias", "problem_focus"]
            },
            
            "stressed": {
                "valence": EmotionalValence.NEGATIVE,
                "activation": ActivationLevel.HIGH,
                "duration_typical": "minutes_to_hours",
                "behavioral_tendencies": ["urgency", "multitasking", "shortened_attention"],
                "cognitive_effects": ["narrow_focus", "reduced_creativity", "error_prone"]
            },
            
            "calm": {
                "valence": EmotionalValence.POSITIVE,
                "activation": ActivationLevel.LOW,
                "duration_typical": "minutes_to_hours",
                "behavioral_tendencies": ["measured_responses", "patience", "reflection"],
                "cognitive_effects": ["broad_perspective", "clear_thinking", "patience"]
            }
        }
    
    def _load_behavioral_patterns(self) -> Dict:
        """Load emotion-behavior correlation patterns"""
        return {
            "walking_patterns": {
                "joy": {"speed": "bouncy", "posture": "upright", "path": "direct_or_exploratory"},
                "anger": {"speed": "fast_determined", "posture": "tense", "path": "direct_forceful"},
                "sadness": {"speed": "slow", "posture": "slumped", "path": "meandering"},
                "fear": {"speed": "quick_nervous", "posture": "alert", "path": "cautious"},
                "calm": {"speed": "steady", "posture": "relaxed", "path": "purposeful"}
            },
            
            "decision_making": {
                "optimistic": {"risk_tolerance": "high", "time_horizon": "long", "options_considered": "many"},
                "pessimistic": {"risk_tolerance": "low", "time_horizon": "short", "options_considered": "few"},
                "stressed": {"risk_tolerance": "variable", "time_horizon": "immediate", "options_considered": "limited"},
                "calm": {"risk_tolerance": "balanced", "time_horizon": "appropriate", "options_considered": "thorough"}
            },
            
            "social_interactions": {
                "joy": {"approach": "engaging", "communication": "open", "cooperation": "high"},
                "anger": {"approach": "confrontational", "communication": "direct", "cooperation": "conditional"},
                "sadness": {"approach": "withdrawn", "communication": "minimal", "cooperation": "passive"},
                "trust": {"approach": "open", "communication": "honest", "cooperation": "high"}
            }
        }
    
    def _load_contextual_modifiers(self) -> Dict:
        """Load contextual factors that influence emotional expression"""
        return {
            "social_context": {
                "public": {"emotional_suppression": 0.7, "socially_acceptable_only": True},
                "private": {"emotional_suppression": 0.2, "full_expression": True},
                "professional": {"emotional_suppression": 0.8, "controlled_expression": True},
                "intimate": {"emotional_suppression": 0.1, "authentic_expression": True}
            },
            
            "cultural_factors": {
                "individualistic": {"emotion_expression": "direct", "intensity_acceptable": "high"},
                "collectivistic": {"emotion_expression": "moderated", "intensity_acceptable": "medium"}
            },
            
            "temporal_factors": {
                "morning": {"energy_modifier": 1.2, "optimism_bias": 1.1},
                "afternoon": {"energy_modifier": 1.0, "optimism_bias": 1.0},
                "evening": {"energy_modifier": 0.8, "optimism_bias": 0.9},
                "night": {"energy_modifier": 0.6, "emotional_volatility": 1.2}
            }
        }
    
    def detect_emotional_state(self, semantic_representation: Dict, 
                             behavioral_cues: List[str] = None,
                             context: Dict = None) -> EmotionalState:
        """Detect emotional state from semantic representation and cues"""
        
        if behavioral_cues is None:
            behavioral_cues = []
        if context is None:
            context = {}
        
        # Extract emotional indicators from semantic representation
        emotional_indicators = self._extract_emotional_indicators(semantic_representation)
        
        # Analyze behavioral cues
        behavioral_emotions = self._analyze_behavioral_cues(behavioral_cues)
        
        # Combine and weight evidence
        primary_emotion, confidence = self._determine_primary_emotion(
            emotional_indicators, behavioral_emotions, context
        )
        
        # Get emotion properties
        emotion_data = self.emotion_taxonomy.get(primary_emotion, {})
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=confidence,
            valence=emotion_data.get("valence", EmotionalValence.NEUTRAL),
            activation=emotion_data.get("activation", ActivationLevel.MEDIUM),
            behavioral_tendencies=emotion_data.get("behavioral_tendencies", []),
            triggers=emotional_indicators.get("triggers", [])
        )
    
    def _extract_emotional_indicators(self, semantic_rep: Dict) -> Dict:
        """Extract emotional indicators from semantic representation"""
        indicators = {
            "triggers": [],
            "behavioral_cues": [],
            "contextual_factors": []
        }
        
        # Check actions for emotional content
        actions = semantic_rep.get("actions", [])
        for action in actions:
            if isinstance(action, str):
                # Look for emotional action patterns
                if "quickly" in action:
                    indicators["behavioral_cues"].append("urgency")
                elif "slowly" in action:
                    indicators["behavioral_cues"].append("lethargy")
                elif "talking" in action and "phone" in action:
                    indicators["contextual_factors"].append("social_connection")
        
        # Check entities for emotional triggers
        entities = semantic_rep.get("entities", [])
        for entity in entities:
            if isinstance(entity, dict):
                entity_type = entity.get("type", "")
                if entity_type == "device" and "phone" in entity.get("concept", ""):
                    indicators["triggers"].append("communication_opportunity")
        
        return indicators
    
    def _analyze_behavioral_cues(self, cues: List[str]) -> Dict[str, float]:
        """Analyze behavioral cues for emotional content"""
        emotion_scores = {}
        
        for cue in cues:
            cue_lower = cue.lower()
            
            # Walking speed/style indicators
            if "fast" in cue_lower or "quick" in cue_lower:
                emotion_scores["urgency"] = emotion_scores.get("urgency", 0) + 0.3
                emotion_scores["stress"] = emotion_scores.get("stress", 0) + 0.2
            elif "slow" in cue_lower:
                emotion_scores["sadness"] = emotion_scores.get("sadness", 0) + 0.3
                emotion_scores["contemplation"] = emotion_scores.get("contemplation", 0) + 0.2
            
            # Posture indicators
            if "upright" in cue_lower:
                emotion_scores["confidence"] = emotion_scores.get("confidence", 0) + 0.3
            elif "slumped" in cue_lower:
                emotion_scores["sadness"] = emotion_scores.get("sadness", 0) + 0.4
            
            # Social indicators
            if "talking" in cue_lower:
                emotion_scores["social_engagement"] = emotion_scores.get("social_engagement", 0) + 0.3
        
        return emotion_scores
    
    def _determine_primary_emotion(self, indicators: Dict, behavioral: Dict, context: Dict) -> Tuple[str, float]:
        """Determine primary emotion and confidence level"""
        
        # Simple scoring system (in production, use ML model)
        emotion_scores = {}
        
        # Score based on behavioral evidence
        for emotion, score in behavioral.items():
            emotion_scores[emotion] = score
        
        # Add context modifiers
        if context.get("time_of_day") == "morning":
            emotion_scores["optimistic"] = emotion_scores.get("optimistic", 0) + 0.2
        
        # Default to neutral/calm if no strong indicators
        if not emotion_scores:
            return "calm", 0.5
        
        # Find highest scoring emotion
        primary = max(emotion_scores.items(), key=lambda x: x[1])
        return primary[0], min(primary[1], 1.0)
    
    def integrate_emotion_into_semantic(self, semantic_rep: Dict, 
                                      emotional_state: EmotionalState,
                                      include_predictions: bool = True) -> Dict:
        """Integrate emotional state into semantic representation"""
        
        # Create enhanced semantic representation
        enhanced = semantic_rep.copy()
        
        # Add emotional layer
        enhanced["emotional_state"] = {
            "primary": emotional_state.primary_emotion,
            "intensity": emotional_state.intensity,
            "valence": emotional_state.valence.value,
            "activation": emotional_state.activation.value
        }
        
        # Add behavioral predictions
        if include_predictions:
            enhanced["behavioral_predictions"] = self._predict_behaviors(emotional_state)
        
        # Modify existing semantic elements based on emotion
        enhanced = self._apply_emotional_modifiers(enhanced, emotional_state)
        
        return enhanced
    
    def _predict_behaviors(self, emotional_state: EmotionalState) -> Dict:
        """Predict likely behaviors based on emotional state"""
        
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        predictions = {
            "decision_making": {},
            "social_behavior": {},
            "movement_patterns": {},
            "risk_tolerance": 0.5  # Default neutral
        }
        
        # Get behavioral patterns for this emotion
        if emotion in self.behavioral_patterns.get("decision_making", {}):
            patterns = self.behavioral_patterns["decision_making"][emotion]
            predictions["decision_making"] = patterns
            
            # Adjust risk tolerance based on emotion and intensity
            if emotion in ["joy", "optimistic"]:
                predictions["risk_tolerance"] = 0.7 + (intensity * 0.2)
            elif emotion in ["fear", "anxiety", "pessimistic"]:
                predictions["risk_tolerance"] = 0.3 - (intensity * 0.2)
        
        return predictions
    
    def _apply_emotional_modifiers(self, semantic_rep: Dict, emotional_state: EmotionalState) -> Dict:
        """Apply emotional modifiers to semantic representation"""
        
        enhanced = semantic_rep.copy()
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Modify actions based on emotion
        if "actions" in enhanced:
            for i, action in enumerate(enhanced["actions"]):
                if isinstance(action, dict) and "type" in action:
                    # Add emotional coloring to actions
                    action_type = action["type"]
                    
                    if action_type == "locomotion":
                        # Modify walking based on emotion
                        if emotion == "joy":
                            action["emotional_modifier"] = "energetic"
                            action["speed_modifier"] = 1.2 * intensity
                        elif emotion == "sadness":
                            action["emotional_modifier"] = "lethargic"
                            action["speed_modifier"] = 0.8 - (0.3 * intensity)
                        elif emotion == "anger":
                            action["emotional_modifier"] = "determined"
                            action["speed_modifier"] = 1.1 + (0.2 * intensity)
                        elif emotion == "fear":
                            action["emotional_modifier"] = "cautious"
                            action["path_modifier"] = "vigilant"
                    
                    elif action_type == "communication":
                        # Modify communication based on emotion
                        if emotion == "joy":
                            action["tone"] = "enthusiastic"
                            action["verbosity"] = "high"
                        elif emotion == "sadness":
                            action["tone"] = "subdued"
                            action["verbosity"] = "low"
                        elif emotion == "anger":
                            action["tone"] = "intense"
                            action["directness"] = "high"
        
        return enhanced

def demonstrate_emotional_integration():
    """Demonstrate emotional integration with semantic processing"""
    
    processor = EmotionalSemanticProcessor()
    
    # Original semantic representation
    original_semantic = {
        "entities": [
            {"type": "agent", "concept": "woman", "attributes": {"height": "tall"}},
            {"type": "location", "concept": "street", "attributes": {"activity": "busy"}},
            {"type": "device", "concept": "phone"}
        ],
        "actions": [
            {"type": "locomotion", "concept": "walk", "modifiers": {"speed": "quickly"}},
            {"type": "communication", "concept": "talk", "medium": "phone"}
        ],
        "spatial_relations": [{"type": "path", "direction": "down"}]
    }
    
    # Detect emotional state from behavioral cues
    behavioral_cues = ["walking_quickly", "talking_on_phone", "upright_posture", "determined_pace"]
    context = {"time_of_day": "morning", "social_context": "public"}
    
    emotional_state = processor.detect_emotional_state(
        original_semantic, 
        behavioral_cues, 
        context
    )
    
    # Integrate emotion into semantic representation
    enhanced_semantic = processor.integrate_emotion_into_semantic(
        original_semantic, 
        emotional_state
    )
    
    print("ORIGINAL SEMANTIC REPRESENTATION:")
    print(json.dumps(original_semantic, indent=2))
    print("\n" + "="*50 + "\n")
    
    print("DETECTED EMOTIONAL STATE:")
    print(f"Primary Emotion: {emotional_state.primary_emotion}")
    print(f"Intensity: {emotional_state.intensity:.2f}")
    print(f"Valence: {emotional_state.valence.value}")
    print(f"Activation: {emotional_state.activation.value}")
    print("\n" + "="*50 + "\n")
    
    print("EMOTIONALLY ENHANCED SEMANTIC REPRESENTATION:")
    print(json.dumps(enhanced_semantic, indent=2))
    
    print("\n" + "="*50 + "\n")
    print("BEHAVIORAL PREDICTIONS:")
    predictions = enhanced_semantic.get("behavioral_predictions", {})
    print(f"Risk Tolerance: {predictions.get('risk_tolerance', 'unknown')}")
    print(f"Decision Making: {predictions.get('decision_making', {})}")

if __name__ == "__main__":
    demonstrate_emotional_integration()
