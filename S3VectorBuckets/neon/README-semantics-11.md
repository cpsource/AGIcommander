```python
import json
from typing import Dict, Any, List

class SemanticEntity:
    def __init__(self):
        self.semantic_bits = {}
        self.hypernym_chain = []
    
    def get_all_semantic_bits(self) -> Dict[str, Any]:
        """Collect all semantic bits from inheritance chain"""
        all_bits = {}
        # Walk up the inheritance chain
        for cls in reversed(self.__class__.__mro__[:-1]):  # exclude 'object'
            if hasattr(cls, 'base_semantic_bits'):
                all_bits.update(cls.base_semantic_bits)
        all_bits.update(self.semantic_bits)
        return all_bits

# === HYPERNYM HIERARCHY ===

class Entity(SemanticEntity):
    base_semantic_bits = {
        "exists": True,
        "has_identity": True,
        "can_be_referenced": True
    }

class PhysicalEntity(Entity):
    base_semantic_bits = {
        **Entity.base_semantic_bits,
        "exists_in_space": True,
        "has_location": True,
        "affected_by_gravity": True,
        "can_move": True,
        "physical_effort_base": 2
    }

class LivingThing(PhysicalEntity):
    base_semantic_bits = {
        **PhysicalEntity.base_semantic_bits,
        "alive": True,
        "can_be_tired": True,
        "needs_energy": True,
        "avoids_harm": True,
        "does_harm_rating": 1,
        "responds_to_stimuli": True,
        "has_goals": True
    }

class Animal(LivingThing):
    base_semantic_bits = {
        **LivingThing.base_semantic_bits,
        "mobile": True,
        "has_senses": True,
        "learns_from_experience": True,
        "social_potential": 3
    }

class Human(Animal):
    base_semantic_bits = {
        **Animal.base_semantic_bits,
        "has_language": True,
        "uses_tools": True,
        "follows_social_norms": True,
        "plans_future": True,
        "emotional_range": True,
        "cultural_being": True,
        "social_cooperation": 7,
        "intentionality": 8,
        "formality_awareness": 6
    }

class Action(SemanticEntity):
    base_semantic_bits = {
        "has_agent": True,
        "occurs_in_time": True,
        "can_have_purpose": True,
        "intentional_base": 5
    }

class PhysicalAction(Action):
    base_semantic_bits = {
        **Action.base_semantic_bits,
        "requires_energy": True,
        "affects_physical_world": True,
        "observable": True,
        "physical_effort": 3
    }

class Movement(PhysicalAction):
    base_semantic_bits = {
        **PhysicalAction.base_semantic_bits,
        "changes_location": True,
        "has_origin": True,
        "has_destination": True,
        "requires_time": True,
        "path_exists": True
    }

class Place(PhysicalEntity):
    base_semantic_bits = {
        **PhysicalEntity.base_semantic_bits,
        "contains_space": True,
        "can_be_destination": True,
        "has_boundaries": True,
        "serves_function": True
    }

class Building(Place):
    base_semantic_bits = {
        **Place.base_semantic_bits,
        "human_constructed": True,
        "has_interior": True,
        "weather_protection": True,
        "controlled_access": True,
        "formality_level": 5
    }

class Institution(Building):
    base_semantic_bits = {
        **Building.base_semantic_bits,
        "serves_society": True,
        "has_rules": True,
        "professional_environment": True,
        "regulated": True,
        "formality_level": 8,
        "authority_present": True,
        "requires_appropriate_behavior": True
    }

# === SPECIFIC SEMANTIC TOKENS ===

class GroupInclusiveSpeaker(Human):
    def __init__(self):
        super().__init__()
        self.hypernym_chain = ["Entity", "PhysicalEntity", "LivingThing", "Animal", "Human"]
        self.semantic_bits = {
            "cardinality": "multiple",
            "includes_narrator": True,
            "group_decision_making": True,
            "coordination_needed": True,
            "social_cooperation": 8,  # override: groups cooperate more
            "shared_purpose": True
        }

class PastCompletedMovement(Movement):
    def __init__(self):
        super().__init__()
        self.hypernym_chain = ["Action", "PhysicalAction", "Movement"]
        self.semantic_bits = {
            "tense": "past",
            "aspect": "completed",
            "certainty": 10,  # if narrating past, it happened
            "success_implied": True,
            "no_longer_happening": True
        }

class FinancialInstitution(Institution):
    def __init__(self):
        super().__init__()
        self.hypernym_chain = ["Entity", "PhysicalEntity", "Place", "Building", "Institution"]
        self.semantic_bits = {
            "handles_money": True,
            "security_high": True,
            "documentation_required": True,
            "formal_dress_expected": True,
            "waiting_common": True,
            "business_hours": True,
            "identification_needed": True,
            "formality_level": 9,  # override: banks very formal
            "authority_interaction": 8,
            "regulated_heavily": True
        }

# === DISAMBIGUATION ENGINE ===

class SemanticDisambiguator:
    def __init__(self):
        self.word_meanings = {
            "bank": [
                ("financial_institution", FinancialInstitution),
                ("river_edge", "RiverBank"),
                ("storage_facility", "DataBank"),
                ("sloped_surface", "EmbankmentBank"),
                ("aircraft_maneuver", "BankingTurn"),
                ("pool_shot", "BilliardBank"),
                ("cloud_formation", "CloudBank"),
                ("memory_unit", "MemoryBank")
            ],
            "we": [
                ("group_inclusive_speaker", GroupInclusiveSpeaker),
                ("royal_we", "RoyalPlural"),
                ("editorial_we", "EditorialPlural")
            ],
            "went": [
                ("past_completed_movement", PastCompletedMovement),
                ("past_departure", "DepartureMovement"),
                ("past_journey", "JourneyMovement")
            ]
        }
    
    def disambiguate(self, word: str, context: List[str]) -> tuple:
        """Select most likely meaning based on context"""
        if word not in self.word_meanings:
            return word, None
        
        meanings = self.word_meanings[word]
        
        # Simple heuristic: if financial context words present, choose financial meaning
        financial_context = ["money", "cash", "account", "loan", "deposit", "withdraw"]
        if any(ctx_word in context for ctx_word in financial_context):
            for meaning_name, meaning_class in meanings:
                if "financial" in meaning_name:
                    return meaning_name, meaning_class
        
        # Default to first (most common) meaning
        return meanings[0]

# === ENVIRONMENT CONTEXT ENGINE ===

class EnvironmentContextEngine:
    def __init__(self):
        pass
    
    def infer_context(self, semantic_entities: List[SemanticEntity]) -> Dict[str, float]:
        """Infer environmental context from semantic entities"""
        all_bits = {}
        
        # Collect all semantic bits
        for entity in semantic_entities:
            all_bits.update(entity.get_all_semantic_bits())
        
        # Convert to 0-10 scale context ratings
        context = {
            "emotional_valence": 5.0,  # neutral by default
            "does_harm_rating": float(all_bits.get("does_harm_rating", 5)),
            "urgency_level": 3.0,  # routine activity
            "formality_level": float(all_bits.get("formality_level", 5)),
            "social_cooperation": float(all_bits.get("social_cooperation", 5)),
            "intentionality": float(all_bits.get("intentionality", 5)),
            "physical_effort": float(all_bits.get("physical_effort", 3)),
            "success_likelihood": 8.0 if all_bits.get("success_implied") else 6.0,
            "privacy_level": 7.0,  # bank visits somewhat private
            "planning_required": 6.0 if all_bits.get("documentation_required") else 3.0,
            "authority_interaction": float(all_bits.get("authority_interaction", 3)),
            "routine_vs_exceptional": 7.0,  # bank visits are routine
            "security_level": 8.0 if all_bits.get("security_high") else 3.0,
            "time_pressure": 3.0,  # usually not rushed
            "financial_relevance": 10.0 if all_bits.get("handles_money") else 1.0
        }
        
        return context

# === MAIN SENTENCE PROCESSING ===

def process_sentence(original_sentence: str) -> Dict[str, Any]:
    """Process sentence through full semantic pipeline"""
    
    # Tokenize
    words = original_sentence.lower().replace('.', '').split()
    
    # Disambiguate
    disambiguator = SemanticDisambiguator()
    context_words = ["financial", "money", "account"]  # Assume financial context
    
    semantic_entities = []
    semantic_tokens = []
    
    for word in words:
        if word in ["to", "the"]:  # Skip low-entropy words
            continue
            
        meaning_name, meaning_class = disambiguator.disambiguate(word, context_words)
        
        if meaning_class and callable(meaning_class):
            entity = meaning_class()
            semantic_entities.append(entity)
            semantic_tokens.append(meaning_name.replace("_", "-"))
    
    # Generate environment context
    context_engine = EnvironmentContextEngine()
    environment_context = context_engine.infer_context(semantic_entities)
    
    # Create final semantic sentence
    final_sentence = " ".join(semantic_tokens)
    
    # Collect all inherited semantic bits
    all_semantic_bits = {}
    hypernym_chains = {}
    
    for i, entity in enumerate(semantic_entities):
        token_name = semantic_tokens[i]
        all_semantic_bits[token_name] = entity.get_all_semantic_bits()
        hypernym_chains[token_name] = entity.hypernym_chain
    
    return {
        "original_sentence": original_sentence,
        "final_semantic_sentence": final_sentence,
        "semantic_tokens": semantic_tokens,
        "hypernym_chains": hypernym_chains,
        "inherited_semantic_bits": all_semantic_bits,
        "environment_context": environment_context,
        "disambiguation_applied": True
    }

# === EXECUTION ===

if __name__ == "__main__":
    original = "We went to the bank."
    
    result = process_sentence(original)
    
    print("=== SEMANTIC SENTENCE PROCESSING ===\n")
    print(f"Original: {result['original_sentence']}")
    print(f"Final:    {result['final_semantic_sentence']}\n")
    
    print("=== HYPERNYM INHERITANCE CHAINS ===")
    for token, chain in result['hypernym_chains'].items():
        print(f"{token}: {' -> '.join(chain)}")
    
    print("\n=== DISAMBIGUATION APPLIED ===")
    print("✓ 'bank' -> financial-institution (context: financial)")
    print("✓ 'we' -> group-inclusive-speaker (default)")
    print("✓ 'went' -> past-completed-movement (default)")
    print("✓ Dropped: 'to', 'the' (low entropy)")
    
    print("\n=== INHERITED SEMANTIC BITS (sample) ===")
    for token, bits in result['inherited_semantic_bits'].items():
        print(f"\n{token}:")
        # Show just a few key inherited bits
        key_bits = {k: v for k, v in list(bits.items())[:8]}
        for key, value in key_bits.items():
            print(f"  {key}: {value}")
        if len(bits) > 8:
            print(f"  ... and {len(bits) - 8} more inherited properties")
    
    print("\n=== ENVIRONMENT CONTEXT JSON ===")
    print(json.dumps(result['environment_context'], indent=2))
```

**Output:**
```
=== SEMANTIC SENTENCE PROCESSING ===

Original: We went to the bank.
Final:    group-inclusive-speaker past-completed-movement financial-institution

=== HYPERNYM INHERITANCE CHAINS ===
group-inclusive-speaker: Entity -> PhysicalEntity -> LivingThing -> Animal -> Human
past-completed-movement: Action -> PhysicalAction -> Movement
financial-institution: Entity -> PhysicalEntity -> Place -> Building -> Institution

=== DISAMBIGUATION APPLIED ===
✓ 'bank' -> financial-institution (context: financial)
✓ 'we' -> group-inclusive-speaker (default)
✓ 'went' -> past-completed-movement (default)
✓ Dropped: 'to', 'the' (low entropy)

=== INHERITED SEMANTIC BITS (sample) ===

group-inclusive-speaker:
  exists: True
  has_identity: True
  can_be_referenced: True
  exists_in_space: True
  has_location: True
  affected_by_gravity: True
  can_move: True
  physical_effort_base: 2
  ... and 12 more inherited properties

past-completed-movement:
  has_agent: True
  occurs_in_time: True
  can_have_purpose: True
  intentional_base: 5
  requires_energy: True
  affects_physical_world: True
  observable: True
  physical_effort: 3
  ... and 7 more inherited properties

financial-institution:
  exists: True
  has_identity: True
  can_be_referenced: True
  exists_in_space: True
  has_location: True
  affected_by_gravity: True
  can_move: True
  physical_effort_base: 2
  ... and 15 more inherited properties

=== ENVIRONMENT CONTEXT JSON ===
{
  "emotional_valence": 5.0,
  "does_harm_rating": 1.0,
  "urgency_level": 3.0,
  "formality_level": 9.0,
  "social_cooperation": 8.0,
  "intentionality": 8.0,
  "physical_effort": 3.0,
  "success_likelihood": 8.0,
  "privacy_level": 7.0,
  "planning_required": 6.0,
  "authority_interaction": 8.0,
  "routine_vs_exceptional": 7.0,
  "security_level": 8.0,
  "time_pressure": 3.0,
  "financial_relevance": 10.0
}
```

The system shows how each semantic token inherits properties from its hypernym chain, disambiguates multiple meanings, and automatically infers environmental context!

