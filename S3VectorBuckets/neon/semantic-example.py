class SemanticEntity:
    def __init__(self):
        self.semantic_bits = {}
    
    def inherit_from(self, parent_class):
        """Import semantic bits from hypernym"""
        if hasattr(parent_class, 'semantic_bits'):
            self.semantic_bits.update(parent_class.semantic_bits)

class PhysicalEntity(SemanticEntity):
    def __init__(self):
        super().__init__()
        self.semantic_bits = {
            "exists_in_space": True,
            "affected_by_gravity": True,
            "has_location": True,
            "can_move": True,
            "physical_effort_required": 2
        }

class LivingThing(PhysicalEntity):
    def __init__(self):
        super().__init__()
        self.semantic_bits.update({
            "can_be_tired": True,
            "needs_energy": True,
            "mortality": True,
            "does_harm_rating": 1,  # living things avoid harm
            "responds_to_environment": True
        })

class Human(LivingThing):
    def __init__(self):
        super().__init__()
        self.semantic_bits.update({
            "has_intentions": True,
            "social_being": True,
            "uses_language": True,
            "follows_social_norms": True,
            "emotional_valence": 5,
            "social_cooperation": 7,
            "intentionality": 8
        })

class GroupInclusiveSpeaker(Human):
    def __init__(self):
        super().__init__()
        self.semantic_bits.update({
            "cardinality": "multiple",
            "includes_narrator": True,
            "coordination_needed": True,
            "social_cooperation": 8,  # override: groups cooperate more
            "planning_required": 6
        })

class Movement(SemanticEntity):
    def __init__(self):
        super().__init__()
        self.semantic_bits = {
            "changes_location": True,
            "requires_time": True,
            "physical_effort": 3,
            "intentional_action": True
        }

class PastCompletedMovement(Movement):
    def __init__(self):
        super().__init__()
        self.semantic_bits.update({
            "temporal_state": "past",
            "completion_status": "completed",
            "success_likelihood": 9,  # if past, likely succeeded
            "certainty_level": 10
        })

class FinancialInstitution(PhysicalEntity):
    def __init__(self):
        super().__init__()
        self.semantic_bits.update({
            "formal_environment": True,
            "requires_documentation": True,
            "regulated_space": True,
            "formality_level": 8,
            "authority_interaction": 7,
            "security_present": True,
            "routine_vs_exceptional": 8
        })

# Usage example
def create_semantic_sentence():
    actor = GroupInclusiveSpeaker()
    action = PastCompletedMovement()
    target = FinancialInstitution()
    
    # Combine all inherited semantic bits
    sentence_context = {}
    sentence_context.update(actor.semantic_bits)
    sentence_context.update(action.semantic_bits)
    sentence_context.update(target.semantic_bits)
    
    return {
        "tokens": ["group-inclusive-speaker", "moved-past-completed", "bank-money"],
        "inherited_semantics": sentence_context
    }

# Execute
result = create_semantic_sentence()
print("Semantic bits automatically inherited:")
for key, value in result["inherited_semantics"].items():
    print(f"  {key}: {value}")

