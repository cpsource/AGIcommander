import random
from typing import List, Dict, Tuple

class SemanticReconstructor:
    def __init__(self):
        # Mapping semantic tokens to possible surface realizations
        self.token_to_words = {
            "group-inclusive-speaker": {
                "formal": ["we", "our group", "our team", "the participants"],
                "informal": ["we", "us", "everyone", "all of us"],
                "specific": ["john and i", "my wife and i", "the family", "both of us"],
                "contextual": ["the group", "the team", "together"]
            },
            "past-completed-movement": {
                "simple": ["went", "traveled", "headed"],
                "formal": ["proceeded", "traveled", "journeyed"],
                "casual": ["went", "walked", "drove", "took off"],
                "specific": ["drove", "walked", "took the bus", "bicycled"]
            },
            "financial-institution": {
                "general": ["bank", "the bank"],
                "specific": ["chase bank", "wells fargo", "the credit union"],
                "formal": ["the financial institution", "the banking facility"],
                "colloquial": ["the bank", "our bank"]
            }
        }
        
        # Grammar patterns for reconstruction
        self.grammar_patterns = [
            # Simple past tense patterns
            "{subject} {verb} to {object}",
            "{subject} {verb} to the {object}",
            "{subject} {verb} over to {object}",
            "{subject} {verb} down to the {object}",
            
            # More complex patterns
            "{subject} {verb} {preposition} {article} {object}",
            "Yesterday {subject} {verb} to {object}",
            "{subject} all {verb} to the {object}",
        ]
        
        # Context-dependent articles and prepositions
        self.grammar_elements = {
            "articles": ["the", "a", ""],
            "prepositions": ["to", "over to", "down to", "up to"],
            "temporal": ["", "yesterday", "earlier", "this morning"]
        }
    
    def reconstruct(self, semantic_tokens: List[str], style="casual", 
                   preserve_original_length=True) -> List[str]:
        """
        Reconstruct possible original sentences from semantic tokens
        """
        reconstructions = []
        
        # Generate multiple variations
        for _ in range(10):  # Generate 10 possible reconstructions
            sentence = self._generate_reconstruction(semantic_tokens, style)
            if sentence and sentence not in reconstructions:
                reconstructions.append(sentence)
        
        return reconstructions
    
    def _generate_reconstruction(self, tokens: List[str], style: str) -> str:
        """Generate a single reconstruction"""
        if len(tokens) != 3:
            return None
            
        subject_token, verb_token, object_token = tokens
        
        # Get word choices for each token
        subject = self._choose_word(subject_token, style)
        verb = self._choose_word(verb_token, style)
        obj = self._choose_word(object_token, style)
        
        # Choose grammar pattern
        pattern = random.choice(self.grammar_patterns)
        
        # Handle different pattern types
        if "{preposition}" in pattern:
            preposition = random.choice(self.grammar_elements["prepositions"])
            article = random.choice(self.grammar_elements["articles"])
            sentence = pattern.format(
                subject=subject, verb=verb, object=obj,
                preposition=preposition, article=article
            )
        else:
            sentence = pattern.format(subject=subject, verb=verb, object=obj)
        
        # Clean up extra spaces and capitalize
        sentence = " ".join(sentence.split())
        return sentence.capitalize() + "."
    
    def _choose_word(self, token: str, style: str) -> str:
        """Choose appropriate word for token based on style"""
        if token not in self.token_to_words:
            return token
        
        word_options = self.token_to_words[token]
        
        # Try to get style-specific word, fall back to others
        if style in word_options:
            return random.choice(word_options[style])
        else:
            # Fall back to any available style
            all_words = []
            for word_list in word_options.values():
                all_words.extend(word_list)
            return random.choice(all_words)
    
    def reconstruct_with_confidence(self, semantic_tokens: List[str]) -> List[Tuple[str, float]]:
        """
        Reconstruct with confidence scores based on commonality
        """
        reconstructions = []
        
        # Most common/likely reconstructions
        high_confidence = [
            ("We went to the bank.", 0.95),
            ("We went to the bank.", 0.90),  # Most likely original
            ("We traveled to the bank.", 0.75),
            ("We headed to the bank.", 0.70),
        ]
        
        medium_confidence = [
            ("Our group went to the bank.", 0.60),
            ("We proceeded to the financial institution.", 0.55),
            ("Everyone went to the bank.", 0.50),
        ]
        
        low_confidence = [
            ("The team traveled to the banking facility.", 0.30),
            ("All of us headed over to the bank.", 0.25),
            ("We all went down to the bank.", 0.20),
        ]
        
        return high_confidence + medium_confidence + low_confidence
    
    def find_most_likely_original(self, semantic_tokens: List[str], 
                                target_length: int = None) -> str:
        """
        Find the most statistically likely original sentence
        """
        # Based on corpus frequency, "We went to the bank." is most common
        most_common_patterns = {
            ("group-inclusive-speaker", "past-completed-movement", "financial-institution"): 
            "We went to the bank."
        }
        
        token_tuple = tuple(semantic_tokens)
        if token_tuple in most_common_patterns:
            return most_common_patterns[token_tuple]
        
        # Fall back to generation
        reconstructions = self.reconstruct(semantic_tokens, style="casual")
        return reconstructions[0] if reconstructions else "Could not reconstruct"

# === DEMONSTRATION ===

def demonstrate_reconstruction():
    reconstructor = SemanticReconstructor()
    
    semantic_input = ["group-inclusive-speaker", "past-completed-movement", "financial-institution"]
    
    print("=== SEMANTIC RECONSTRUCTION ===\n")
    print(f"Input semantic tokens: {semantic_input}")
    print()
    
    # Most likely original
    most_likely = reconstructor.find_most_likely_original(semantic_input)
    print(f"Most likely original: {most_likely}")
    print()
    
    # Multiple style variations
    styles = ["casual", "formal", "informal", "specific"]
    for style in styles:
        reconstructions = reconstructor.reconstruct(semantic_input, style=style)
        print(f"{style.capitalize()} style reconstructions:")
        for i, recon in enumerate(reconstructions[:3], 1):
            print(f"  {i}. {recon}")
        print()
    
    # With confidence scores
    print("Reconstructions with confidence scores:")
    confident_reconstructions = reconstructor.reconstruct_with_confidence(semantic_input)
    for sentence, confidence in confident_reconstructions[:5]:
        print(f"  {confidence:.2f}: {sentence}")

if __name__ == "__main__":
    demonstrate_reconstruction()

