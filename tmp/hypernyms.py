#!/usr/bin/env python3
"""
Hypernym Hierarchy Reducer

Converts WordNet hypernym hierarchies into concise JSON structures
for efficient LLM processing. Reduces token count while preserving
essential semantic relationships.
"""

import json
import re
from typing import Dict, List, Set, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

class SemanticLevel(Enum):
    """Semantic abstraction levels for filtering relevance"""
    SPECIFIC = 1      # Direct category (e.g., "female person")
    GENERAL = 2       # Broader category (e.g., "living thing") 
    ABSTRACT = 3      # High-level abstraction (e.g., "entity")
    UNIVERSAL = 4     # Too abstract for most contexts

@dataclass
class ConceptNode:
    """Represents a semantic concept with metadata"""
    term: str
    synonyms: List[str]
    level: SemanticLevel
    domain: Optional[str] = None
    biological_classification: Optional[Dict[str, str]] = None

class HypernymReducer:
    """
    Reduces WordNet hypernym hierarchies to essential semantic structures.
    
    Like having a librarian who can summarize the entire Dewey Decimal
    System into just the categories you actually need for understanding.
    """
    
    def __init__(self):
        # Core semantic categories that are usually relevant
        self.core_categories = {
            # Biological
            'organism', 'living_thing', 'life_form', 'creature',
            'animal', 'plant', 'human', 'person', 'individual',
            
            # Physical
            'object', 'artifact', 'device', 'tool', 'machine',
            'substance', 'material', 'structure',
            
            # Abstract but useful
            'concept', 'idea', 'quality', 'attribute', 'property',
            'action', 'activity', 'process', 'event', 'state',
            
            # Social/Human
            'role', 'occupation', 'relationship', 'group', 'organization',
            'place', 'location', 'building', 'area'
        }
        
        # Terms too abstract to be useful
        self.abstract_terms = {
            'entity', 'thing', 'whole', 'unit', 'part', 'component',
            'physical_entity', 'abstraction', 'psychological_feature',
            'cognition', 'content', 'causal_agent', 'cause', 'causal_agency'
        }
        
        # Biological classification patterns
        self.bio_patterns = {
            'kingdom': ['animalia', 'plantae', 'fungi', 'protista', 'monera'],
            'phylum': ['chordata', 'arthropoda', 'mollusca', 'cnidaria'],
            'class': ['mammalia', 'aves', 'reptilia', 'amphibia', 'pisces'],
            'order': ['primates', 'carnivora', 'rodentia', 'artiodactyla'],
            'family': ['hominidae', 'felidae', 'canidae', 'bovidae'],
            'genus': ['homo', 'felis', 'canis', 'bos'],
            'species': ['sapiens', 'domesticus', 'lupus']
        }
    
    def parse_hierarchy_text(self, hierarchy_text: str) -> Dict[str, any]:
        """Parse the WordNet-style hierarchy text into structured data"""
        lines = hierarchy_text.strip().split('\n')
        
        # Extract the main term from first line
        first_line = lines[0].strip()
        main_term_match = re.match(r'^([^,]+)', first_line)
        main_term = main_term_match.group(1).strip() if main_term_match else "unknown"
        
        # Extract synonyms from first line
        synonyms = []
        if ',' in first_line:
            synonym_part = first_line.split(',', 1)[1].strip()
            synonyms = [s.strip() for s in synonym_part.split(',')]
        
        # Parse hierarchy paths
        paths = []
        current_path = []
        
        for line in lines:
            # Count indentation level
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            # Extract terms from line
            clean_line = line.strip()
            if '=>' in clean_line:
                terms_part = clean_line.split('=>')[1].strip()
                terms = [t.strip() for t in terms_part.split(',')]
                
                # Adjust path based on indentation
                current_path = current_path[:indent_level]
                current_path.extend(terms)
                paths.append(current_path.copy())
        
        return {
            'main_term': main_term,
            'synonyms': synonyms,
            'hierarchy_paths': paths
        }
    
    def extract_semantic_concepts(self, parsed_data: Dict) -> List[ConceptNode]:
        """Extract relevant semantic concepts from hierarchy paths"""
        concepts = []
        seen_terms = set()
        
        main_term = parsed_data['main_term']
        synonyms = parsed_data['synonyms']
        
        # Add the main concept
        main_concept = ConceptNode(
            term=main_term,
            synonyms=synonyms,
            level=SemanticLevel.SPECIFIC,
            biological_classification=self._extract_bio_classification(parsed_data)
        )
        concepts.append(main_concept)
        seen_terms.add(main_term)
        
        # Process hierarchy paths
        for path in parsed_data['hierarchy_paths']:
            for term in path:
                clean_term = self._clean_term(term)
                
                if clean_term in seen_terms or clean_term in self.abstract_terms:
                    continue
                
                # Determine semantic level and relevance
                level = self._determine_semantic_level(clean_term)
                if level == SemanticLevel.UNIVERSAL:
                    continue
                
                concept = ConceptNode(
                    term=clean_term,
                    synonyms=[],
                    level=level,
                    domain=self._identify_domain(clean_term)
                )
                
                concepts.append(concept)
                seen_terms.add(clean_term)
        
        return concepts
    
    def _clean_term(self, term: str) -> str:
        """Clean and normalize terms"""
        # Remove extra spaces and convert to lowercase
        clean = re.sub(r'\s+', ' ', term.strip().lower())
        
        # Handle compound terms
        clean = clean.replace(' ', '_')
        
        return clean
    
    def _determine_semantic_level(self, term: str) -> SemanticLevel:
        """Determine the semantic abstraction level of a term"""
        term_lower = term.lower()
        
        if term_lower in self.abstract_terms:
            return SemanticLevel.UNIVERSAL
        
        if term_lower in self.core_categories:
            return SemanticLevel.GENERAL
        
        # Check for biological terms
        if any(term_lower in patterns for patterns in self.bio_patterns.values()):
            return SemanticLevel.GENERAL
        
        # Check for specific descriptors
        if any(desc in term_lower for desc in ['adult', 'young', 'male', 'female']):
            return SemanticLevel.SPECIFIC
        
        return SemanticLevel.GENERAL
    
    def _identify_domain(self, term: str) -> Optional[str]:
        """Identify the semantic domain of a term"""
        term_lower = term.lower()
        
        biological_terms = ['organism', 'living_thing', 'creature', 'animal', 'plant', 'human', 'person']
        if any(bio in term_lower for bio in biological_terms):
            return 'biological'
        
        physical_terms = ['object', 'artifact', 'device', 'tool', 'machine', 'substance']
        if any(phys in term_lower for phys in physical_terms):
            return 'physical'
        
        social_terms = ['role', 'occupation', 'group', 'organization', 'relationship']
        if any(soc in term_lower for soc in social_terms):
            return 'social'
        
        return None
    
    def _extract_bio_classification(self, parsed_data: Dict) -> Optional[Dict[str, str]]:
        """Extract biological classification if present"""
        classification = {}
        
        # Look through all terms in hierarchy
        all_terms = [parsed_data['main_term']] + parsed_data['synonyms']
        for path in parsed_data['hierarchy_paths']:
            all_terms.extend(path)
        
        for term in all_terms:
            term_lower = self._clean_term(term)
            
            for rank, examples in self.bio_patterns.items():
                if term_lower in examples:
                    classification[rank] = term_lower
        
        # Add common classifications for humans
        main_term = parsed_data['main_term'].lower()
        if any(human_term in main_term for human_term in ['person', 'human', 'man', 'woman', 'individual']):
            classification.update({
                'kingdom': 'animalia',
                'phylum': 'chordata', 
                'class': 'mammalia',
                'order': 'primates',
                'family': 'hominidae',
                'genus': 'homo',
                'species': 'sapiens'
            })
        
        return classification if classification else None
    
    def create_compact_representation(self, concepts: List[ConceptNode]) -> Dict[str, any]:
        """Create a compact JSON representation optimized for LLM processing"""
        
        # Separate concepts by level
        specific_concepts = [c for c in concepts if c.level == SemanticLevel.SPECIFIC]
        general_concepts = [c for c in concepts if c.level == SemanticLevel.GENERAL]
        
        # Build compact structure
        compact = {
            'primary': specific_concepts[0].term if specific_concepts else concepts[0].term,
            'synonyms': specific_concepts[0].synonyms if specific_concepts else [],
            'categories': [c.term for c in general_concepts[:3]],  # Limit to top 3
            'domains': list(set(c.domain for c in concepts if c.domain))[:2]  # Max 2 domains
        }
        
        # Add biological classification if present
        bio_classification = None
        for concept in concepts:
            if concept.biological_classification:
                bio_classification = concept.biological_classification
                break
        
        if bio_classification:
            # Include only the most relevant taxonomic levels
            relevant_levels = ['kingdom', 'class', 'order', 'family', 'genus', 'species']
            compact['taxonomy'] = {
                level: bio_classification[level] 
                for level in relevant_levels 
                if level in bio_classification
            }
        
        return compact
    
    def reduce_hierarchy(self, hierarchy_text: str) -> Dict[str, any]:
        """Main method: reduce WordNet hierarchy to compact JSON"""
        try:
            # Parse the hierarchy text
            parsed_data = self.parse_hierarchy_text(hierarchy_text)
            
            # Extract semantic concepts
            concepts = self.extract_semantic_concepts(parsed_data)
            
            # Create compact representation
            compact_rep = self.create_compact_representation(concepts)
            
            # Add metadata
            compact_rep['_meta'] = {
                'original_terms': len(set(
                    [parsed_data['main_term']] + 
                    parsed_data['synonyms'] + 
                    [term for path in parsed_data['hierarchy_paths'] for term in path]
                )),
                'reduced_terms': len(compact_rep.get('categories', [])) + 1,
                'compression_ratio': round(
                    len(compact_rep.get('categories', [])) / max(1, len(set(
                        [parsed_data['main_term']] + 
                        [term for path in parsed_data['hierarchy_paths'] for term in path]
                    ))), 2
                )
            }
            
            return compact_rep
            
        except Exception as e:
            return {
                'error': f"Failed to process hierarchy: {str(e)}",
                'primary': 'unknown',
                'categories': [],
                'domains': []
            }


def example1_usage():
    """Demonstrate the hypernym reducer with the woman example"""
    
    woman_hierarchy = """woman, adult female
       => female, female person
           => person, individual, someone, somebody, mortal, soul
               => organism, being
                   => living thing, animate thing
                       => whole, unit
                           => object, physical object
                               => physical entity
                                   => entity
               => causal agent, cause, causal agency
                   => physical entity
                       => entity
       => adult, grownup
           => person, individual, someone, somebody, mortal, soul
               => organism, being
                   => living thing, animate thing
                       => whole, unit
                           => object, physical object
                               => physical entity
                                   => entity
               => causal agent, cause, causal agency
                   => physical entity
                       => entity"""

def example_usage():
    """Demonstrate the hypernym reducer with the woman example"""

    woman_hierarchy = """rock, stone
       => natural object
           => whole, unit
               => object, physical object
                   => physical entity
                     => entity"""
    
    reducer = HypernymReducer()
    result = reducer.reduce_hierarchy(woman_hierarchy)
    
    print("Original hierarchy:")
    print(woman_hierarchy)
    print("\n" + "="*50 + "\n")
    print("Reduced JSON structure:")
    print(json.dumps(result, indent=2))
    
    # Show token count reduction
    original_tokens = len(woman_hierarchy.split())
    reduced_tokens = len(json.dumps(result, separators=(',', ':')).split())
    
    print(f"\nToken reduction:")
    print(f"Original: ~{original_tokens} tokens")
    print(f"Reduced: ~{reduced_tokens} tokens")
    print(f"Reduction: {((original_tokens - reduced_tokens) / original_tokens * 100):.1f}%")

if __name__ == "__main__":
    example_usage()

