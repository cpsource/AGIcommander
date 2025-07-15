#!/usr/bin/env python3
"""
Metadata Feature Extractor

This module extracts structural and statistical features from text that can be
learned quickly in Stage 1 training. Think of it as extracting the "grammar rules"
before learning to understand meaning.
"""

import re
import string
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class MetadataExtractor:
    """
    Extracts metadata features that represent structural and statistical 
    properties of text, which can be learned quickly in the first training stage.
    
    Like analyzing the structure of a building before understanding what happens inside.
    """
    
    def __init__(self, language='en'):
        self.language = language
        
        # Load spaCy model for linguistic analysis
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. "
                  "Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Common punctuation patterns
        self.punct_patterns = {
            'question_marks': r'\?',
            'exclamation_marks': r'!',
            'periods': r'\.',
            'commas': r',',
            'semicolons': r';',
            'colons': r':',
            'quotation_marks': r'["\']',
            'parentheses': r'[\(\)]',
            'brackets': r'[\[\]]'
        }
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata features from text.
        
        Returns a dictionary with different categories of features.
        """
        
        if not text or not text.strip():
            return self._empty_features()
        
        # Clean text for processing
        clean_text = text.strip()
        
        features = {}
        
        # 1. Length and size features
        features.update(self._extract_length_features(clean_text))
        
        # 2. Punctuation and formatting features  
        features.update(self._extract_punctuation_features(clean_text))
        
        # 3. Lexical diversity features
        features.update(self._extract_lexical_features(clean_text))
        
        # 4. Syntactic features (if spaCy available)
        if self.nlp:
            features.update(self._extract_syntactic_features(clean_text))
        
        # 5. Readability features
        features.update(self._extract_readability_features(clean_text))
        
        # 6. Statistical features
        features.update(self._extract_statistical_features(clean_text))
        
        # Convert to consistent format (all numeric)
        return self._normalize_features(features)
    
    def _extract_length_features(self, text: str) -> Dict[str, float]:
        """Extract features related to text length and structure"""
        
        # Basic length metrics
        char_count = len(text)
        word_count = len(word_tokenize(text))
        sent_count = len(sent_tokenize(text))
        
        # Paragraph count (simple heuristic)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Average lengths
        avg_word_length = np.mean([len(word) for word in word_tokenize(text)]) if word_count > 0 else 0
        avg_sent_length = word_count / sent_count if sent_count > 0 else 0
        
        return {
            'char_count': float(char_count),
            'word_count': float(word_count),
            'sentence_count': float(sent_count),
            'paragraph_count': float(paragraph_count),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sent_length,
            'char_to_word_ratio': char_count / word_count if word_count > 0 else 0
        }
    
    def _extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation and formatting patterns"""
        
        features = {}
        text_length = len(text)
        
        # Count different punctuation types
        for punct_name, pattern in self.punct_patterns.items():
            count = len(re.findall(pattern, text))
            features[f'{punct_name}_count'] = float(count)
            features[f'{punct_name}_density'] = count / text_length if text_length > 0 else 0
        
        # Overall punctuation density
        total_punct = sum(1 for char in text if char in string.punctuation)
        features['total_punctuation_density'] = total_punct / text_length if text_length > 0 else 0
        
        # Capitalization patterns
        upper_count = sum(1 for char in text if char.isupper())
        features['uppercase_density'] = upper_count / text_length if text_length > 0 else 0
        
        # Digital character patterns
        digit_count = sum(1 for char in text if char.isdigit())
        features['digit_density'] = digit_count / text_length if text_length > 0 else 0
        
        return features
    
    def _extract_lexical_features(self, text: str) -> Dict[str, float]:
        """Extract lexical diversity and vocabulary features"""
        
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]  # Only alphabetic words
        
        if not words:
            return {
                'lexical_diversity': 0.0,
                'stopword_ratio': 0.0,
                'unique_word_ratio': 0.0,
                'long_word_ratio': 0.0
            }
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        
        # Stopword analysis
        stopword_count = sum(1 for word in words if word in self.stop_words)
        stopword_ratio = stopword_count / len(words)
        
        # Long words (> 6 characters)
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / len(words)
        
        # Unique word ratio
        unique_word_ratio = len(unique_words) / len(words)
        
        return {
            'lexical_diversity': lexical_diversity,
            'stopword_ratio': stopword_ratio,
            'unique_word_ratio': unique_word_ratio,
            'long_word_ratio': long_word_ratio
        }
    
    def _extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """Extract syntactic patterns using spaCy"""
        
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            # POS tag distribution
            pos_counts = Counter([token.pos_ for token in doc])
            total_tokens = len(doc)
            
            pos_features = {}
            for pos_tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ']:
                count = pos_counts.get(pos_tag, 0)
                pos_features[f'{pos_tag.lower()}_ratio'] = count / total_tokens if total_tokens > 0 else 0
            
            # Named entity features
            ent_counts = Counter([ent.label_ for ent in doc.ents])
            ent_features = {
                'named_entity_density': len(doc.ents) / total_tokens if total_tokens > 0 else 0,
                'person_entities': ent_counts.get('PERSON', 0),
                'org_entities': ent_counts.get('ORG', 0),
                'location_entities': ent_counts.get('GPE', 0)
            }
            
            # Dependency features
            dep_counts = Counter([token.dep_ for token in doc])
            dep_features = {
                'root_ratio': dep_counts.get('ROOT', 0) / total_tokens if total_tokens > 0 else 0,
                'compound_ratio': dep_counts.get('compound', 0) / total_tokens if total_tokens > 0 else 0
            }
            
            # Combine all syntactic features
            features = {**pos_features, **ent_features, **dep_features}
            
            return features
            
        except Exception as e:
            print(f"Warning: Error in syntactic analysis: {e}")
            return {}
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity metrics"""
        
        try:
            # Flesch Reading Ease (higher = easier)
            flesch_ease = flesch_reading_ease(text)
            
            # Flesch-Kincaid Grade Level
            fk_grade = flesch_kincaid_grade(text)
            
            return {
                'flesch_reading_ease': float(flesch_ease),
                'flesch_kincaid_grade': float(fk_grade)
            }
            
        except Exception as e:
            print(f"Warning: Error computing readability: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0
            }
    
    def _extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Extract statistical patterns in the text"""
        
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        if not words:
            return {
                'word_freq_entropy': 0.0,
                'char_freq_entropy': 0.0,
                'most_common_word_freq': 0.0
            }
        
        # Word frequency distribution
        word_counts = Counter(words)
        word_freqs = np.array(list(word_counts.values()))
        word_freq_entropy = self._calculate_entropy(word_freqs)
        
        # Character frequency distribution
        chars = [char.lower() for char in text if char.isalpha()]
        char_counts = Counter(chars)
        char_freqs = np.array(list(char_counts.values()))
        char_freq_entropy = self._calculate_entropy(char_freqs)
        
        # Most common word frequency
        most_common_freq = word_counts.most_common(1)[0][1] / len(words) if word_counts else 0
        
        return {
            'word_freq_entropy': word_freq_entropy,
            'char_freq_entropy': char_freq_entropy,
            'most_common_word_freq': most_common_freq
        }
    
    def _calculate_entropy(self, frequencies: np.ndarray) -> float:
        """Calculate Shannon entropy of frequency distribution"""
        if len(frequencies) == 0:
            return 0.0
        
        # Normalize to probabilities
        probabilities = frequencies / np.sum(frequencies)
        
        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Ensure all features are numeric and handle any missing values"""
        
        normalized = {}
        
        for key, value in features.items():
            try:
                # Convert to float
                if isinstance(value, (int, float)):
                    normalized[key] = float(value)
                elif isinstance(value, str):
                    # Try to convert string numbers
                    normalized[key] = float(value)
                else:
                    # Default for non-numeric values
                    normalized[key] = 0.0
                
                # Handle infinite or NaN values
                if not np.isfinite(normalized[key]):
                    normalized[key] = 0.0
                    
            except (ValueError, TypeError):
                normalized[key] = 0.0
        
        return normalized
    
    def _empty_features(self) -> Dict[str, float]:
        """Return zero-filled features for empty text"""
        
        # Define all possible feature names with default values
        empty_features = {
            # Length features
            'char_count': 0.0, 'word_count': 0.0, 'sentence_count': 0.0,
            'paragraph_count': 0.0, 'avg_word_length': 0.0, 'avg_sentence_length': 0.0,
            'char_to_word_ratio': 0.0,
            
            # Punctuation features
            'question_marks_count': 0.0, 'exclamation_marks_count': 0.0,
            'periods_count': 0.0, 'commas_count': 0.0,
            'total_punctuation_density': 0.0, 'uppercase_density': 0.0,
            'digit_density': 0.0,
            
            # Lexical features
            'lexical_diversity': 0.0, 'stopword_ratio': 0.0,
            'unique_word_ratio': 0.0, 'long_word_ratio': 0.0,
            
            # Readability features
            'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0,
            
            # Statistical features
            'word_freq_entropy': 0.0, 'char_freq_entropy': 0.0,
            'most_common_word_freq': 0.0
        }
        
        return empty_features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names that this extractor produces"""
        
        # Extract features from a sample text to get all feature names
        sample_features = self.extract_features("This is a sample text for feature extraction.")
        return list(sample_features.keys())
    
    def get_feature_vector(self, text: str) -> np.ndarray:
        """Return features as a numpy array for ML models"""
        
        features = self.extract_features(text)
        feature_names = self.get_feature_names()
        
        # Ensure consistent ordering
        vector = [features.get(name, 0.0) for name in feature_names]
        
        return np.array(vector, dtype=np.float32)


def demo_metadata_extraction():
    """Demonstrate metadata extraction on sample texts"""
    
    extractor = MetadataExtractor()
    
    sample_texts = [
        "This is a short, simple sentence.",
        "This is a much longer and more complex sentence that contains multiple clauses, various punctuation marks, and demonstrates different linguistic patterns that our metadata extractor should be able to identify and quantify!",
        "What is the meaning of life? It's a question that philosophers have pondered for centuries. The answer, as Douglas Adams famously suggested, might just be 42.",
        "Machine learning models can benefit from two-stage training approaches. First, we train on metadata features. Then, we fine-tune on semantic features."
    ]
    
    print("Metadata Extraction Demo")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nText {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        print("-" * 30)
        
        features = extractor.extract_features(text)
        
        # Show some key features
        key_features = [
            'word_count', 'sentence_count', 'avg_word_length',
            'lexical_diversity', 'flesch_reading_ease', 'total_punctuation_density'
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"{feature}: {features[feature]:.3f}")
        
        print(f"Total features extracted: {len(features)}")


if __name__ == "__main__":
    demo_metadata_extraction()
