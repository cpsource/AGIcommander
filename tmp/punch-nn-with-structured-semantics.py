#!/usr/bin/env python3
"""
Semantic Neural Network Architecture

Transforms structured semantic objects into neural network representations
while preserving the rich hierarchical information and relationships.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json

@dataclass
class SemanticToken:
    """Structured semantic token with properties"""
    concept: str
    properties: Dict
    relationships: List[str]
    embedding_vector: Optional[torch.Tensor] = None

class SemanticEmbeddingLayer(nn.Module):
    """
    Converts structured semantic tokens into neural network representations.
    
    Think of this as a translator that takes rich semantic objects and converts
    them into the mathematical language that neural networks understand.
    """
    
    def __init__(self, 
                 concept_vocab_size: int = 50000,
                 property_vocab_size: int = 10000,
                 embedding_dim: int = 512,
                 max_properties: int = 8):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_properties = max_properties
        
        # Core concept embedding
        self.concept_embeddings = nn.Embedding(concept_vocab_size, embedding_dim)
        
        # Property embeddings (for different types of properties)
        self.property_key_embeddings = nn.Embedding(property_vocab_size, embedding_dim // 4)
        self.property_value_embeddings = nn.Embedding(property_vocab_size, embedding_dim // 4)
        
        # Specialized property encoders for different data types
        self.numeric_encoder = nn.Linear(1, embedding_dim // 4)
        self.taxonomy_encoder = nn.Linear(10, embedding_dim // 4)  # For biological classifications
        self.emotional_encoder = nn.Linear(4, embedding_dim // 4)  # For emotional states
        
        # Attention mechanism for property importance
        self.property_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim // 4,
            num_heads=4,
            batch_first=True
        )
        
        # Combination layers
        self.concept_property_fusion = nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Concept vocabulary mapping
        self.concept_vocab = self._build_concept_vocabulary()
        self.property_vocab = self._build_property_vocabulary()
    
    def _build_concept_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping for semantic concepts"""
        base_vocab = {
            # Biological concepts
            "homo_sapiens": 1,
            "female_person": 2,
            "male_person": 3,
            "living_thing": 4,
            "organism": 5,
            
            # Action concepts  
            "bipedal_locomotion": 10,
            "rapid_movement": 11,
            "facial_assault": 12,
            "communication": 13,
            "pursuit": 14,
            
            # Emotional concepts
            "anger": 20,
            "calm": 21,
            "fear": 22,
            "joy": 23,
            
            # Location concepts
            "public_roadway": 30,
            "street": 31,
            "intersection": 32,
            
            # Ethical concepts
            "FIRST_LAW_VIOLATION": 40,
            "ethical_approved": 41,
            "harm_assessment": 42,
            
            # Relationship concepts
            "romantic_partner": 50,
            "family_member": 51,
            "stranger": 52,
            
            # Special tokens
            "<PAD>": 0,
            "<UNK>": 99,
        }
        return base_vocab
    
    def _build_property_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for property keys and values"""
        return {
            # Property keys
            "intensity": 1,
            "valence": 2,
            "activation": 3,
            "severity": 4,
            "probability": 5,
            "speed_modifier": 6,
            "harm_score": 7,
            
            # Property values
            "high": 10,
            "medium": 11,
            "low": 12,
            "positive": 13,
            "negative": 14,
            "neutral": 15,
            
            # Numeric indicators
            "NUMERIC": 100,
            "TAXONOMY": 101,
            "EMOTIONAL": 102,
            
            "<PAD>": 0,
        }
    
    def encode_semantic_token(self, semantic_object: Dict) -> torch.Tensor:
        """
        Convert a single semantic object into neural representation
        
        Input: {"concept": "angry_female_homo_sapiens", "properties": {"intensity": 0.9, "valence": "negative"}}
        Output: [512-dim embedding vector]
        """
        
        # 1. Encode main concept
        concept = semantic_object.get("concept", "<UNK>")
        concept_id = self.concept_vocab.get(concept, self.concept_vocab["<UNK>"])
        concept_embedding = self.concept_embeddings(torch.tensor(concept_id))
        
        # 2. Encode properties
        properties = semantic_object.get("properties", {})
        property_embeddings = []
        
        for key, value in properties.items():
            if len(property_embeddings) >= self.max_properties:
                break
                
            # Encode property key
            key_id = self.property_vocab.get(key, 0)
            key_embedding = self.property_key_embeddings(torch.tensor(key_id))
            
            # Encode property value based on type
            if isinstance(value, (int, float)):
                # Numeric property
                value_embedding = self.numeric_encoder(torch.tensor([float(value)]))
            elif key in ["taxonomy", "biological_classification"]:
                # Biological taxonomy
                tax_vector = self._encode_taxonomy(value)
                value_embedding = self.taxonomy_encoder(tax_vector)
            elif key in ["emotional_state", "emotion"]:
                # Emotional state
                emotion_vector = self._encode_emotional_state(value)
                value_embedding = self.emotional_encoder(emotion_vector)
            else:
                # String property
                value_id = self.property_vocab.get(str(value), 0)
                value_embedding = self.property_value_embeddings(torch.tensor(value_id))
            
            # Combine key and value
            property_embedding = key_embedding + value_embedding
            property_embeddings.append(property_embedding)
        
        # 3. Pad properties to max_properties
        while len(property_embeddings) < self.max_properties:
            property_embeddings.append(torch.zeros(self.embedding_dim // 4))
        
        # 4. Apply attention to properties
        property_tensor = torch.stack(property_embeddings).unsqueeze(0)  # [1, max_props, dim/4]
        attended_properties, _ = self.property_attention(
            property_tensor, property_tensor, property_tensor
        )
        aggregated_properties = attended_properties.mean(dim=1).squeeze(0)  # [dim/4]
        
        # 5. Fuse concept and properties
        combined = torch.cat([concept_embedding, aggregated_properties])
        fused = self.concept_property_fusion(combined)
        
        # 6. Apply normalization and dropout
        output = self.layer_norm(fused)
        output = self.dropout(output)
        
        return output
    
    def _encode_taxonomy(self, taxonomy: Dict) -> torch.Tensor:
        """Encode biological taxonomy into fixed-size vector"""
        # Create 10-dimensional vector for taxonomy levels
        tax_vector = torch.zeros(10)
        
        taxonomy_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        taxonomy_values = {
            "animalia": 1.0, "plantae": 0.8, "fungi": 0.6,
            "chordata": 1.0, "arthropoda": 0.8,
            "mammalia": 1.0, "aves": 0.8, "reptilia": 0.6,
            "primates": 1.0, "carnivora": 0.8, "rodentia": 0.6,
            "hominidae": 1.0, "felidae": 0.8, "canidae": 0.6,
            "homo": 1.0, "felis": 0.8, "canis": 0.6,
            "sapiens": 1.0, "domesticus": 0.8, "lupus": 0.6
        }
        
        for i, level in enumerate(taxonomy_levels):
            if level in taxonomy:
                value = taxonomy[level]
                tax_vector[i] = taxonomy_values.get(value, 0.5)
        
        return tax_vector
    
    def _encode_emotional_state(self, emotion: Union[str, Dict]) -> torch.Tensor:
        """Encode emotional state into 4-dimensional vector"""
        if isinstance(emotion, str):
            emotion_dict = {"primary": emotion, "intensity": 0.5}
        else:
            emotion_dict = emotion
        
        # 4D emotional encoding: [valence, arousal, intensity, stability]
        emotion_mappings = {
            "anger": [0.0, 0.9, 0.8, 0.3],      # negative, high arousal, high intensity, unstable
            "joy": [1.0, 0.7, 0.7, 0.8],        # positive, high arousal, high intensity, stable
            "sadness": [0.0, 0.2, 0.6, 0.6],    # negative, low arousal, medium intensity, stable
            "fear": [0.0, 0.9, 0.8, 0.2],       # negative, high arousal, high intensity, unstable
            "calm": [0.5, 0.1, 0.3, 0.9],       # neutral, low arousal, low intensity, very stable
        }
        
        primary_emotion = emotion_dict.get("primary", "calm")
        base_vector = emotion_mappings.get(primary_emotion, [0.5, 0.5, 0.5, 0.5])
        
        # Adjust intensity if provided
        if "intensity" in emotion_dict:
            intensity_modifier = emotion_dict["intensity"]
            base_vector[2] = intensity_modifier  # Override intensity
        
        return torch.tensor(base_vector, dtype=torch.float32)
    
    def forward(self, semantic_sequence: List[Dict]) -> torch.Tensor:
        """
        Process a sequence of semantic objects
        
        Input: [
            {"concept": "angry_female_homo_sapiens", "properties": {"intensity": 0.9}},
            {"concept": "rapid_bipedal_pursuit", "properties": {"speed_modifier": 1.8}},
            {"concept": "FIRST_LAW_VIOLATION", "properties": {"harm_score": 0.94}}
        ]
        Output: [seq_len, embedding_dim] tensor ready for transformer
        """
        
        embeddings = []
        for semantic_obj in semantic_sequence:
            embedding = self.encode_semantic_token(semantic_obj)
            embeddings.append(embedding)
        
        return torch.stack(embeddings)

class SemanticTransformer(nn.Module):
    """
    Transformer that operates on semantic embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 feedforward_dim: int = 2048):
        super().__init__()
        
        self.semantic_embedding = SemanticEmbeddingLayer(embedding_dim=embedding_dim)
        
        # Positional encoding for sequence order
        self.pos_encoding = nn.Parameter(torch.randn(1000, embedding_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projections
        self.concept_classifier = nn.Linear(embedding_dim, 50000)  # Predict next concept
        self.property_predictor = nn.Linear(embedding_dim, 100)    # Predict properties
        self.ethical_assessor = nn.Linear(embedding_dim, 5)        # Asimov law compliance
        
    def forward(self, semantic_sequence: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process semantic sequence and generate predictions
        """
        
        # Convert semantic objects to embeddings
        embeddings = self.semantic_embedding(semantic_sequence)  # [seq_len, embed_dim]
        
        # Add positional encoding
        seq_len = embeddings.size(0)
        embeddings = embeddings + self.pos_encoding[:seq_len]
        
        # Add batch dimension and process through transformer
        embeddings = embeddings.unsqueeze(0)  # [1, seq_len, embed_dim]
        
        # Transformer processing
        hidden_states = self.transformer(embeddings)  # [1, seq_len, embed_dim]
        
        # Generate predictions
        last_hidden = hidden_states[0, -1, :]  # Last token representation
        
        predictions = {
            "next_concept": self.concept_classifier(last_hidden),
            "next_properties": self.property_predictor(last_hidden),
            "ethical_assessment": self.ethical_assessor(last_hidden)
        }
        
        return predictions

def demonstrate_semantic_neural_architecture():
    """Demonstrate the semantic neural architecture"""
    
    # Example semantic sequence for "She angrily ran down the street and punched her boyfriend"
    semantic_sequence = [
        {
            "concept": "angry_female_homo_sapiens",
            "properties": {
                "intensity": 0.9,
                "valence": "negative",
                "taxonomy": {"genus": "homo", "species": "sapiens"}
            }
        },
        {
            "concept": "rapid_bipedal_pursuit", 
            "properties": {
                "speed_modifier": 1.8,
                "purpose": "confrontation"
            }
        },
        {
            "concept": "intended_facial_assault",
            "properties": {
                "harm_score": 0.94,
                "target": "romantic_partner"
            }
        },
        {
            "concept": "FIRST_LAW_VIOLATION",
            "properties": {
                "severity": 4,
                "intervention_required": True
            }
        }
    ]
    
    print("SEMANTIC NEURAL ARCHITECTURE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize model
    model = SemanticTransformer(embedding_dim=512)
    
    print(f"Input: {len(semantic_sequence)} semantic concepts")
    for i, concept in enumerate(semantic_sequence):
        print(f"  {i+1}. {concept['concept']} with {len(concept['properties'])} properties")
    
    # Process through model
    with torch.no_grad():
        predictions = model(semantic_sequence)
    
    print(f"\nOutput shapes:")
    for key, tensor in predictions.items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"\nNeural representation ready for:")
    print(f"  • Next concept prediction")
    print(f"  • Property inference") 
    print(f"  • Ethical assessment")
    print(f"  • Timeline branching")
    print(f"  • Behavioral modeling")
    
    # Show embedding layer details
    embedding_layer = model.semantic_embedding
    sample_token = semantic_sequence[0]
    
    print(f"\nSemantic Token Processing:")
    print(f"  Input: {sample_token}")
    
    with torch.no_grad():
        embedding = embedding_layer.encode_semantic_token(sample_token)
    
    print(f"  Output: {embedding.shape} neural vector")
    print(f"  Captures: concept + properties + relationships")

if __name__ == "__main__":
    demonstrate_semantic_neural_architecture()
