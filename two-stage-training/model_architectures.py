#!/usr/bin/env python3
"""
Neural Network Model Architectures for Two-Stage Training

This module defines the neural network architectures used in the two-stage
training approach. Think of it as blueprints for different types of brains:
one that learns structure (metadata) and one that learns meaning (semantics).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MetadataEncoder(nn.Module):
    """
    Encoder for metadata features (Stage 1).
    
    This network learns structural patterns quickly - like learning
    grammar rules before understanding sentence meaning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 dropout: float = 0.1, num_layers: int = 2):
        super(MetadataEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Remove the last dropout for cleaner output
        layers = layers[:-1]
        
        self.encoder = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through metadata encoder.
        
        Args:
            metadata_features: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded metadata representation of shape (batch_size, hidden_dim)
        """
        
        # Handle potential NaN or infinite values
        metadata_features = torch.nan_to_num(metadata_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Encode metadata features
        encoded = self.encoder(metadata_features)
        
        # Apply output projection
        output = self.output_projection(encoded)
        
        return output


class SemanticEncoder(nn.Module):
    """
    Encoder for semantic features (Stage 2).
    
    This network learns deep semantic understanding - like understanding
    the meaning and context after knowing the grammar rules.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.1, use_attention: bool = True):
        super(SemanticEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-layer encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)
        ])
        
        # Self-attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through semantic encoder.
        
        Args:
            semantic_features: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded semantic representation of shape (batch_size, hidden_dim)
        """
        
        # Project to hidden dimension
        x = self.input_projection(semantic_features)
        
        # Add sequence dimension for transformer (treat as sequence of length 1)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply self-attention if enabled
        if self.use_attention:
            attended, _ = self.attention(x, x, x)
            x = x + attended  # Residual connection
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Apply normalization and output projection
        x = self.output_norm(x)
        output = self.output_projection(x)
        
        return output


class FeatureFusion(nn.Module):
    """
    Fuses metadata and semantic representations.
    
    This is where the magic happens - combining structural understanding
    with semantic understanding for better overall comprehension.
    """
    
    def __init__(self, metadata_dim: int, semantic_dim: int, 
                 fusion_dim: int = 256, fusion_type: str = "concat"):
        super(FeatureFusion, self).__init__()
        
        self.metadata_dim = metadata_dim
        self.semantic_dim = semantic_dim
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # Simple concatenation
            self.fusion_layer = nn.Linear(metadata_dim + semantic_dim, fusion_dim)
            
        elif fusion_type == "attention":
            # Attention-based fusion
            self.metadata_proj = nn.Linear(metadata_dim, fusion_dim)
            self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
            self.attention_weights = nn.Linear(fusion_dim * 2, 2)
            
        elif fusion_type == "gated":
            # Gated fusion mechanism
            self.metadata_proj = nn.Linear(metadata_dim, fusion_dim)
            self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
            self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, metadata_features: torch.Tensor, 
                semantic_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse metadata and semantic features.
        
        Args:
            metadata_features: Tensor of shape (batch_size, metadata_dim)
            semantic_features: Tensor of shape (batch_size, semantic_dim)
            
        Returns:
            Fused representation of shape (batch_size, fusion_dim)
        """
        
        if self.fusion_type == "concat":
            # Simple concatenation and projection
            combined = torch.cat([metadata_features, semantic_features], dim=1)
            fused = self.fusion_layer(combined)
            
        elif self.fusion_type == "attention":
            # Attention-based fusion
            meta_proj = self.metadata_proj(metadata_features)
            sem_proj = self.semantic_proj(semantic_features)
            
            # Compute attention weights
            combined = torch.cat([meta_proj, sem_proj], dim=1)
            attention_scores = F.softmax(self.attention_weights(combined), dim=1)
            
            # Apply attention
            fused = attention_scores[:, 0:1] * meta_proj + attention_scores[:, 1:2] * sem_proj
            
        elif self.fusion_type == "gated":
            # Gated fusion
            meta_proj = self.metadata_proj(metadata_features)
            sem_proj = self.semantic_proj(semantic_features)
            
            # Compute gate
            combined = torch.cat([meta_proj, sem_proj], dim=1)
            gate_values = torch.sigmoid(self.gate(combined))
            
            # Apply gating
            fused = gate_values * meta_proj + (1 - gate_values) * sem_proj
        
        # Apply normalization and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused


class TwoStageModel(nn.Module):
    """
    Complete two-stage model combining metadata and semantic encoders.
    
    This is the full architecture that first learns structure, then meaning,
    like a student who first learns grammar rules then applies them to understand literature.
    """
    
    def __init__(self, metadata_encoder: MetadataEncoder, 
                 semantic_encoder: SemanticEncoder,
                 hidden_dim: int = 512, num_classes: int = 2,
                 dropout: float = 0.1, fusion_type: str = "concat"):
        super(TwoStageModel, self).__init__()
        
        self.metadata_encoder = metadata_encoder
        self.semantic_encoder = semantic_encoder
        
        # Feature fusion
        self.fusion = FeatureFusion(
            metadata_dim=metadata_encoder.hidden_dim,
            semantic_dim=semantic_encoder.hidden_dim,
            fusion_dim=hidden_dim,
            fusion_type=fusion_type
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, metadata_features: torch.Tensor, 
                semantic_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            metadata_features: Tensor of shape (batch_size, metadata_dim)
            semantic_features: Tensor of shape (batch_size, semantic_dim)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        
        # Encode metadata and semantic features
        metadata_encoded = self.metadata_encoder(metadata_features)
        semantic_encoded = self.semantic_encoder(semantic_features)
        
        # Fuse the representations
        fused_features = self.fusion(metadata_encoded, semantic_encoded)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_representations(self, metadata_features: torch.Tensor, 
                          semantic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate representations for analysis.
        
        Returns:
            Tuple of (metadata_encoded, semantic_encoded, fused_features)
        """
        
        metadata_encoded = self.metadata_encoder(metadata_features)
        semantic_encoded = self.semantic_encoder(semantic_features)
        fused_features = self.fusion(metadata_encoded, semantic_encoded)
        
        return metadata_encoded, semantic_encoded, fused_features
    
    def freeze_metadata_encoder(self):
        """Freeze metadata encoder parameters for Stage 2 training"""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_metadata_encoder(self):
        """Unfreeze metadata encoder parameters"""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self) -> Tuple[int, int, int]:
        """
        Get number of parameters in different components.
        
        Returns:
            Tuple of (metadata_params, semantic_params, total_params)
        """
        
        metadata_params = sum(p.numel() for p in self.metadata_encoder.parameters())
        semantic_params = sum(p.numel() for p in self.semantic_encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return metadata_params, semantic_params, total_params


class AdaptiveUnfreezing(nn.Module):
    """
    Module for adaptive unfreezing of metadata encoder during Stage 2.
    
    This allows gradual introduction of metadata parameter updates
    as semantic training progresses.
    """
    
    def __init__(self, model: TwoStageModel, 
                 unfreezing_schedule: dict = None):
        super(AdaptiveUnfreezing, self).__init__()
        
        self.model = model
        self.unfreezing_schedule = unfreezing_schedule or {}
        self.current_epoch = 0
        
        # Store original parameter states
        self.metadata_params = list(model.metadata_encoder.parameters())
        self.param_frozen_state = [not p.requires_grad for p in self.metadata_params]
        
    def update_epoch(self, epoch: int):
        """Update current epoch and apply unfreezing schedule"""
        self.current_epoch = epoch
        
        # Check if any layers should be unfrozen
        epoch_key = f'epoch_{epoch}'
        if epoch_key in self.unfreezing_schedule:
            layers_to_unfreeze = self.unfreezing_schedule[epoch_key]
            self._unfreeze_layers(layers_to_unfreeze)
    
    def _unfreeze_layers(self, layer_names):
        """Unfreeze specific layers in metadata encoder"""
        
        if 'all_metadata_layers' in layer_names:
            self.model.unfreeze_metadata_encoder()
            print("Unfroze all metadata encoder layers")
            return
        
        # More granular unfreezing would require layer-specific naming
        # This is a simplified version
        for layer_name in layer_names:
            print(f"Unfreezing layer: {layer_name}")
            # Implementation depends on specific layer naming convention
    
    def forward(self, metadata_features: torch.Tensor, 
                semantic_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(metadata_features, semantic_features)


def model_summary(model: TwoStageModel):
    """Print a summary of the model architecture"""
    
    metadata_params, semantic_params, total_params = model.get_num_parameters()
    
    print("Two-Stage Model Summary")
    print("=" * 50)
    print(f"Metadata Encoder Parameters: {metadata_params:,}")
    print(f"Semantic Encoder Parameters: {semantic_params:,}")
    print(f"Total Parameters: {total_params:,}")
    print()
    
    print("Architecture:")
    print(f"  Metadata Encoder: {model.metadata_encoder.input_dim} -> {model.metadata_encoder.hidden_dim}")
    print(f"  Semantic Encoder: {model.semantic_encoder.input_dim} -> {model.semantic_encoder.hidden_dim}")
    print(f"  Fusion Type: {model.fusion.fusion_type}")
    print(f"  Classification: {model.fusion.fusion_dim} -> {len(model.classifier)} layers")


def create_sample_model():
    """Create a sample model for testing"""
    
    # Sample dimensions
    metadata_dim = 25  # From metadata extractor
    semantic_dim = 384  # From sentence transformer
    
    # Create encoders
    metadata_encoder = MetadataEncoder(
        input_dim=metadata_dim,
        hidden_dim=128,
        dropout=0.1
    )
    
    semantic_encoder = SemanticEncoder(
        input_dim=semantic_dim,
        hidden_dim=256,
        dropout=0.1,
        use_attention=True
    )
    
    # Create full model
    model = TwoStageModel(
        metadata_encoder=metadata_encoder,
        semantic_encoder=semantic_encoder,
        hidden_dim=512,
        num_classes=2,
        dropout=0.1,
        fusion_type="attention"
    )
    
    return model


if __name__ == "__main__":
    # Test the model architectures
    print("Testing Two-Stage Model Architectures")
    print("=" * 50)
    
    # Create sample model
    model = create_sample_model()
    model_summary(model)
    
    # Test forward pass
    batch_size = 4
    metadata_features = torch.randn(batch_size, 25)
    semantic_features = torch.randn(batch_size, 384)
    
    print("\nTesting forward pass...")
    
    # Test metadata encoder only
    with torch.no_grad():
        metadata_encoded = model.metadata_encoder(metadata_features)
        print(f"Metadata encoded shape: {metadata_encoded.shape}")
        
        # Test semantic encoder only
        semantic_encoded = model.semantic_encoder(semantic_features)
        print(f"Semantic encoded shape: {semantic_encoded.shape}")
        
        # Test full model
        output = model(metadata_features, semantic_features)
        print(f"Model output shape: {output.shape}")
        
        # Test representation extraction
        meta_repr, sem_repr, fused_repr = model.get_representations(
            metadata_features, semantic_features
        )
        print(f"Fused representation shape: {fused_repr.shape}")
    
    print("\nModel test completed successfully!")
