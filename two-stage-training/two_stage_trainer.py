#!/usr/bin/env python3
"""
Two-Stage Neural Network Training Implementation

This module implements the two-stage training approach:
1. Stage 1: Pre-train metadata encoder on structural features
2. Stage 2: Train semantic layers with frozen metadata encoder

Like teaching someone to drive by first learning traffic rules, then applying
them while actually driving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# Import our custom modules
from metadata_extractor import MetadataExtractor
from semantic_processor import SemanticProcessor
from model_architectures import TwoStageModel, MetadataEncoder, SemanticEncoder


class TextDataset(Dataset):
    """Dataset class for handling text and metadata features"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 metadata_extractor: MetadataExtractor,
                 semantic_processor: Optional[SemanticProcessor] = None):
        self.texts = texts
        self.labels = labels
        self.metadata_extractor = metadata_extractor
        self.semantic_processor = semantic_processor
        
        # Pre-extract metadata for efficiency
        print("Extracting metadata features...")
        self.metadata_features = [
            self.metadata_extractor.extract_features(text) 
            for text in texts
        ]
        
        if semantic_processor:
            print("Processing semantic features...")
            self.semantic_features = [
                self.semantic_processor.process(text, meta)
                for text, meta in zip(texts, self.metadata_features)
            ]
        else:
            self.semantic_features = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            'text': self.texts[idx],
            'metadata': self.metadata_features[idx],
            'label': self.labels[idx]
        }
        
        if self.semantic_features:
            item['semantic'] = self.semantic_features[idx]
            
        return item


class TwoStageTrainer:
    """
    Main trainer class implementing the two-stage training approach.
    
    Think of this as a coach who first teaches fundamentals (metadata),
    then applies those fundamentals to real game situations (semantic training).
    """
    
    def __init__(self, metadata_extractor: MetadataExtractor,
                 semantic_processor: SemanticProcessor,
                 config_path: str = None,
                 device: str = None):
        
        self.metadata_extractor = metadata_extractor
        self.semantic_processor = semantic_processor
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model components
        self.metadata_encoder = None
        self.semantic_encoder = None
        self.full_model = None
        
        # Training state
        self.stage_1_complete = False
        self.training_history = {'stage_1': [], 'stage_2': []}
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'stage_1': {
                    'epochs': 5,
                    'batch_size': 256,
                    'lr': 0.001,
                    'weight_decay': 1e-5
                },
                'stage_2': {
                    'epochs': 20,
                    'batch_size': 64,
                    'lr': 0.0001,
                    'weight_decay': 1e-5,
                    'freeze_metadata': True
                },
                'model': {
                    'metadata_dim': 128,
                    'semantic_dim': 256,
                    'hidden_dim': 512,
                    'num_classes': 2,
                    'dropout': 0.1
                }
            }
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_models(self, metadata_feature_dim: int, semantic_feature_dim: int):
        """Build the neural network components"""
        
        model_config = self.config['model']
        
        # Metadata encoder - learns structural patterns
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_feature_dim,
            hidden_dim=model_config['metadata_dim'],
            dropout=model_config['dropout']
        )
        
        # Semantic encoder - learns deep text understanding  
        self.semantic_encoder = SemanticEncoder(
            input_dim=semantic_feature_dim,
            hidden_dim=model_config['semantic_dim'],
            dropout=model_config['dropout']
        )
        
        # Combined model
        self.full_model = TwoStageModel(
            metadata_encoder=self.metadata_encoder,
            semantic_encoder=self.semantic_encoder,
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout']
        )
        
        # Move to device
        self.metadata_encoder.to(self.device)
        self.semantic_encoder.to(self.device) 
        self.full_model.to(self.device)
        
        self.logger.info(f"Models built and moved to {self.device}")
        
    def train_stage_1(self, train_texts: List[str], train_labels: List[int],
                     val_texts: List[str] = None, val_labels: List[int] = None):
        """
        Stage 1: Pre-train metadata encoder on structural features
        
        Like teaching grammar rules before reading complex literature.
        """
        
        self.logger.info("Starting Stage 1: Metadata Pre-training")
        
        # Create datasets (metadata only for stage 1)
        train_dataset = TextDataset(train_texts, train_labels, self.metadata_extractor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['stage_1']['batch_size'],
            shuffle=True
        )
        
        val_loader = None
        if val_texts and val_labels:
            val_dataset = TextDataset(val_texts, val_labels, self.metadata_extractor)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Build models if not already built
        if self.metadata_encoder is None:
            sample_batch = next(iter(train_loader))
            metadata_dim = len(sample_batch['metadata'][0])
            semantic_dim = 768  # Default BERT embedding size
            self.build_models(metadata_dim, semantic_dim)
        
        # Setup optimizer for metadata encoder only
        optimizer = optim.Adam(
            self.metadata_encoder.parameters(),
            lr=self.config['stage_1']['lr'],
            weight_decay=self.config['stage_1']['weight_decay']
        )
        
        # Use auxiliary task for metadata training (e.g., predicting text properties)
        criterion = nn.MSELoss()  # For regression tasks like length prediction
        
        # Training loop
        for epoch in range(self.config['stage_1']['epochs']):
            self.metadata_encoder.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Extract metadata features and create auxiliary targets
                metadata_features = torch.stack([
                    torch.tensor(meta['length_features'], dtype=torch.float32)
                    for meta in batch['metadata']
                ]).to(self.device)
                
                # Use text length as auxiliary target (can be extended)
                targets = torch.tensor([
                    len(text.split()) for text in batch['text']
                ], dtype=torch.float32).to(self.device)
                
                # Forward pass through metadata encoder
                encoded = self.metadata_encoder(metadata_features)
                
                # Predict auxiliary task (text length)
                predictions = torch.sum(encoded, dim=1)  # Simple aggregation
                
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation
            if val_loader:
                val_loss = self._validate_stage_1(val_loader, criterion)
                self.logger.info(f"Epoch {epoch+1}/{self.config['stage_1']['epochs']}: "
                               f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.config['stage_1']['epochs']}: "
                               f"Train Loss: {avg_loss:.4f}")
            
            # Save training history
            self.training_history['stage_1'].append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss if val_loader else None
            })
        
        self.stage_1_complete = True
        self.logger.info("Stage 1 training completed!")
        
        # Save metadata encoder
        torch.save(self.metadata_encoder.state_dict(), 'metadata_encoder.pth')
        
    def train_stage_2(self, train_texts: List[str], train_labels: List[int],
                     val_texts: List[str] = None, val_labels: List[int] = None):
        """
        Stage 2: Train semantic layers with frozen metadata encoder
        
        Like applying grammar rules while reading and understanding complex texts.
        """
        
        if not self.stage_1_complete:
            raise ValueError("Must complete Stage 1 training before Stage 2!")
        
        self.logger.info("Starting Stage 2: Semantic Fine-tuning")
        
        # Create datasets with both metadata and semantic features
        train_dataset = TextDataset(
            train_texts, train_labels, 
            self.metadata_extractor, self.semantic_processor
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['stage_2']['batch_size'],
            shuffle=True
        )
        
        val_loader = None
        if val_texts and val_labels:
            val_dataset = TextDataset(
                val_texts, val_labels,
                self.metadata_extractor, self.semantic_processor
            )
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Freeze metadata encoder if specified
        if self.config['stage_2']['freeze_metadata']:
            for param in self.metadata_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Metadata encoder frozen")
        
        # Setup optimizer for semantic components only
        trainable_params = [p for p in self.full_model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params,
            lr=self.config['stage_2']['lr'],
            weight_decay=self.config['stage_2']['weight_decay']
        )
        
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        
        # Training loop
        for epoch in range(self.config['stage_2']['epochs']):
            self.full_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Prepare inputs
                metadata_features = torch.stack([
                    torch.tensor(meta['length_features'], dtype=torch.float32)
                    for meta in batch['metadata']
                ]).to(self.device)
                
                semantic_features = torch.stack([
                    torch.tensor(sem['embeddings'], dtype=torch.float32)
                    for sem in batch['semantic']
                ]).to(self.device)
                
                labels = torch.tensor(batch['label']).to(self.device)
                
                # Forward pass
                outputs = self.full_model(metadata_features, semantic_features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self._validate_stage_2(val_loader, criterion)
                
                self.logger.info(f"Epoch {epoch+1}/{self.config['stage_2']['epochs']}: "
                               f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.full_model.state_dict(), 'best_model.pth')
                    self.logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.config['stage_2']['epochs']}: "
                               f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Save training history
            self.training_history['stage_2'].append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_acc': train_acc,
                'val_loss': val_loss if val_loader else None,
                'val_acc': val_acc if val_loader else None
            })
        
        self.logger.info("Stage 2 training completed!")
        
        # Save final model and training history
        torch.save(self.full_model.state_dict(), 'final_model.pth')
        with open('training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _validate_stage_1(self, val_loader: DataLoader, criterion) -> float:
        """Validation for Stage 1"""
        self.metadata_encoder.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                metadata_features = torch.stack([
                    torch.tensor(meta['length_features'], dtype=torch.float32)
                    for meta in batch['metadata']
                ]).to(self.device)
                
                targets = torch.tensor([
                    len(text.split()) for text in batch['text']
                ], dtype=torch.float32).to(self.device)
                
                encoded = self.metadata_encoder(metadata_features)
                predictions = torch.sum(encoded, dim=1)
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_stage_2(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validation for Stage 2"""
        self.full_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                metadata_features = torch.stack([
                    torch.tensor(meta['length_features'], dtype=torch.float32)
                    for meta in batch['metadata']
                ]).to(self.device)
                
                semantic_features = torch.stack([
                    torch.tensor(sem['embeddings'], dtype=torch.float32)
                    for sem in batch['semantic']
                ]).to(self.device)
                
                labels = torch.tensor(batch['label']).to(self.device)
                
                outputs = self.full_model(metadata_features, semantic_features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path: str = "two_stage_model.pth"):
        """Save the complete trained model"""
        if self.full_model is None:
            raise ValueError("No model to save!")
        
        checkpoint = {
            'full_model_state': self.full_model.state_dict(),
            'metadata_encoder_state': self.metadata_encoder.state_dict(),
            'semantic_encoder_state': self.semantic_encoder.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        
        # Rebuild models
        # Note: You'll need to provide dimensions or save them in checkpoint
        # For now, using default values
        self.build_models(metadata_feature_dim=10, semantic_feature_dim=768)
        
        self.metadata_encoder.load_state_dict(checkpoint['metadata_encoder_state'])
        self.semantic_encoder.load_state_dict(checkpoint['semantic_encoder_state'])
        self.full_model.load_state_dict(checkpoint['full_model_state'])
        
        self.stage_1_complete = True
        self.logger.info(f"Model loaded from {path}")
    
    def predict(self, texts: List[str]) -> List[int]:
        """Make predictions on new texts"""
        if self.full_model is None:
            raise ValueError("No trained model available!")
        
        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = TextDataset(
            texts, dummy_labels,
            self.metadata_extractor, self.semantic_processor
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.full_model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                metadata_features = torch.stack([
                    torch.tensor(meta['length_features'], dtype=torch.float32)
                    for meta in batch['metadata']
                ]).to(self.device)
                
                semantic_features = torch.stack([
                    torch.tensor(sem['embeddings'], dtype=torch.float32)
                    for sem in batch['semantic']
                ]).to(self.device)
                
                outputs = self.full_model(metadata_features, semantic_features)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy().tolist())
        
        return predictions


if __name__ == "__main__":
    # Example usage
    from metadata_extractor import MetadataExtractor
    from semantic_processor import SemanticProcessor
    
    # Initialize components
    metadata_extractor = MetadataExtractor()
    semantic_processor = SemanticProcessor()
    
    # Create trainer
    trainer = TwoStageTrainer(
        metadata_extractor=metadata_extractor,
        semantic_processor=semantic_processor,
        config_path='config.yaml'
    )
    
    # Example data (replace with your actual data)
    train_texts = ["This is a positive example.", "This is negative."] * 100
    train_labels = [1, 0] * 100
    
    val_texts = ["Validation positive.", "Validation negative."] * 20
    val_labels = [1, 0] * 20
    
    # Train the model
    print("Training two-stage model...")
    trainer.train_stage_1(train_texts, train_labels, val_texts, val_labels)
    trainer.train_stage_2(train_texts, train_labels, val_texts, val_labels)
    
    # Save the trained model
    trainer.save_model("trained_two_stage_model.pth")
    print("Training completed!")

