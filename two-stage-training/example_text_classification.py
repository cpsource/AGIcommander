#!/usr/bin/env python3
"""
Example: Text Classification with Two-Stage Training

This example demonstrates how to use the two-stage training approach for text 
classification. Like learning to read by first understanding grammar rules,
then applying them to understand stories.
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import time

# Import our modules
from metadata_extractor import MetadataExtractor
from semantic_processor import SemanticProcessor
from two_stage_trainer import TwoStageTrainer
import yaml


def load_sample_data() -> Tuple[List[str], List[int]]:
    """
    Load sample text classification data.
    In practice, replace this with your actual dataset.
    """
    
    # Sample positive examples (label 1)
    positive_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the best thing I've ever bought. Highly recommended!",
        "Excellent quality and fast delivery. Very satisfied with my purchase.",
        "Outstanding performance and great value for money. Five stars!",
        "Perfect solution to my problem. Couldn't be happier with the results.",
        "Wonderful experience from start to finish. Will definitely buy again.",
        "Impressive features and user-friendly design. Love it!",
        "Fantastic product that exceeded all my expectations.",
        "Great customer service and high-quality product. Very pleased.",
        "This has made my life so much easier. Brilliant innovation!"
    ] * 20  # Repeat to get more samples
    
    # Sample negative examples (label 0)
    negative_texts = [
        "This product is terrible. Complete waste of money.",
        "Poor quality and doesn't work as advertised. Very disappointed.",
        "Worst purchase I've ever made. Avoid at all costs.",
        "Cheap materials and bad design. Fell apart immediately.",
        "Horrible customer service and defective product. One star.",
        "Don't buy this. It's a scam and doesn't deliver what it promises.",
        "Useless product that broke after one day. Total garbage.",
        "Overpriced and underperforms. Not worth the money.",
        "Frustrating experience with poor build quality.",
        "Regret buying this. Save your money and look elsewhere."
    ] * 20  # Repeat to get more samples
    
    # Combine texts and labels
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels


def create_config() -> dict:
    """Create configuration for training"""
    
    config = {
        'stage_1': {
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.001,
            'weight_decay': 1e-5
        },
        'stage_2': {
            'epochs': 15,
            'batch_size': 16,
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


def save_config(config: dict, path: str = "config.yaml"):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {path}")


def plot_training_history(training_history: dict, save_path: str = "training_plots.png"):
    """Plot training history for both stages"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Two-Stage Training Progress', fontsize=16)
    
    # Stage 1 plots
    stage_1_data = training_history['stage_1']
    if stage_1_data:
        epochs_1 = [entry['epoch'] for entry in stage_1_data]
        train_loss_1 = [entry['train_loss'] for entry in stage_1_data]
        val_loss_1 = [entry['val_loss'] for entry in stage_1_data if entry['val_loss'] is not None]
        
        axes[0, 0].plot(epochs_1, train_loss_1, 'b-', label='Train Loss')
        if val_loss_1:
            axes[0, 0].plot(epochs_1[:len(val_loss_1)], val_loss_1, 'r-', label='Val Loss')
        axes[0, 0].set_title('Stage 1: Metadata Pre-training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Stage 2 plots
    stage_2_data = training_history['stage_2']
    if stage_2_data:
        epochs_2 = [entry['epoch'] for entry in stage_2_data]
        train_loss_2 = [entry['train_loss'] for entry in stage_2_data]
        train_acc_2 = [entry['train_acc'] for entry in stage_2_data]
        val_loss_2 = [entry['val_loss'] for entry in stage_2_data if entry['val_loss'] is not None]
        val_acc_2 = [entry['val_acc'] for entry in stage_2_data if entry['val_acc'] is not None]
        
        # Loss plot
        axes[0, 1].plot(epochs_2, train_loss_2, 'b-', label='Train Loss')
        if val_loss_2:
            axes[0, 1].plot(epochs_2[:len(val_loss_2)], val_loss_2, 'r-', label='Val Loss')
        axes[0, 1].set_title('Stage 2: Semantic Training Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy plot
        axes[1, 0].plot(epochs_2, train_acc_2, 'b-', label='Train Accuracy')
        if val_acc_2:
            axes[1, 0].plot(epochs_2[:len(val_acc_2)], val_acc_2, 'r-', label='Val Accuracy')
        axes[1, 0].set_title('Stage 2: Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Summary plot - combine both stages
    if stage_1_data and stage_2_data:
        all_epochs = list(range(1, len(stage_1_data) + len(stage_2_data) + 1))
        all_train_losses = train_loss_1 + train_loss_2
        
        axes[1, 1].plot(all_epochs[:len(stage_1_data)], train_loss_1, 'g-', 
                       linewidth=2, label='Stage 1 (Metadata)')
        axes[1, 1].plot(all_epochs[len(stage_1_data):len(stage_1_data)+len(stage_2_data)], 
                       train_loss_2, 'b-', linewidth=2, label='Stage 2 (Semantic)')
        axes[1, 1].axvline(x=len(stage_1_data), color='red', linestyle='--', 
                          alpha=0.7, label='Stage Transition')
        axes[1, 1].set_title('Complete Training Progress')
        axes[1, 1].set_xlabel('Total Epochs')
        axes[1, 1].set_ylabel('Train Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training plots saved to {save_path}")


def evaluate_model(trainer: TwoStageTrainer, test_texts: List[str], 
                  test_labels: List[int]) -> dict:
    """Evaluate the trained model on test data"""
    
    print("Evaluating model on test data...")
    
    # Make predictions
    start_time = time.time()
    predictions = trainer.predict(test_texts)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Average time per sample: {inference_time/len(test_texts)*1000:.2f} ms")
    
    # Detailed classification report
    report = classification_report(test_labels, predictions, 
                                 target_names=['Negative', 'Positive'])
    print(f"\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'inference_time': inference_time,
        'predictions': predictions,
        'classification_report': report
    }


def main():
    """Main execution function"""
    
    print("Two-Stage Text Classification Example")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading sample data...")
    texts, labels = load_sample_data()
    print(f"Loaded {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"\nData split:")
    print(f"Train: {len(train_texts)} samples")
    print(f"Validation: {len(val_texts)} samples")
    print(f"Test: {len(test_texts)} samples")
    
    # Create configuration
    config = create_config()
    save_config(config)
    
    # Initialize components
    print("\nInitializing components...")
    metadata_extractor = MetadataExtractor()
    semantic_processor = SemanticProcessor()
    
    # Fit TF-IDF on training data
    semantic_processor.fit_tfidf(train_texts)
    
    # Create trainer
    trainer = TwoStageTrainer(
        metadata_extractor=metadata_extractor,
        semantic_processor=semantic_processor,
        config_path='config.yaml'
    )
    
    # Training Stage 1
    print("\n" + "="*50)
    print("STAGE 1: METADATA PRE-TRAINING")
    print("="*50)
    start_time = time.time()
    
    trainer.train_stage_1(train_texts, train_labels, val_texts, val_labels)
    
    stage_1_time = time.time() - start_time
    print(f"Stage 1 completed in {stage_1_time:.2f} seconds")
    
    # Training Stage 2
    print("\n" + "="*50)
    print("STAGE 2: SEMANTIC FINE-TUNING")
    print("="*50)
    start_time = time.time()
    
    trainer.train_stage_2(train_texts, train_labels, val_texts, val_labels)
    
    stage_2_time = time.time() - start_time
    total_training_time = stage_1_time + stage_2_time
    print(f"Stage 2 completed in {stage_2_time:.2f} seconds")
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    # Save the trained model
    trainer.save_model("text_classification_model.pth")
    
    # Plot training history
    plot_training_history(trainer.training_history)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    results = evaluate_model(trainer, test_texts, test_labels)
    
    # Print summary
    print(f"\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Stage 1 time: {stage_1_time:.2f}s")
    print(f"Stage 2 time: {stage_2_time:.2f}s")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Inference time: {results['inference_time']:.2f}s")
    
    # Example predictions
    print(f"\nExample Predictions:")
    print("-" * 30)
    example_texts = [
        "This product is absolutely fantastic!",
        "Terrible quality, waste of money.",
        "Average product, nothing special.",
        "Love it! Highly recommended!"
    ]
    
    example_predictions = trainer.predict(example_texts)
    for text, pred in zip(example_texts, example_predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}")
        print(f"Prediction: {sentiment}\n")
    
    print("Example completed successfully!")
    print("Check the generated files:")
    print("- config.yaml: Training configuration")
    print("- training_plots.png: Training progress visualization")
    print("- confusion_matrix.png: Model performance matrix")
    print("- text_classification_model.pth: Trained model")
    print("- training_*.log: Training logs")


if __name__ == "__main__":
    main()

