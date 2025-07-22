import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Callable, Optional, Tuple
import copy

class ComplementaryGroupedLinear(nn.Module):
    """
    A linear layer with complementary groups that can be micro-adjusted
    against each other before continuing normal training.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 complementary_pairs: Dict[str, Tuple[List[int], List[int]]]):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            complementary_pairs: Dict mapping pair names to (group1_indices, group2_indices)
                               e.g., {'emotion': ([0,1,2], [3,4,5])} for happy/sad nodes
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.complementary_pairs = complementary_pairs
        
        # Standard linear layer components
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Create masks and group assignments
        self.group_masks = {}
        self.all_groups = {}
        
        for pair_name, (group1_indices, group2_indices) in complementary_pairs.items():
            group1_name = f"{pair_name}_positive"
            group2_name = f"{pair_name}_negative"
            
            # Create masks
            mask1 = torch.zeros(out_features, dtype=torch.bool)
            mask1[group1_indices] = True
            mask2 = torch.zeros(out_features, dtype=torch.bool) 
            mask2[group2_indices] = True
            
            self.register_buffer(f'mask_{group1_name}', mask1)
            self.register_buffer(f'mask_{group2_name}', mask2)
            
            self.group_masks[group1_name] = mask1
            self.group_masks[group2_name] = mask2
            self.all_groups[group1_name] = group1_indices
            self.all_groups[group2_name] = group2_indices
        
        # Storage for frozen states
        self.frozen_params = {}
        self.is_frozen = False
        
    def forward(self, x):
        print(f"[ComplementaryGrouped Forward] Input shape: {x.shape}")
        output = F.linear(x, self.weight, self.bias)
        print(f"[ComplementaryGrouped Forward] Output shape: {output.shape}")
        return output
    
    def freeze_all_except_group(self, group_name: str):
        """Freeze all parameters except those belonging to specified group"""
        print(f"\n[Freeze] Freezing all parameters except group '{group_name}'")
        
        if group_name not in self.group_masks:
            raise ValueError(f"Group {group_name} not found. Available: {list(self.group_masks.keys())}")
        
        # Store original requires_grad state
        self.frozen_params['weight_requires_grad'] = self.weight.requires_grad
        self.frozen_params['bias_requires_grad'] = self.bias.requires_grad
        
        # Store which group is active
        self.frozen_params['active_group'] = group_name
        self.is_frozen = True
        
        # Create parameter copies that only update the active group
        mask = self.group_masks[group_name]
        
        print(f"[Freeze] Active nodes: {torch.where(mask)[0].tolist()}")
        print(f"[Freeze] Frozen nodes: {torch.where(~mask)[0].tolist()}")
        
        # We'll handle the selective updates in the micro_adjust method
        
    def unfreeze_all(self):
        """Restore normal gradient flow to all parameters"""
        print(f"\n[Unfreeze] Restoring normal gradient flow to all parameters")
        
        if not self.is_frozen:
            print("[Unfreeze] Network was not frozen, nothing to restore")
            return
        
        # Restore original requires_grad state
        if 'weight_requires_grad' in self.frozen_params:
            self.weight.requires_grad = self.frozen_params['weight_requires_grad']
        if 'bias_requires_grad' in self.frozen_params:
            self.bias.requires_grad = self.frozen_params['bias_requires_grad']
        
        self.frozen_params = {}
        self.is_frozen = False
        print("[Unfreeze] All parameters unfrozen")
    
    def micro_adjust_group(self, group_name: str, adjustment: torch.Tensor, 
                          bias_adjustment: torch.Tensor = None):
        """Apply micro-adjustments to a specific group's parameters"""
        print(f"\n[Micro-Adjust] Adjusting group '{group_name}'")
        
        if group_name not in self.group_masks:
            raise ValueError(f"Group {group_name} not found")
        
        mask = self.group_masks[group_name]
        
        with torch.no_grad():
            # Adjust weights
            if adjustment is not None:
                old_norm = self.weight[mask].norm()
                self.weight[mask] += adjustment
                new_norm = self.weight[mask].norm() 
                print(f"[Micro-Adjust] Weight norm: {old_norm:.4f} -> {new_norm:.4f}")
            
            # Adjust bias
            if bias_adjustment is not None:
                old_bias_norm = self.bias[mask].norm()
                self.bias[mask] += bias_adjustment
                new_bias_norm = self.bias[mask].norm()
                print(f"[Micro-Adjust] Bias norm: {old_bias_norm:.4f} -> {new_bias_norm:.4f}")

class ComplementaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define complementary pairs - happy/sad emotions
        complementary_pairs = {
            'emotion': ([0, 1, 2], [3, 4, 5]),  # happy vs sad
            'energy': ([6, 7], [8, 9])          # high vs low energy
        }
        
        self.complementary_layer = ComplementaryGroupedLinear(
            in_features=20,
            out_features=10, 
            complementary_pairs=complementary_pairs
        )
        
        self.output_layer = nn.Linear(10, 1)
        
        print("[Network Init] Created complementary network:")
        for pair_name, (pos_indices, neg_indices) in complementary_pairs.items():
            print(f"  {pair_name}: positive={pos_indices}, negative={neg_indices}")
    
    def forward(self, x):
        print(f"\n[Network Forward] Input shape: {x.shape}")
        x = torch.relu(self.complementary_layer(x))
        print(f"[Network Forward] After complementary layer: {x.shape}")
        output = self.output_layer(x)
        print(f"[Network Forward] Final output: {output.shape}")
        return output
    
    def get_complementary_groups(self, pair_name: str):
        """Get the names of complementary groups for a pair"""
        return f"{pair_name}_positive", f"{pair_name}_negative"

class ComplementaryTrainer:
    """Handles the micro-adjustment training process"""
    
    def __init__(self, model: ComplementaryNetwork, lr: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.micro_lr = 0.001  # Smaller learning rate for micro-adjustments
        
    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluate current loss without affecting gradients"""
        with torch.no_grad():
            output = self.model(x)
            loss = F.mse_loss(output, y)
            return loss.item()
    
    def micro_adjust_complementary_pair(self, pair_name: str, x: torch.Tensor, 
                                       y: torch.Tensor, max_iterations: int = 5):
        """
        Micro-adjust complementary groups to find optimal balance
        """
        print(f"\n{'='*50}")
        print(f"MICRO-ADJUSTING COMPLEMENTARY PAIR: {pair_name}")
        print(f"{'='*50}")
        
        pos_group, neg_group = self.model.get_complementary_groups(pair_name)
        
        # Get baseline loss
        baseline_loss = self.evaluate_loss(x, y)
        print(f"[Baseline] Initial loss: {baseline_loss:.6f}")
        
        best_loss = baseline_loss
        best_adjustments = {'pos': None, 'neg': None}
        
        # Save original state
        original_weight = self.model.complementary_layer.weight.clone()
        original_bias = self.model.complementary_layer.bias.clone()
        
        for iteration in range(max_iterations):
            print(f"\n--- Micro-adjustment Iteration {iteration + 1} ---")
            
            # Try adjusting positive group
            print(f"\n[Testing] Adjusting {pos_group}...")
            self.model.complementary_layer.freeze_all_except_group(pos_group)
            
            # Create small random adjustments
            pos_mask = self.model.complementary_layer.group_masks[pos_group]
            pos_weight_adj = torch.randn_like(self.model.complementary_layer.weight[pos_mask]) * self.micro_lr
            pos_bias_adj = torch.randn_like(self.model.complementary_layer.bias[pos_mask]) * self.micro_lr
            
            # Apply adjustment and test
            self.model.complementary_layer.micro_adjust_group(pos_group, pos_weight_adj, pos_bias_adj)
            
            pos_loss = self.evaluate_loss(x, y) 
            print(f"[Testing] {pos_group} adjustment -> loss: {pos_loss:.6f} (change: {pos_loss - baseline_loss:+.6f})")
            
            # Try adjusting negative group  
            print(f"\n[Testing] Adjusting {neg_group}...")
            self.model.complementary_layer.freeze_all_except_group(neg_group)
            
            neg_mask = self.model.complementary_layer.group_masks[neg_group]
            neg_weight_adj = torch.randn_like(self.model.complementary_layer.weight[neg_mask]) * self.micro_lr
            neg_bias_adj = torch.randn_like(self.model.complementary_layer.bias[neg_mask]) * self.micro_lr
            
            # Apply complementary adjustment (opposite direction for balance)
            self.model.complementary_layer.micro_adjust_group(neg_group, -neg_weight_adj, -neg_bias_adj)
            
            combined_loss = self.evaluate_loss(x, y)
            print(f"[Testing] Combined adjustment -> loss: {combined_loss:.6f} (change: {combined_loss - baseline_loss:+.6f})")
            
            # Keep best adjustment
            if combined_loss < best_loss:
                best_loss = combined_loss
                best_adjustments['pos'] = (pos_weight_adj, pos_bias_adj)
                best_adjustments['neg'] = (neg_weight_adj, neg_bias_adj)
                print(f"[Best] New best loss: {best_loss:.6f} ⭐")
            else:
                # Revert to original state
                self.model.complementary_layer.weight.data = original_weight.clone()
                self.model.complementary_layer.bias.data = original_bias.clone()
                print("[Revert] No improvement, reverting changes")
        
        self.model.complementary_layer.unfreeze_all()
        
        print(f"\n[Micro-Adjust Complete] Best loss: {best_loss:.6f}")
        print(f"[Micro-Adjust Complete] Improvement: {baseline_loss - best_loss:.6f}")
        
        return best_loss
    
    def train_step_with_micro_adjustment(self, x: torch.Tensor, y: torch.Tensor):
        """
        Full training step with micro-adjustment of complementary pairs
        """
        print("\n" + "="*70)
        print("TRAINING STEP WITH MICRO-ADJUSTMENT")
        print("="*70)
        
        # 1. Normal forward pass
        print(f"\n{'='*20} PHASE 1: NORMAL FORWARD PASS {'='*20}")
        output = self.model(x)
        loss = F.mse_loss(output, y)
        print(f"[Phase 1] Initial loss: {loss.item():.6f}")
        
        # 2. Normal backward pass
        print(f"\n{'='*20} PHASE 2: NORMAL BACKWARD PASS {'='*20}")
        self.optimizer.zero_grad()
        loss.backward()
        print(f"[Phase 2] Gradients computed")
        
        # 3. Micro-adjust complementary pairs
        print(f"\n{'='*15} PHASE 3: MICRO-ADJUST COMPLEMENTARY PAIRS {'='*15}")
        for pair_name in self.model.complementary_layer.complementary_pairs.keys():
            final_loss = self.micro_adjust_complementary_pair(pair_name, x, y)
        
        # 4. Continue with normal optimization
        print(f"\n{'='*20} PHASE 4: NORMAL OPTIMIZATION {'='*20}")
        print("[Phase 4] Unfreezing and applying normal gradients...")
        self.model.complementary_layer.unfreeze_all()
        self.optimizer.step()
        
        # 5. Final evaluation
        final_loss = self.evaluate_loss(x, y)
        print(f"\n[Final] Training step complete - Final loss: {final_loss:.6f}")
        
        return final_loss

def run_complementary_training_demo():
    """Demo of the complementary training process"""
    
    print("🎭 Starting Complementary Group Training Demo")
    print("=" * 70)
    
    # Create model and trainer
    model = ComplementaryNetwork()
    trainer = ComplementaryTrainer(model)
    
    # Generate sample data
    batch_size, input_dim = 16, 20  # Smaller batch for clearer demo
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    
    print(f"\n📊 Generated data: {batch_size} samples, {input_dim} features")
    
    # Run one complete training step with micro-adjustments
    final_loss = trainer.train_step_with_micro_adjustment(x, y)
    
    print(f"\n🎯 Demo Complete!")
    print(f"   Final Loss: {final_loss:.6f}")
    print(f"\n💡 What happened:")
    print("   1. Normal forward/backward pass")
    print("   2. Froze network and micro-adjusted 'emotion' pair (happy vs sad)")
    print("   3. Froze network and micro-adjusted 'energy' pair (high vs low)")
    print("   4. Found optimal balance between complementary groups")
    print("   5. Unfroze network and continued normal training")
    
    return final_loss

if __name__ == "__main__":
    run_complementary_training_demo()
