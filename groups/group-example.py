import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Callable, Optional

class GroupedLinear(nn.Module):
    """
    A linear layer where nodes are organized into groups with different
    backpropagation behavior for each group.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 group_assignments: Dict[str, List[int]],
                 group_rules: Dict[str, Callable] = None):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            group_assignments: Dict mapping group names to lists of output node indices
                             e.g., {'fast_learners': [0, 1, 2], 'slow_learners': [3, 4]}
            group_rules: Dict mapping group names to gradient transformation functions
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.group_assignments = group_assignments
        
        # Standard linear layer components
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Default group rules if none provided
        self.group_rules = group_rules or {}
        
        # Create masks for each group for efficient computation
        self.group_masks = {}
        for group_name, indices in group_assignments.items():
            mask = torch.zeros(out_features, dtype=torch.bool)
            mask[indices] = True
            self.register_buffer(f'mask_{group_name}', mask)
            self.group_masks[group_name] = mask
    
    def forward(self, x):
        # Standard forward pass
        output = F.linear(x, self.weight, self.bias)
        
        # Store input for potential use in backward pass
        self.last_input = x
        
        return output
    
    def apply_group_gradients(self):
        """Apply group-specific gradient modifications"""
        if not hasattr(self.weight, 'grad') or self.weight.grad is None:
            return
            
        with torch.no_grad():
            for group_name, rule_func in self.group_rules.items():
                if group_name in self.group_masks:
                    mask = self.group_masks[group_name]
                    
                    # Apply rule to weight gradients for this group
                    if self.weight.grad is not None:
                        group_grad = self.weight.grad[mask]
                        modified_grad = rule_func(group_grad, group_name)
                        self.weight.grad[mask] = modified_grad
                    
                    # Apply rule to bias gradients for this group
                    if self.bias.grad is not None:
                        group_bias_grad = self.bias.grad[mask]
                        modified_bias_grad = rule_func(group_bias_grad, group_name)
                        self.bias.grad[mask] = modified_bias_grad

# Example gradient modification functions
def slow_learning_rule(grad, group_name):
    """Reduce gradient magnitude by 50%"""
    return grad * 0.5

def fast_learning_rule(grad, group_name):
    """Increase gradient magnitude by 2x"""
    return grad * 2.0

def sparse_learning_rule(grad, group_name):
    """Only keep top 20% of gradients by magnitude"""
    flat_grad = grad.flatten()
    threshold = torch.quantile(torch.abs(flat_grad), 0.8)
    mask = torch.abs(grad) >= threshold
    return grad * mask.float()

def adaptive_learning_rule(grad, group_name):
    """Normalize gradients and apply group-specific scaling"""
    norm = torch.norm(grad)
    if norm > 0:
        normalized = grad / norm
        # Different scaling based on group
        if 'critical' in group_name:
            return normalized * 0.1  # Very conservative updates
        elif 'exploration' in group_name:
            return normalized * 1.5  # Aggressive updates
    return grad

# Custom backward hook for more complex group behavior
class GroupedBackwardHook:
    def __init__(self, layer, group_schedules=None):
        self.layer = layer
        self.group_schedules = group_schedules or {}
        self.step_count = 0
    
    def __call__(self, module, grad_input, grad_output):
        # Apply scheduled group rules based on training step
        for group_name, schedule in self.group_schedules.items():
            if self.step_count in schedule:
                new_rule = schedule[self.step_count]
                self.layer.group_rules[group_name] = new_rule
        
        self.step_count += 1
        return grad_input

# Example usage and training loop
class GroupedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define groups - imagine we have 10 output nodes
        groups = {
            'fast_learners': [0, 1, 2],      # First 3 nodes learn quickly
            'slow_learners': [3, 4, 5],     # Middle 3 nodes learn slowly  
            'sparse_learners': [6, 7],      # Next 2 use sparse updates
            'adaptive_learners': [8, 9]     # Last 2 use adaptive scaling
        }
        
        # Define rules for each group
        rules = {
            'fast_learners': fast_learning_rule,
            'slow_learners': slow_learning_rule,
            'sparse_learners': sparse_learning_rule,
            'adaptive_learners': adaptive_learning_rule
        }
        
        # Create the grouped layer
        self.grouped_layer = GroupedLinear(
            in_features=20, 
            out_features=10,
            group_assignments=groups,
            group_rules=rules
        )
        
        # Add more standard layers
        self.output_layer = nn.Linear(10, 1)
        
        # Register backward hook for advanced scheduling
        self.hook = self.grouped_layer.register_backward_hook(
            GroupedBackwardHook(self.grouped_layer)
        )
    
    def forward(self, x):
        x = torch.relu(self.grouped_layer(x))
        return self.output_layer(x)

# Training example
def train_step():
    model = GroupedNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Sample data
    batch_size, input_dim = 32, 20
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    
    # Forward pass
    output = model(x)
    loss = F.mse_loss(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Apply group-specific gradient modifications
    model.grouped_layer.apply_group_gradients()
    
    # Update weights
    optimizer.step()
    
    return loss.item()

# Example of dynamic group rule changes
def create_scheduled_rules():
    """Create rules that change during training"""
    return {
        'adaptive_learners': {
            0: adaptive_learning_rule,           # Start with adaptive
            100: slow_learning_rule,             # Switch to slow after 100 steps
            500: lambda g, n: g * 0.1           # Very conservative after 500 steps
        }
    }

if __name__ == "__main__":
    # Quick test
    print("Testing grouped layer...")
    loss = train_step()
    print(f"Loss: {loss:.4f}")
    
    # Show group assignments
    model = GroupedNetwork()
    print("\nGroup assignments:")
    for group, indices in model.grouped_layer.group_assignments.items():
        print(f"  {group}: nodes {indices}")

