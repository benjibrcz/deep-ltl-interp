#!/usr/bin/env python3
"""
Auxiliary Chained Distance Prediction

This module implements an auxiliary prediction head that forces the network
to compute chained distances (d(agent→intermediate) + d(intermediate→goal)).

The idea: if we add a supervised loss that requires predicting chained distances,
the network must develop internal representations that encode this computation.

Implementation approach:
1. Add a prediction head to the model
2. During rollouts, collect ground-truth chained distances
3. Add auxiliary loss to the PPO training objective:
   total_loss = policy_loss + value_loss + lambda * aux_loss

Usage:
    # During model construction:
    model = build_model_with_aux_head(...)

    # During training:
    aux_predictions = model.predict_chained_distances(obs)
    aux_loss = F.mse_loss(aux_predictions, true_chained_distances)
    total_loss = ppo_loss + aux_lambda * aux_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChainedDistanceHead(nn.Module):
    """
    Auxiliary head that predicts chained distances.

    Given the current observation and the intermediate goal,
    predicts d(agent→intermediate) + d(intermediate→goal).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict single scalar: total path length
        )

    def forward(self, combined_embedding):
        """
        Args:
            combined_embedding: The combined env+ltl embedding from the main model

        Returns:
            Predicted chained distance (batch_size, 1)
        """
        return self.net(combined_embedding)


def compute_chained_distances(env_info: dict, intermediate_color: str, final_color: str) -> float:
    """
    Compute ground-truth chained distance from environment info.

    Args:
        env_info: Dictionary with 'agent_pos' and 'zone_positions'
        intermediate_color: Color of intermediate goal (e.g., 'blue')
        final_color: Color of final goal (e.g., 'green')

    Returns:
        Total distance: d(agent→nearest_intermediate) + d(intermediate→nearest_final)
    """
    agent_pos = np.array(env_info['agent_pos'][:2])

    # Find nearest intermediate zone
    intermediate_zones = [
        pos for name, pos in env_info['zone_positions'].items()
        if name.startswith(intermediate_color)
    ]
    d_to_intermediates = [np.linalg.norm(agent_pos - z) for z in intermediate_zones]
    nearest_intermediate_idx = np.argmin(d_to_intermediates)
    intermediate_pos = intermediate_zones[nearest_intermediate_idx]

    # Find distance from that intermediate to nearest final
    final_zones = [
        pos for name, pos in env_info['zone_positions'].items()
        if name.startswith(final_color)
    ]
    d_intermediate_to_finals = [np.linalg.norm(intermediate_pos - z) for z in final_zones]
    min_d_to_final = min(d_intermediate_to_finals)

    # Return chained distance
    return d_to_intermediates[nearest_intermediate_idx] + min_d_to_final


def compute_optimal_chained_distance(env_info: dict, intermediate_color: str, final_color: str) -> float:
    """
    Compute optimal chained distance (choosing best intermediate).

    This is what the agent SHOULD predict if it's planning correctly.
    """
    agent_pos = np.array(env_info['agent_pos'][:2])

    intermediate_zones = [
        pos for name, pos in env_info['zone_positions'].items()
        if name.startswith(intermediate_color)
    ]

    final_zones = [
        pos for name, pos in env_info['zone_positions'].items()
        if name.startswith(final_color)
    ]

    # Compute total path for each intermediate
    total_paths = []
    for inter_pos in intermediate_zones:
        d_to_inter = np.linalg.norm(agent_pos - inter_pos)
        d_inter_to_finals = [np.linalg.norm(inter_pos - f) for f in final_zones]
        total = d_to_inter + min(d_inter_to_finals)
        total_paths.append(total)

    return min(total_paths)


class AuxiliaryDistanceLoss:
    """
    Computes auxiliary loss for chained distance prediction.

    Usage:
        aux_loss_fn = AuxiliaryDistanceLoss(lambda_coef=0.1)

        # During training:
        predictions = model.aux_head(combined_embedding)
        aux_loss = aux_loss_fn(predictions, true_distances)
        total_loss = ppo_loss + aux_loss
    """

    def __init__(self, lambda_coef: float = 0.1):
        self.lambda_coef = lambda_coef

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size, 1)
            targets: True chained distances (batch_size, 1)

        Returns:
            Weighted MSE loss
        """
        return self.lambda_coef * F.mse_loss(predictions, targets)


# Example integration with existing model
"""
# In model/model.py:

class ActorCriticWithAux(ActorCritic):
    def __init__(self, ...):
        super().__init__(...)
        # Add auxiliary head
        combined_dim = env_dim + ltl_dim  # Size of combined embedding
        self.aux_head = ChainedDistanceHead(combined_dim)

    def forward_with_aux(self, obs):
        # Compute standard forward pass
        dist, value = self.forward(obs)

        # Also compute aux prediction
        env_embedding = self.env_net(obs.features)
        ltl_embedding = self.ltl_net(obs.seq)
        combined = torch.cat([env_embedding, ltl_embedding], dim=1)
        aux_pred = self.aux_head(combined)

        return dist, value, aux_pred


# In torch_ac/ppo.py:

def update_parameters(self, exps):
    # Standard PPO update...

    # Add auxiliary loss
    if hasattr(self.model, 'aux_head'):
        aux_pred = self.model.aux_head(combined_embedding)
        aux_loss = self.aux_loss_fn(aux_pred, exps.chained_distances)
        total_loss = policy_loss + value_loss + aux_loss

    # Backprop...
"""


if __name__ == '__main__':
    # Test the chained distance computation
    env_info = {
        'agent_pos': np.array([0.0, 0.0, 0.0]),
        'zone_positions': {
            'blue_zone0': np.array([1.0, 0.0]),
            'blue_zone1': np.array([-1.0, 0.0]),
            'green_zone0': np.array([-2.0, 0.0]),
            'green_zone1': np.array([2.0, 0.0]),
        }
    }

    # Agent at origin
    # Blue1 at (1,0) distance 1.0, Green0 at (-2,0) so blue1→green0 = 3.0, total = 4.0
    # Blue2 at (-1,0) distance 1.0, Green0 at (-2,0) so blue2→green0 = 1.0, total = 2.0

    optimal = compute_optimal_chained_distance(env_info, 'blue', 'green')
    print(f"Optimal chained distance: {optimal:.2f}")  # Should be 2.0

    nearest = compute_chained_distances(env_info, 'blue', 'green')
    print(f"Nearest intermediate chained distance: {nearest:.2f}")  # Depends on which is "nearest"
