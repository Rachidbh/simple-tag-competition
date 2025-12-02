"""
Template for student agent submission.

Students should implement the StudentAgent class with both prey and predator behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    """
    Template agent class for Simple Tag competition.
    
    Students must implement this class with their own agent logic.
    The agent should handle both "prey" and "predator" types.
    """
    
    def __init__(self, agent_type: str):
        """
        Initialize your agent.
        
        Args:
            agent_type (str): Either "prey" or "predator"
        """
        self.agent_type = agent_type
        
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        
        if agent_type == "prey":
            # Example: Load prey model
            # model_path = self.submission_dir / "prey_model.pth"
            # self.model = self.load_model(model_path)
            pass
        elif agent_type == "predator":
            # Example: Load predator model
            # model_path = self.submission_dir / "predator_model.pth"
            # self.model = self.load_model(model_path)
            pass
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment (numpy array)
                         - Prey (good agent): shape (16,)
                         - Predator (adversary): shape (14,)
            agent_id (str): Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right  
                    3 = move down
                    4 = move up
        """
        # IMPLEMENT YOUR POLICY HERE
        
        # Example random policy (replace with your trained policy):
        # Action space is Discrete(5) by default
        # Note: During evaluation, RNGs are seeded per episode for determinism
        action = np.random.randint(0, 5)
        
        return action
    
    def load_model(self, model_path):
        """
        Helper method to load a PyTorch model.
        
        Args:
            model_path: Path to the .pth file
            
        Returns:
            Loaded model
        """
        # Example implementation:
        # model = YourNeuralNetwork()
        # if model_path.exists():
        #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
        #     model.eval()
        # return model
        pass


# Example Neural Network Architecture (customize as needed)
class ExampleNetwork(nn.Module):
    """
    Example neural network for the agent.
    Students should replace this with their own architecture.
    
    For discrete action space:
    - Input: observation (16 dims for prey, 14 dims for predator)
    - Output: 5 action logits (for Discrete(5) action space)
    """
    
    def __init__(self, input_dim, output_dim=5, hidden_dim=128):
        super(ExampleNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # No activation on output - these are logits for discrete actions
        )
    
    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Example usage
    print("Testing StudentAgent...")
    
    # Test prey agent (good agent has 16-dim observation)
    prey_agent = StudentAgent("prey")
    prey_obs = np.random.randn(16)  # Prey observation size
    prey_action = prey_agent.get_action(prey_obs, "agent_0")
    print(f"Prey observation shape: {prey_obs.shape}")
    print(f"Prey action: {prey_action} (should be in [0, 4])")
    
    # Test predator agent (adversary has 14-dim observation)
    predator_agent = StudentAgent("predator")
    predator_obs = np.random.randn(14)  # Predator observation size
    predator_action = predator_agent.get_action(predator_obs, "adversary_0")
    print(f"Predator observation shape: {predator_obs.shape}")
    print(f"Predator action: {predator_action} (should be in [0, 4])")
    
    print("✓ Agent template is working!")
