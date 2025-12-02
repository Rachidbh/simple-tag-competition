"""
Evaluation script for Simple Tag competition.

This script evaluates student submissions against private reference implementations.
It is designed to be run in the GitHub Actions workflow.
"""

import sys
import json
import importlib.util
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pettingzoo.mpe import simple_tag_v3
except ImportError:
    print("Error: pettingzoo is not installed. Run: pip install pettingzoo[mpe]")
    sys.exit(1)


class AgentLoader:
    """Utility class to load agent implementations."""
    
    @staticmethod
    def load_agent_from_file(file_path: Path, agent_type: str):
        """
        Dynamically load a StudentAgent from a Python file.
        
        Args:
            file_path: Path to the agent.py file
            agent_type: "prey" or "predator"
            
        Returns:
            Instantiated agent
        """
        try:
            spec = importlib.util.spec_from_file_location("student_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'StudentAgent'):
                raise AttributeError("Module must contain a 'StudentAgent' class")
            
            agent = module.StudentAgent(agent_type)
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to load agent from {file_path}: {e}")


class SimpleTagEvaluator:
    """Evaluator for Simple Tag environment."""
    
    def __init__(self):
        """Initialize evaluator (no seeding)."""
        pass
    
    def evaluate(
        self,
        prey_agent_path: Path,
        predator_agent_path: Path,
        num_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate agents in the Simple Tag environment.
        
        Args:
            prey_agent_path: Path to prey agent.py
            predator_agent_path: Path to predator agent.py
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading prey agent from: {prey_agent_path}")
        print(f"Loading predator agent from: {predator_agent_path}")
        
        # Load agents
        try:
            prey_agents = {}
            predator_agents = {}
            
            # We'll load agents as needed when we see agent IDs
            prey_loader = lambda: AgentLoader.load_agent_from_file(prey_agent_path, "prey")
            predator_loader = lambda: AgentLoader.load_agent_from_file(predator_agent_path, "predator")
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prey_score": 0,
                "predator_score": 0
            }
        
        # Run evaluation
        prey_rewards = []
        predator_rewards = []
        
        for episode in range(num_episodes):
            # Seed all RNGs deterministically for each episode
            episode_seed = episode
            np.random.seed(episode_seed)
            random.seed(episode_seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
            
            env = simple_tag_v3.parallel_env(
                num_good=1,  # Number of prey
                num_adversaries=3,  # Number of predators
                num_obstacles=2,
                max_cycles=max_steps,
                continuous_actions=False
            )
            
            # Seed each episode deterministically with the episode index
            observations, infos = env.reset(seed=episode)
            
            # Initialize agents for this episode
            prey_agents = {}
            predator_agents = {}
            
            episode_prey_reward = 0
            episode_predator_reward = 0
            steps = 0
            
            while env.agents:
                actions = {}
                
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    
                    # Determine if this is prey or predator
                    if "adversary" in agent_id:
                        # This is a predator
                        if agent_id not in predator_agents:
                            predator_agents[agent_id] = predator_loader()
                        action = predator_agents[agent_id].get_action(obs, agent_id)
                    else:
                        # This is prey
                        if agent_id not in prey_agents:
                            prey_agents[agent_id] = prey_loader()
                        action = prey_agents[agent_id].get_action(obs, agent_id)
                    
                    actions[agent_id] = action
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    if "adversary" in agent_id:
                        episode_predator_reward += reward
                    else:
                        episode_prey_reward += reward
                
                steps += 1
                
                if steps >= max_steps:
                    break
            
            prey_rewards.append(episode_prey_reward)
            predator_rewards.append(episode_predator_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")
            
            env.close()
        
        # Calculate statistics
        results = {
            "success": True,
            "prey_score": float(np.mean(prey_rewards)),
            "prey_std": float(np.std(prey_rewards)),
            "predator_score": float(np.mean(predator_rewards)),
            "predator_std": float(np.std(predator_rewards)),
            "num_episodes": num_episodes
        }
        
        return results


def evaluate_submission(
    student_submission_dir: Path,
    private_agents_dir: Path,
    output_file: Path,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """
    Evaluate a student submission against private reference implementations.
    
    Args:
        student_submission_dir: Directory containing student's agent.py
        private_agents_dir: Directory containing private reference agents
        output_file: Path to save evaluation results
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results dictionary
    """
    student_agent_path = student_submission_dir / "agent.py"
    private_prey_path = private_agents_dir / "prey_agent.py"
    private_predator_path = private_agents_dir / "predator_agent.py"
    
    # Validate paths
    if not student_agent_path.exists():
        return {
            "success": False,
            "error": f"Student agent not found at {student_agent_path}"
        }
    
    if not private_prey_path.exists() or not private_predator_path.exists():
        return {
            "success": False,
            "error": "Private reference agents not found"
        }
    
    print(f"\n{'='*60}")
    print(f"Evaluating submission: {student_submission_dir.name}")
    print(f"{'='*60}\n")
    
    evaluator = SimpleTagEvaluator()
    
    # Evaluate student as prey vs private predator
    print("\n--- Evaluating student PREY vs private PREDATOR ---")
    prey_results = evaluator.evaluate(
        prey_agent_path=student_agent_path,
        predator_agent_path=private_predator_path,
        num_episodes=num_episodes
    )
    
    # Evaluate student as predator vs private prey
    print("\n--- Evaluating student PREDATOR vs private PREY ---")
    predator_results = evaluator.evaluate(
        prey_agent_path=private_prey_path,
        predator_agent_path=student_agent_path,
        num_episodes=num_episodes
    )
    
    # Combine results
    if not prey_results["success"] or not predator_results["success"]:
        error_msg = prey_results.get("error", "") + " " + predator_results.get("error", "")
        return {
            "success": False,
            "error": error_msg.strip()
        }
    
    results = {
        "success": True,
        "student": student_submission_dir.name,
        "timestamp": datetime.now().isoformat(),
        "prey_score": prey_results["prey_score"],
        "prey_std": prey_results["prey_std"],
        "predator_score": predator_results["predator_score"],
        "predator_std": predator_results["predator_std"],
        "num_episodes": num_episodes
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Prey score: {results['prey_score']:.4f}")
    print(f"Predator score: {results['predator_score']:.4f}")
    print(f"Combined score: {((results['prey_score'] + results['predator_score']) / 2):.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point for evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Simple Tag competition submission")
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to student submission directory"
    )
    parser.add_argument(
        "--private-agents-dir",
        type=Path,
        default=Path("private_agents"),
        help="Path to private reference agents directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_evaluation.json"),
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes"
    )
    
    args = parser.parse_args()
    
    results = evaluate_submission(
        student_submission_dir=args.submission_dir,
        private_agents_dir=args.private_agents_dir,
        output_file=args.output,
        num_episodes=args.episodes
    )
    
    if not results["success"]:
        print(f"ERROR: {results['error']}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
