import os
from multiprocessing import Process
from typing import Dict, List, Optional
from uuid import uuid4
import torch

from distributed_rl.learner import start_distributed
from metrics_store import delete_experiment_metrics

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

class ExperimentManager:
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}

    def start_experiment(
        self,
        num_actors: int = 2,
        env_id: str = "CartPole-v1",
        algorithm: str = "ppo",
        max_updates: Optional[int] = None,
        target_reward: Optional[float] = None,
    ) -> str:
        exp_id = str(uuid4())
        save_path = os.path.join(MODELS_DIR, f"{exp_id}.pt")
        learner, actors, model, stop_event = start_distributed(
            exp_id=exp_id,
            num_actors=num_actors,
            env_id=env_id,
            algorithm=algorithm,
            max_updates=max_updates,
            target_reward=target_reward,
            save_path=save_path,
        )
        self.experiments[exp_id] = {
            "learner": learner,
            "actors": actors,
            "algorithm": algorithm,
            "env_id": env_id,
            "model": model,
            "stop_event": stop_event,
        }
        return exp_id

    def stop_experiment(self, exp_id: str) -> Optional[str]:
        exp = self.experiments.get(exp_id)
        if exp is None:
            return None

        learner: Process = exp["learner"]
        actors: List[Process] = exp["actors"]
        stop_event = exp.get("stop_event")

        if stop_event is not None:
            stop_event.set()

        learner.join(timeout=5.0)
        if learner.is_alive():
            learner.terminate()

        for p in actors:
            if p.is_alive():
                p.terminate()
        for p in actors:
            p.join(timeout=1.0)

        saved_path = os.path.join(MODELS_DIR, f"{exp_id}.pt")
        if not os.path.exists(saved_path):
            saved_path = None

        del self.experiments[exp_id]
        delete_experiment_metrics(exp_id)
        return saved_path

    def list_saved_models(self) -> List[Dict]:
        """List all saved models."""
        models = []
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pt"):
                path = os.path.join(MODELS_DIR, filename)
                try:
                    checkpoint = torch.load(path, weights_only=False)
                    models.append({
                        "id": filename[:-3],  # Remove .pt extension
                        "algorithm": checkpoint.get("algorithm", "unknown"),
                        "env_id": checkpoint.get("env_id", "unknown"),
                        "path": path,
                    })
                except Exception:
                    pass
        return models

    def delete_saved_model(self, model_id: str) -> bool:
        """Delete a saved model."""
        path = os.path.join(MODELS_DIR, f"{model_id}.pt")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_experiments(self):
        out = []
        for exp_id, exp in self.experiments.items():
            learner: Process = exp["learner"]
            actors: List[Process] = exp["actors"]
            out.append({
                "id": exp_id,
                "learner_alive": learner.is_alive(),
                "num_actors": len(actors),
                "actors_alive": [p.is_alive() for p in actors],
            })
        return out

manager = ExperimentManager()
