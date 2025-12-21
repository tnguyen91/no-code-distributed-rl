from multiprocessing import Process
from typing import Dict, List
from uuid import uuid4

from distributed_rl.learner import start_distributed
from metrics_store import delete_experiment_metrics

class ExperimentManager:
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}

    def start_experiment(
        self,
        num_actors: int = 2,
        env_id: str = "CartPole-v1",
        algorithm: str = "ppo",
    ) -> str:
        exp_id = str(uuid4())
        learner, actors = start_distributed(
            exp_id=exp_id, num_actors=num_actors, env_id=env_id, algorithm=algorithm
        )
        self.experiments[exp_id] = {"learner": learner, "actors": actors, "algorithm": algorithm}
        return exp_id

    def stop_experiment(self, exp_id: str) -> bool:
        exp = self.experiments.get(exp_id)
        if exp is None:
            return False

        learner: Process = exp["learner"]
        actors: List[Process] = exp["actors"]

        for p in [learner, *actors]:
            if p.is_alive():
                p.terminate()
        for p in [learner, *actors]:
            p.join(timeout=2.0)

        del self.experiments[exp_id]
        delete_experiment_metrics(exp_id)
        return True

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
