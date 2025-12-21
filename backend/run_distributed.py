from uuid import uuid4
from distributed_rl.learner import start_distributed

if __name__ == "__main__":
    exp_id = str(uuid4())
    print(f"Starting experiment: {exp_id}")
    learner, actors = start_distributed(exp_id=exp_id, num_actors=2)
    learner.join()
    for p in actors:
        p.join()