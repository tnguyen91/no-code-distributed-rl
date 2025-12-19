from distributed_rl.learner import start_distributed

if __name__ == "__main__":
    learner, actors = start_distributed(num_actors=2)
    learner.join()
    for p in actors:
        p.join()