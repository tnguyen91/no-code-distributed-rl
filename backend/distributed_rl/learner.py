from multiprocessing import Process, Queue
from typing import List
from .actors import actor_loop

def learner_loop(experience_queue: Queue):
    batch = []
    while True:
        exp = experience_queue.get()  # blocking
        batch.append(exp)
        if len(batch) >= 1024:
            print(f"Got batch of {len(batch)} experiences")
            # TODO: implement policy gradient update here
            batch.clear()

def start_distributed(num_actors: int = 2):
    experience_queue: Queue = Queue(maxsize=10_000)

    # Start learner
    learner = Process(target=learner_loop, args=(experience_queue,))
    learner.start()

    # Start actors
    actors: List[Process] = []
    for i in range(num_actors):
        p = Process(target=actor_loop, args=(i, experience_queue))
        p.start()
        actors.append(p)

    learner.join()
    for p in actors:
        p.join()