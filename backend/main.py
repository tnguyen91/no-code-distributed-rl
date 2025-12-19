from fastapi import FastAPI
from fastapi import BackgroundTasks
from rl_baseline import train_cartpole
from pydantic import BaseModel
from experiments import manager

app = FastAPI()

class StartRequest(BaseModel):
    num_actors: int = 2

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train/baseline")
def train_baseline(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_cartpole)
    return {"status": "started"}

@app.post("/experiments")
def start_experiment(req: StartRequest):
    exp_id = manager.start_experiment(num_actors=req.num_actors)
    return {"experiment_id": exp_id}

@app.get("/experiments")
def list_experiments():
    return {"experiments": manager.list_experiments()}

@app.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str):
    ok = manager.stop_experiment(exp_id)
    return {"ok": ok}