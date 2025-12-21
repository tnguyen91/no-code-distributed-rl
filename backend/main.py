from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from rl_baseline import train_cartpole
from pydantic import BaseModel
from experiments import manager
from metrics_store import get_metrics

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    num_actors: int = 2
    env_id: str = "CartPole-v1"
    algorithm: str = "ppo"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train/baseline")
def train_baseline(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_cartpole)
    return {"status": "started"}

@app.post("/experiments")
def start_experiment(req: StartRequest):
    exp_id = manager.start_experiment(
        num_actors=req.num_actors, env_id=req.env_id, algorithm=req.algorithm
    )
    return {"experiment_id": exp_id}

@app.get("/experiments")
def list_experiments():
    return {"experiments": manager.list_experiments()}

@app.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str):
    ok = manager.stop_experiment(exp_id)
    return {"ok": ok}

@app.get("/experiments/{exp_id}/metrics")
def get_experiment_metrics(exp_id: str):
    metrics = get_metrics(exp_id)
    return {"experiment_id": exp_id, "metrics": metrics}