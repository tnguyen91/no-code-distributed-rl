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

class StopRequest(BaseModel):
    save_model: bool = True

@app.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str, req: StopRequest = StopRequest()):
    saved_path = manager.stop_experiment(exp_id, save_model=req.save_model)
    return {"ok": saved_path is not None or not req.save_model, "saved_path": saved_path}

@app.get("/experiments/{exp_id}/metrics")
def get_experiment_metrics(exp_id: str):
    metrics = get_metrics(exp_id)
    return {"experiment_id": exp_id, "metrics": metrics}

@app.get("/models")
def list_saved_models():
    return {"models": manager.list_saved_models()}

@app.delete("/models/{model_id}")
def delete_saved_model(model_id: str):
    ok = manager.delete_saved_model(model_id)
    return {"ok": ok}