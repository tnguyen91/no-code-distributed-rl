import os
from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rl_baseline import train_cartpole
from pydantic import BaseModel
from experiments import manager, MODELS_DIR
from metrics_store import get_metrics
from distributed_rl.evaluate import evaluate_model

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

from typing import Optional

class StartRequest(BaseModel):
    num_actors: int = 2
    env_id: str = "CartPole-v1"
    algorithm: str = "ppo"
    max_updates: Optional[int] = None
    target_reward: Optional[float] = None

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
        num_actors=req.num_actors,
        env_id=req.env_id,
        algorithm=req.algorithm,
        max_updates=req.max_updates,
        target_reward=req.target_reward,
    )
    return {"experiment_id": exp_id}

@app.get("/experiments")
def list_experiments():
    return {"experiments": manager.list_experiments()}

@app.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str):
    saved_path = manager.stop_experiment(exp_id)
    return {"ok": True, "saved_path": saved_path}

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

class EvaluateRequest(BaseModel):
    num_episodes: int = 10
    record_video: bool = True
    num_episodes_to_record: int = 1

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "eval_videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

@app.post("/models/{model_id}/evaluate")
def evaluate_saved_model(model_id: str, req: EvaluateRequest = EvaluateRequest()):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
    if not os.path.exists(model_path):
        return {"ok": False, "error": "Model not found"}

    video_dir = os.path.join(VIDEOS_DIR, model_id) if req.record_video else None

    result = evaluate_model(
        model_path=model_path,
        num_episodes=req.num_episodes,
        record_video=req.record_video,
        video_dir=video_dir,
        num_episodes_to_record=req.num_episodes_to_record,
    )

    video_urls = []
    for video_path in result.get("video_paths", []):
        rel_path = os.path.relpath(video_path, VIDEOS_DIR)
        video_urls.append(f"/videos/{rel_path.replace(os.sep, '/')}")

    return {
        "ok": True,
        "avg_reward": result["avg_reward"],
        "min_reward": result["min_reward"],
        "max_reward": result["max_reward"],
        "episode_rewards": result["episode_rewards"],
        "episode_lengths": result["episode_lengths"],
        "video_urls": video_urls,
        "env_id": result["env_id"],
        "algorithm": result["algorithm"],
    }