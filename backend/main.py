from fastapi import FastAPI
from fastapi import BackgroundTasks
from rl_baseline import train_cartpole

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train/baseline")
def train_baseline(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_cartpole)
    return {"status": "started"}