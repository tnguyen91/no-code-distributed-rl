from multiprocessing.managers import SyncManager
from typing import Dict, List, Any, Optional

_manager: Optional[SyncManager] = None
experiments_metrics: Optional[Dict[str, List[Dict[str, Any]]]] = None

def _ensure_initialized():
    global _manager, experiments_metrics
    if _manager is None:
        from multiprocessing import Manager
        _manager = Manager()
        experiments_metrics = _manager.dict()

def init_experiment_metrics(exp_id: str):
    """Initialize metrics for an experiment. Returns the shared list for passing to child processes."""
    _ensure_initialized()
    experiments_metrics[exp_id] = _manager.list()
    return experiments_metrics[exp_id]

def get_metrics_list(exp_id: str):
    """Get the shared list object for passing to child processes."""
    if experiments_metrics is None or exp_id not in experiments_metrics:
        return None
    return experiments_metrics[exp_id]

def add_metric(exp_id: str, update_index: int, avg_return: float):
    if experiments_metrics is None or exp_id not in experiments_metrics:
        return
    experiments_metrics[exp_id].append(
        {"update": update_index, "avg_return": avg_return}
    )

def add_metric_to_list(metrics_list, update_index: int, avg_return: float):
    """Add metric directly to a shared list (for use in child processes)."""
    metrics_list.append({"update": update_index, "avg_return": avg_return})

def get_metrics(exp_id: str):
    if experiments_metrics is None or exp_id not in experiments_metrics:
        return []
    return list(experiments_metrics[exp_id])

def delete_experiment_metrics(exp_id: str):
    if experiments_metrics is not None and exp_id in experiments_metrics:
        del experiments_metrics[exp_id]