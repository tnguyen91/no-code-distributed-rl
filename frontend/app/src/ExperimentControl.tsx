import React, { useState, useEffect, useCallback } from "react";
import {
    startExperiment,
    stopExperiment,
    listExperiments,
    fetchMetrics,
    listSavedModels,
    deleteSavedModel,
    ExperimentSummary,
    MetricPoint,
    SavedModel,
} from "./api";
import MetricsChart from "./MetricsChart";
import "./App.css";

const POLL_INTERVAL_MS = 2000;

const ExperimentControl: React.FC = () => {
    const [numActors, setNumActors] = useState<number>(2);
    const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
    const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
    const [metrics, setMetrics] = useState<MetricPoint[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [envId, setEnvId] = useState<string>("CartPole-v1");
    const [algorithm, setAlgorithm] = useState<string>("ppo");
    const [isStarting, setIsStarting] = useState(false);
    const [isStopping, setIsStopping] = useState(false);
    const [savedModels, setSavedModels] = useState<SavedModel[]>([]);

    const refreshExperiments = useCallback(async () => {
        try {
            const list = await listExperiments();
            setExperiments(list);
        } catch (err: any) {
            setError(err.message ?? "Failed to list experiments");
        }
    }, []);

    const refreshSavedModels = useCallback(async () => {
        try {
            const models = await listSavedModels();
            setSavedModels(models);
        } catch (err: any) {}
    }, []);

    useEffect(() => {
        refreshExperiments();
        refreshSavedModels();
        const interval = setInterval(refreshExperiments, POLL_INTERVAL_MS);
        return () => clearInterval(interval);
    }, [refreshExperiments, refreshSavedModels]);

    useEffect(() => {
        if (!selectedExpId) {
            setMetrics([]);
            return;
        }
        fetchMetrics(selectedExpId).then(setMetrics).catch(() => {});

        const interval = setInterval(async () => {
            try {
                const m = await fetchMetrics(selectedExpId);
                setMetrics(m);
            } catch (err: any) {
                // experiment might have been stopped
            }
        }, POLL_INTERVAL_MS);
        return () => clearInterval(interval);
    }, [selectedExpId]);

    async function handleStart() {
        setError(null);
        setIsStarting(true);
        try {
            const expId = await startExperiment(numActors, envId, algorithm);
            setSelectedExpId(expId);
            await refreshExperiments();
        } catch (err: any) {
            setError(err.message ?? "Failed to start experiment");
        } finally {
            setIsStarting(false);
        }
    }

    async function handleStop() {
        if (!selectedExpId) return;
        setError(null);
        setIsStopping(true);
        try {
            const ok = await stopExperiment(selectedExpId);
            if (!ok) {
                setError("Failed to stop experiment");
                return;
            }
            setSelectedExpId(null);
            setMetrics([]);
            await refreshExperiments();
            await refreshSavedModels();
        } catch (err: any) {
            setError(err.message ?? "Failed to stop experiment");
        } finally {
            setIsStopping(false);
        }
    }

    async function handleDeleteModel(modelId: string) {
        try {
            await deleteSavedModel(modelId);
            await refreshSavedModels();
        } catch (err: any) {
            setError(err.message ?? "Failed to delete model");
        }
    }

    const selectedExp = experiments.find((e) => e.id === selectedExpId);
    const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
    const maxReward = metrics.length > 0 ? Math.max(...metrics.map((m) => m.avg_reward)) : 0;

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h1>Distributed RL Dashboard</h1>
                <p>Train reinforcement learning agents with distributed actors</p>
            </header>

            {error && (
                <div className="error-banner">
                    <span>{error}</span>
                    <button
                        onClick={() => setError(null)}
                        style={{ marginLeft: "auto", background: "none", border: "none", color: "inherit", cursor: "pointer" }}
                    >
                        Dismiss
                    </button>
                </div>
            )}

            <div className="main-grid">
                <aside className="sidebar">
                    {/* New Experiment Card */}
                    <div className="card">
                        <h2>New Experiment</h2>
                        <div className="form-group">
                            <label>Algorithm</label>
                            <select
                                value={algorithm}
                                onChange={(e) => setAlgorithm(e.target.value)}
                            >
                                <option value="ppo">PPO</option>
                                <option value="vtrace">V-trace (IMPALA)</option>
                                <option value="a2c">A2C</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Environment</label>
                            <select
                                value={envId}
                                onChange={(e) => setEnvId(e.target.value)}
                            >
                                <option value="CartPole-v1">CartPole-v1</option>
                                <option value="LunarLander-v2">LunarLander-v2</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Number of Actors</label>
                            <input
                                type="number"
                                min={1}
                                max={16}
                                value={numActors}
                                onChange={(e) => setNumActors(Math.max(1, parseInt(e.target.value || "1", 10)))}
                            />
                        </div>
                        <button
                            className="btn btn-primary btn-block"
                            onClick={handleStart}
                            disabled={isStarting}
                        >
                            {isStarting ? (
                                <>
                                    <span className="loading-spinner" />
                                    Starting...
                                </>
                            ) : (
                                "Start Training"
                            )}
                        </button>
                    </div>

                    {/* Experiments List Card */}
                    <div className="card">
                        <h2>Experiments</h2>
                        {experiments.length === 0 ? (
                            <div className="empty-state">
                                No active experiments.<br />
                                Start one above!
                            </div>
                        ) : (
                            <ul className="experiment-list">
                                {experiments.map((exp) => (
                                    <li
                                        key={exp.id}
                                        className={`experiment-item ${selectedExpId === exp.id ? "selected" : ""}`}
                                        onClick={() => setSelectedExpId(exp.id)}
                                    >
                                        <div className="experiment-item-info">
                                            <span className="experiment-item-id">
                                                {exp.id.slice(0, 8)}
                                            </span>
                                            <span className="experiment-item-meta">
                                                <span className={`status-dot ${exp.learner_alive ? "alive" : "stopped"}`} />
                                                {exp.num_actors} actor{exp.num_actors !== 1 ? "s" : ""}
                                            </span>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>

                    {/* Saved Models Card */}
                    <div className="card">
                        <h2>Saved Models</h2>
                        {savedModels.length === 0 ? (
                            <div className="empty-state">
                                No saved models yet.<br />
                                Stop an experiment to save its model.
                            </div>
                        ) : (
                            <ul className="experiment-list">
                                {savedModels.map((model) => (
                                    <li key={model.id} className="experiment-item">
                                        <div className="experiment-item-info">
                                            <span className="experiment-item-id">
                                                {model.id.slice(0, 8)}
                                            </span>
                                            <span className="experiment-item-meta">
                                                {model.algorithm.toUpperCase()} / {model.env_id}
                                            </span>
                                        </div>
                                        <button
                                            className="btn-icon"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteModel(model.id);
                                            }}
                                            title="Delete model"
                                        >
                                            &times;
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </aside>

                {/* Main Content Area */}
                <main>
                    <div className="card">
                        {!selectedExpId ? (
                            <div className="empty-state">
                                Select an experiment to view training metrics,<br />
                                or start a new experiment.
                            </div>
                        ) : (
                            <>
                                <div className="metrics-header">
                                    <div>
                                        <h2>Training Metrics</h2>
                                        <p style={{ margin: 0, color: "#71767b", fontSize: "0.85rem" }}>
                                            Experiment: {selectedExpId.slice(0, 8)}...
                                            {selectedExp && (
                                                <span className={`status-dot ${selectedExp.learner_alive ? "alive" : "stopped"}`} style={{ marginLeft: "0.5rem" }} />
                                            )}
                                            {selectedExp?.learner_alive ? "Training" : "Stopped"}
                                        </p>
                                    </div>
                                    <button
                                        className="btn btn-danger"
                                        onClick={handleStop}
                                        disabled={isStopping}
                                    >
                                        {isStopping ? (
                                            <>
                                                <span className="loading-spinner" />
                                                Stopping...
                                            </>
                                        ) : (
                                            "Stop"
                                        )}
                                    </button>
                                </div>

                                {metrics.length === 0 ? (
                                    <div className="empty-state">
                                        <span className="loading-spinner" style={{ width: 24, height: 24, marginBottom: "1rem" }} />
                                        <br />
                                        Waiting for training data...
                                    </div>
                                ) : (
                                    <>
                                        <div className="metrics-stats">
                                            <div className="stat-card">
                                                <div className="stat-value">{lastMetric?.update ?? 0}</div>
                                                <div className="stat-label">Updates</div>
                                            </div>
                                            <div className="stat-card">
                                                <div className="stat-value">{lastMetric?.avg_reward.toFixed(1) ?? "—"}</div>
                                                <div className="stat-label">Current Reward</div>
                                            </div>
                                            <div className="stat-card">
                                                <div className="stat-value">{maxReward.toFixed(1)}</div>
                                                <div className="stat-label">Best Reward</div>
                                            </div>
                                        </div>

                                        <div className="chart-container">
                                            <MetricsChart metrics={metrics} />
                                        </div>
                                    </>
                                )}
                            </>
                        )}
                    </div>
                </main>
            </div>
        </div>
    );
};

export default ExperimentControl;
