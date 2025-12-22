const BASE_URL = "http://127.0.0.1:8000";

export interface StartExperimentResponse {
    experiment_id: string;
}

export interface ExperimentSummary {
    id: string;
    learner_alive: boolean;
    num_actors: number;
    actors_alive: boolean[];
}

export interface ListExperimentsResponse {
    experiments: ExperimentSummary[];
}

export interface MetricPoint {
    update: number;
    avg_reward: number;
}

export interface MetricsResponse {
    experiment_id: string;
    metrics: MetricPoint[];
}

export async function startExperiment(
    numActors: number,
    envId: string,
    algorithm: string = "ppo"
): Promise<string> {
    const res = await fetch(`${BASE_URL}/experiments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_actors: numActors, env_id: envId, algorithm }),
    });
    if (!res.ok) {
        throw new Error("Failed to start experiment");
    }
    const data: StartExperimentResponse = await res.json();
    return data.experiment_id;
}

export async function stopExperiment(expId: string): Promise<boolean> {
    const res = await fetch(`${BASE_URL}/experiments/${expId}/stop`, {
        method: "POST",
    });
    if (!res.ok) {
        return false;
    }
    const data = await res.json();
    return data.ok;
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
    const res = await fetch(`${BASE_URL}/experiments`);
    if (!res.ok) {
        throw new Error("Failed to list experiments");
    }
    const data: ListExperimentsResponse = await res.json();
    return data.experiments;
}

export async function fetchMetrics(expId: string): Promise<MetricPoint[]> {
    const res = await fetch(`${BASE_URL}/experiments/${expId}/metrics`);
    if (!res.ok) {
        throw new Error("Failed to fetch metrics");
    }
    const data: MetricsResponse = await res.json();
    return data.metrics;
}

export interface SavedModel {
    id: string;
    algorithm: string;
    env_id: string;
}

export async function listSavedModels(): Promise<SavedModel[]> {
    const res = await fetch(`${BASE_URL}/models`);
    if (!res.ok) {
        throw new Error("Failed to list models");
    }
    const data = await res.json();
    return data.models;
}

export async function deleteSavedModel(modelId: string): Promise<boolean> {
    const res = await fetch(`${BASE_URL}/models/${modelId}`, {
        method: "DELETE",
    });
    if (!res.ok) {
        return false;
    }
    const data = await res.json();
    return data.ok;
}