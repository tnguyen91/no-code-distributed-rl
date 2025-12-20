import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";
import { MetricPoint } from "./api";

interface Props {
    metrics: MetricPoint[];
}

const MetricsChart: React.FC<Props> = ({ metrics }) => {
    if (metrics.length === 0) {
        return <div className="empty-state">No metrics to display.</div>;
    }

    const maxReturn = Math.max(...metrics.map((m) => m.avg_return));

    return (
        <ResponsiveContainer width="100%" height={350}>
            <LineChart data={metrics} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2f3336" />
                <XAxis
                    dataKey="update"
                    stroke="#71767b"
                    tick={{ fill: "#71767b", fontSize: 12 }}
                    axisLine={{ stroke: "#2f3336" }}
                    tickLine={{ stroke: "#2f3336" }}
                    label={{
                        value: "Training Updates",
                        position: "insideBottom",
                        offset: -10,
                        fill: "#71767b",
                        fontSize: 12,
                    }}
                />
                <YAxis
                    stroke="#71767b"
                    tick={{ fill: "#71767b", fontSize: 12 }}
                    axisLine={{ stroke: "#2f3336" }}
                    tickLine={{ stroke: "#2f3336" }}
                    label={{
                        value: "Average Return",
                        angle: -90,
                        position: "insideLeft",
                        fill: "#71767b",
                        fontSize: 12,
                        dx: 10,
                    }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: "#16181c",
                        border: "1px solid #2f3336",
                        borderRadius: 8,
                        color: "#e7e9ea",
                    }}
                    labelStyle={{ color: "#71767b" }}
                    formatter={(value) => [typeof value === "number" ? value.toFixed(2) : value, "Avg Return"]}
                    labelFormatter={(label) => `Update ${label}`}
                />
                <ReferenceLine
                    y={maxReturn}
                    stroke="#00ba7c"
                    strokeDasharray="5 5"
                    strokeOpacity={0.5}
                />
                <Line
                    type="monotone"
                    dataKey="avg_return"
                    stroke="#1d9bf0"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#1d9bf0", stroke: "#fff", strokeWidth: 2 }}
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default MetricsChart;
