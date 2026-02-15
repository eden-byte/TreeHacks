"use client";

import { motion } from "framer-motion";
import {
  Cpu,
  HardDrive,
  Thermometer,
  Zap,
  Activity,
  CheckCircle,
  AlertCircle,
  Database,
} from "lucide-react";
import { useEffect, useState } from "react";

interface SystemMetrics {
  cpu: number;
  gpu: number;
  ram: number;
  temp: number;
  power: number;
}

interface FeatureStatus {
  name: string;
  status: "active" | "inactive" | "error";
  lastUpdate: string;
}

export default function SystemStatusPage() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: 45,
    gpu: 38,
    ram: 62,
    temp: 52,
    power: 78,
  });

  const features: FeatureStatus[] = [
    {
      name: "Object Detection (YOLOv5)",
      status: "active",
      lastUpdate: "2s ago",
    },
    {
      name: "Depth Estimation",
      status: "active",
      lastUpdate: "1s ago",
    },
    {
      name: "Face Recognition",
      status: "active",
      lastUpdate: "3s ago",
    },
    {
      name: "Scene Description (LLaMA)",
      status: "active",
      lastUpdate: "5s ago",
    },
    {
      name: "Haptic Motor Control",
      status: "active",
      lastUpdate: "1s ago",
    },
    {
      name: "Text-to-Speech",
      status: "active",
      lastUpdate: "2s ago",
    },
    {
      name: "RAG Indexing",
      status: "active",
      lastUpdate: "10s ago",
    },
    {
      name: "Cloud Sync",
      status: "inactive",
      lastUpdate: "N/A",
    },
  ];

  // Simulate metric updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        cpu: Math.max(20, Math.min(90, metrics.cpu + (Math.random() - 0.5) * 10)),
        gpu: Math.max(15, Math.min(85, metrics.gpu + (Math.random() - 0.5) * 10)),
        ram: Math.max(40, Math.min(80, metrics.ram + (Math.random() - 0.5) * 5)),
        temp: Math.max(45, Math.min(70, metrics.temp + (Math.random() - 0.5) * 3)),
        power: Math.max(60, Math.min(95, metrics.power + (Math.random() - 0.5) * 8)),
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [metrics]);

  const getStatusColor = (status: FeatureStatus["status"]) => {
    switch (status) {
      case "active":
        return "text-secondary bg-secondary/20";
      case "inactive":
        return "text-text-secondary bg-surface-elevated";
      case "error":
        return "text-red-600 bg-red-100";
    }
  };

  const getStatusIcon = (status: FeatureStatus["status"]) => {
    switch (status) {
      case "active":
        return <CheckCircle className="w-5 h-5" />;
      case "inactive":
        return <AlertCircle className="w-5 h-5" />;
      case "error":
        return <AlertCircle className="w-5 h-5" />;
    }
  };

  const getMetricColor = (value: number, thresholds: [number, number]) => {
    if (value < thresholds[0]) return "text-secondary";
    if (value < thresholds[1]) return "text-accent";
    return "text-red-600";
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-text-primary mb-3">
          System Status
        </h1>
        <p className="text-xl text-text-secondary">
          Real-time monitoring of Jetson Nano and system components
        </p>
      </motion.div>

      {/* Jetson Nano Metrics */}
      <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {/* CPU Usage */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
              <Cpu className="w-6 h-6 text-primary" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-secondary">
                CPU Usage
              </h3>
            </div>
          </div>
          <div className={`text-4xl font-bold ${getMetricColor(metrics.cpu, [60, 80])}`}>
            {metrics.cpu.toFixed(0)}%
          </div>
          <div className="w-full bg-surface rounded-full h-2 mt-3">
            <div
              className="bg-primary h-2 rounded-full transition-all duration-500"
              style={{ width: `${metrics.cpu}%` }}
            />
          </div>
        </motion.div>

        {/* GPU Usage */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-secondary/10 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-secondary" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-secondary">
                GPU Usage
              </h3>
            </div>
          </div>
          <div className={`text-4xl font-bold ${getMetricColor(metrics.gpu, [60, 80])}`}>
            {metrics.gpu.toFixed(0)}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className="bg-secondary h-2 rounded-full transition-all duration-500"
              style={{ width: `${metrics.gpu}%` }}
            />
          </div>
        </motion.div>

        {/* RAM Usage */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
              <HardDrive className="w-6 h-6 text-accent" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-secondary">
                RAM Usage
              </h3>
            </div>
          </div>
          <div className={`text-4xl font-bold ${getMetricColor(metrics.ram, [70, 85])}`}>
            {metrics.ram.toFixed(0)}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className="bg-accent h-2 rounded-full transition-all duration-500"
              style={{ width: `${metrics.ram}%` }}
            />
          </div>
          <div className="text-sm text-text-secondary mt-2">2.5 GB / 4 GB</div>
        </motion.div>

        {/* Temperature */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
              <Thermometer className="w-6 h-6 text-red-600" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-secondary">
                Temperature
              </h3>
            </div>
          </div>
          <div className={`text-4xl font-bold ${getMetricColor(metrics.temp, [60, 75])}`}>
            {metrics.temp.toFixed(0)}°C
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className="bg-red-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${(metrics.temp / 85) * 100}%` }}
            />
          </div>
          <div className="text-sm text-text-secondary mt-2">Max: 85°C</div>
        </motion.div>

        {/* Power Draw */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <Zap className="w-6 h-6 text-green-600" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-secondary">
                Battery
              </h3>
            </div>
          </div>
          <div className={`text-4xl font-bold ${getMetricColor(100 - metrics.power, [60, 80])}`}>
            {(100 - metrics.power).toFixed(0)}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className="bg-green-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${100 - metrics.power}%` }}
            />
          </div>
          <div className="text-sm text-text-secondary mt-2">~3h remaining</div>
        </motion.div>
      </div>

      {/* Feature Status */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Active Features */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
          className="card p-6"
        >
          <h2 className="text-2xl font-bold text-text-primary mb-6 flex items-center gap-3">
            <Activity className="w-7 h-7 text-primary" aria-hidden="true" />
            <span>Feature Status</span>
          </h2>
          <div className="space-y-3">
            {features.map((feature, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 bg-background rounded-lg border border-border"
              >
                <div className="flex items-center gap-3">
                  <div className={`${getStatusColor(feature.status)} rounded-full p-2`}>
                    {getStatusIcon(feature.status)}
                  </div>
                  <div>
                    <div className="font-semibold text-text-primary">
                      {feature.name}
                    </div>
                    <div className="text-sm text-text-secondary">
                      {feature.lastUpdate}
                    </div>
                  </div>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-semibold capitalize ${getStatusColor(
                    feature.status
                  )}`}
                >
                  {feature.status}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* RAG & Storage */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.7 }}
          className="space-y-6"
        >
          <div className="card p-6">
            <h2 className="text-2xl font-bold text-text-primary mb-6 flex items-center gap-3">
              <Database className="w-7 h-7 text-secondary" aria-hidden="true" />
              <span>RAG Indexing</span>
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Total Documents</span>
                <span className="text-2xl font-bold text-text-primary">142</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Indexed Conversations</span>
                <span className="text-2xl font-bold text-text-primary">387</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Vector Database Size</span>
                <span className="text-2xl font-bold text-text-primary">256 MB</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Last Update</span>
                <span className="text-lg font-semibold text-secondary">12s ago</span>
              </div>
            </div>
          </div>

          <div className="card p-6 bg-primary/5 border-2 border-primary/20">
            <h3 className="text-xl font-bold text-text-primary mb-4">
              System Health
            </h3>
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle className="w-8 h-8 text-secondary" aria-hidden="true" />
              <div>
                <div className="text-lg font-semibold text-text-primary">
                  All Systems Operational
                </div>
                <div className="text-sm text-text-secondary">
                  Last check: just now
                </div>
              </div>
            </div>
            <div className="text-sm text-text-secondary">
              Uptime: 4d 12h 34m
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
