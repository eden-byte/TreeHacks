"use client";

import { motion } from "framer-motion";
import { Wifi, WifiOff, Play, Pause } from "lucide-react";
import { useState, useEffect } from "react";

interface MotorState {
  id: number;
  intensity: 0 | 33 | 66 | 100;
  direction: string;
  angle: number;
}

const mockMotorData: MotorState[] = [
  { id: 1, intensity: 66, direction: "Forward", angle: 0 },
  { id: 2, intensity: 33, direction: "Left-45°", angle: -45 },
  { id: 3, intensity: 100, direction: "Left-90°", angle: -90 },
  { id: 4, intensity: 33, direction: "Right-45°", angle: 45 },
  { id: 5, intensity: 0, direction: "Right-90°", angle: 90 },
];

const getAnimationClass = (intensity: number) => {
  switch (intensity) {
    case 0:
      return "";
    case 33:
      return "animate-pulse-weak";
    case 66:
      return "animate-pulse-medium";
    case 100:
      return "animate-pulse-strong";
    default:
      return "";
  }
};

const getOpacity = (intensity: number) => {
  switch (intensity) {
    case 0:
      return "opacity-20";
    case 33:
      return "opacity-40";
    case 66:
      return "opacity-70";
    case 100:
      return "opacity-100";
    default:
      return "opacity-20";
  }
};

export default function MotorVisualizationPage() {
  const [isConnected] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [motorStates, setMotorStates] = useState<MotorState[]>(mockMotorData);

  // Simulate real-time motor data updates
  useEffect(() => {
    if (isPaused || !isConnected) return;

    const interval = setInterval(() => {
      setMotorStates((prev) =>
        prev.map((motor) => ({
          ...motor,
          intensity: [0, 33, 66, 100][
            Math.floor(Math.random() * 4)
          ] as MotorState["intensity"],
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, [isPaused, isConnected]);

  return (
    <div className="max-w-6xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-4xl font-bold text-text-primary mb-3">
              Haptic Motor Visualization
            </h1>
            <p className="text-xl text-text-secondary">
              Real-time feedback from the 5-motor haptic necklace array
            </p>
          </div>

          {/* Connection Status & Controls */}
          <div className="flex items-center gap-4">
            <div
              className={`flex items-center gap-2 px-4 py-3 rounded-lg ${
                isConnected
                  ? "bg-secondary/20 text-secondary"
                  : "bg-red-100 text-red-600"
              }`}
            >
              {isConnected ? (
                <>
                  <Wifi className="w-5 h-5" aria-hidden="true" />
                  <span className="font-semibold">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5" aria-hidden="true" />
                  <span className="font-semibold">Disconnected</span>
                </>
              )}
            </div>

            <button
              onClick={() => setIsPaused(!isPaused)}
              className="btn-primary flex items-center gap-2"
              aria-label={isPaused ? "Resume visualization" : "Pause visualization"}
            >
              {isPaused ? (
                <>
                  <Play className="w-5 h-5" aria-hidden="true" />
                  <span>Resume</span>
                </>
              ) : (
                <>
                  <Pause className="w-5 h-5" aria-hidden="true" />
                  <span>Pause</span>
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Motor Visualization */}
      <div className="card p-12 md:p-16 bg-gradient-to-br from-primary/5 to-secondary/5">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="relative mx-auto"
          style={{ maxWidth: "600px", height: "400px" }}
        >
          {/* Necklace Arc Container */}
          <div className="absolute inset-0 flex items-center justify-center">
            {/* Central Reference Point */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-3 h-3 bg-text-secondary/30 rounded-full"></div>
            </div>

            {/* Motor Disks in Semicircular Arc */}
            {motorStates.map((motor, index) => {
              // Calculate position in semicircular arc (180 degrees, top half)
              const totalMotors = motorStates.length;
              const angleStep = 180 / (totalMotors - 1);
              const angle = -90 + index * angleStep; // -90 to +90 degrees
              const radius = 180; // Distance from center
              const radian = (angle * Math.PI) / 180;
              const x = Math.cos(radian) * radius;
              const y = Math.sin(radian) * radius;

              return (
                <motion.div
                  key={motor.id}
                  className="absolute"
                  style={{
                    left: `calc(50% + ${x}px)`,
                    top: `calc(50% + ${y}px)`,
                    transform: "translate(-50%, -50%)",
                  }}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div
                    className={`motor-disk-active ${getOpacity(
                      motor.intensity
                    )} ${getAnimationClass(motor.intensity)} ${
                      motor.intensity > 0 ? "shadow-2xl" : "shadow-lg"
                    }`}
                    style={{
                      width: "80px",
                      height: "80px",
                      boxShadow:
                        motor.intensity > 0
                          ? `0 0 ${motor.intensity / 2}px rgba(59, 130, 246, 0.6)`
                          : undefined,
                    }}
                    role="status"
                    aria-label={`Motor ${motor.id}: ${motor.direction}, ${motor.intensity}% intensity`}
                  >
                    {/* Motor ID */}
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {motor.id}
                      </div>
                      <div
                        className={`text-xs font-semibold ${
                          motor.intensity > 0 ? "text-secondary" : "text-text-secondary"
                        }`}
                      >
                        {motor.intensity}%
                      </div>
                    </div>
                  </div>

                  {/* Direction Label */}
                  <div
                    className="absolute top-full mt-3 left-1/2 transform -translate-x-1/2 whitespace-nowrap"
                  >
                    <div className="px-3 py-1 bg-surface rounded-full border-2 border-border shadow-md">
                      <span className="text-sm font-semibold text-text-primary">
                        {motor.direction}
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}

            {/* Arc Line (Visual Guide) */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ opacity: 0.2 }}
            >
              <path
                d="M 60 200 Q 300 20, 540 200"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeDasharray="5,5"
                className="text-text-secondary"
              />
            </svg>
          </div>
        </motion.div>
      </div>

      {/* Motor Status Table */}
      <div className="grid md:grid-cols-5 gap-4 mt-8">
        {motorStates.map((motor) => (
          <motion.div
            key={motor.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: motor.id * 0.1 }}
            className="card p-6 text-center"
          >
            <div className="text-3xl font-bold text-primary mb-2">
              Motor {motor.id}
            </div>
            <div className="text-base text-text-secondary mb-3">
              {motor.direction}
            </div>
            <div
              className={`text-2xl font-bold ${
                motor.intensity === 0
                  ? "text-text-secondary"
                  : motor.intensity === 33
                  ? "text-accent"
                  : motor.intensity === 66
                  ? "text-accent"
                  : "text-secondary"
              }`}
            >
              {motor.intensity}%
            </div>
            <div className="text-sm text-text-secondary mt-2">
              {motor.intensity === 0
                ? "Inactive"
                : motor.intensity === 33
                ? "Weak"
                : motor.intensity === 66
                ? "Medium"
                : "Strong"}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Legend */}
      <div className="card p-6 mt-8 bg-background">
        <h3 className="text-xl font-bold text-text-primary mb-4">
          Intensity Legend
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-full bg-white opacity-20"></div>
            <span className="text-base text-text-secondary">0% - Inactive</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-full bg-white opacity-40 animate-pulse-weak"></div>
            <span className="text-base text-text-secondary">33% - Weak</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-full bg-white opacity-70 animate-pulse-medium"></div>
            <span className="text-base text-text-secondary">66% - Medium</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-full bg-white opacity-100 animate-pulse-strong"></div>
            <span className="text-base text-text-secondary">100% - Strong</span>
          </div>
        </div>
      </div>
    </div>
  );
}
