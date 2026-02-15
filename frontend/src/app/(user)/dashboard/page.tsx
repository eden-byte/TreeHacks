"use client";

import { motion } from "framer-motion";
import { Video, Camera, Mic, Volume2, Navigation } from "lucide-react";
import { useState, useEffect } from "react";
import { useUserRole } from "../layout";

interface MotorState {
  id: number;
  intensity: 0 | 33 | 66 | 100;
  direction: string;
  angle: number;
}

const initialMotorStates: MotorState[] = [
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

export default function DashboardPage() {
  const userRole = useUserRole();
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [isMicActive, setIsMicActive] = useState(true);
  const [isSoundActive, setIsSoundActive] = useState(true);
  const [motorStates, setMotorStates] = useState<MotorState[]>(initialMotorStates);
  const [detectedObjects, setDetectedObjects] = useState<string[]>([]);
  const [fps, setFps] = useState(30);

  // Simulate real-time motor updates
  useEffect(() => {
    if (!isCameraActive) return;

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
  }, [isCameraActive]);

  // Simulate object detection
  useEffect(() => {
    if (!isCameraActive) {
      setDetectedObjects([]);
      return;
    }

    const objects = ["Person", "Chair", "Table", "Door", "Window", "Book", "Phone"];
    const interval = setInterval(() => {
      const numObjects = Math.floor(Math.random() * 4) + 1;
      const detected = [];
      for (let i = 0; i < numObjects; i++) {
        detected.push(objects[Math.floor(Math.random() * objects.length)]);
      }
      setDetectedObjects(detected);
    }, 3000);

    return () => clearInterval(interval);
  }, [isCameraActive]);

  // Simulate FPS changes
  useEffect(() => {
    const interval = setInterval(() => {
      setFps(Math.floor(Math.random() * 5) + 28); // 28-32 fps
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Only show content for Healthcare Provider mode
  if (userRole !== "Healthcare Provider") {
    return null;
  }

  return (
    <div className="h-full flex flex-col">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-6"
      >
        <h1 className="text-4xl font-bold text-text-primary mb-2">
          Live Camera Feed
        </h1>
        <p className="text-lg text-text-secondary">
          Bring the visually impaired one step closer to full independence
        </p>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6 flex-1">
        {/* Video Feed - Takes 2 columns */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-2 flex flex-col"
        >
          <div className="card p-0 overflow-hidden flex-1 flex flex-col">
          <div className="relative flex-1 bg-gray-900 min-h-[500px]">
            {/* Video Placeholder */}
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
              <div className="text-center">
                <Video className="w-32 h-32 text-gray-600 mx-auto mb-4" aria-hidden="true" />
                <p className="text-text-secondary text-xl">Camera feed will appear here</p>
                <p className="text-text-tertiary text-sm mt-2">Connect your smart glasses to begin</p>
              </div>
            </div>

            {/* Live Indicator */}
            {isCameraActive && (
              <div className="absolute top-4 right-4 px-4 py-2 bg-black/70 text-white rounded-lg text-sm font-semibold">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                  <span>LIVE</span>
                </div>
              </div>
            )}

            {/* Status Info Overlay */}
            <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between flex-wrap gap-2">
              <div className="px-4 py-2 bg-black/70 text-white rounded-lg text-sm">
                <span className="text-text-secondary">Objects: </span>
                <span className="font-semibold text-secondary">{detectedObjects.length}</span>
              </div>
              <div className="px-4 py-2 bg-black/70 text-white rounded-lg text-sm">
                <span className="text-text-secondary">FPS: </span>
                <span className="font-semibold">{fps}</span>
              </div>
              <div className="px-4 py-2 bg-black/70 text-white rounded-lg text-sm max-w-xs truncate">
                <span className="text-text-secondary">Detecting: </span>
                <span className="font-semibold">{detectedObjects.join(", ") || "None"}</span>
              </div>
            </div>
          </div>

          {/* Controls Bar */}
          <div className="p-6 bg-surface-elevated border-t-2 border-border">
            <div className="flex items-center justify-center gap-4">
              <button
                onClick={() => setIsCameraActive(!isCameraActive)}
                className={`flex items-center gap-3 px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                  isCameraActive
                    ? "bg-primary text-white shadow-md"
                    : "bg-surface text-text-secondary border-2 border-border"
                }`}
                aria-label={isCameraActive ? "Turn off camera" : "Turn on camera"}
              >
                <Camera className="w-6 h-6" aria-hidden="true" />
                <span className="text-lg">Camera {isCameraActive ? "On" : "Off"}</span>
              </button>

              <button
                onClick={() => setIsMicActive(!isMicActive)}
                className={`flex items-center gap-3 px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                  isMicActive
                    ? "bg-secondary text-white shadow-md"
                    : "bg-surface text-text-secondary border-2 border-border"
                }`}
                aria-label={isMicActive ? "Mute microphone" : "Unmute microphone"}
              >
                <Mic className="w-6 h-6" aria-hidden="true" />
                <span className="text-lg">Voice {isMicActive ? "On" : "Off"}</span>
              </button>

              <button
                onClick={() => setIsSoundActive(!isSoundActive)}
                className={`flex items-center gap-3 px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                  isSoundActive
                    ? "bg-accent text-white shadow-md"
                    : "bg-surface text-text-secondary border-2 border-border"
                }`}
                aria-label={isSoundActive ? "Mute audio" : "Unmute audio"}
              >
                <Volume2 className="w-6 h-6" aria-hidden="true" />
                <span className="text-lg">Audio {isSoundActive ? "On" : "Off"}</span>
              </button>
            </div>
          </div>
          </div>
        </motion.div>

        {/* Haptic Motor Visualization - 1 column */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="flex flex-col"
        >
          <div className="card p-6 flex-1 bg-gradient-to-br from-primary/5 to-secondary/5">
            <h2 className="text-2xl font-bold text-text-primary mb-4 flex items-center gap-2">
              <Navigation className="w-6 h-6" />
              <span>Haptic Feedback</span>
            </h2>

            {/* Motor Visualization */}
            <div className="relative mx-auto" style={{ height: "300px", maxWidth: "400px" }}>
              <div className="absolute inset-0 flex items-center justify-center">
                {/* Central Reference Point */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="w-2 h-2 bg-text-secondary/30 rounded-full"></div>
                </div>

                {/* Motor Disks in Semicircular Arc */}
                {motorStates.map((motor, index) => {
                  const totalMotors = motorStates.length;
                  const angleStep = 180 / (totalMotors - 1);
                  const angle = -180 + index * angleStep; // Rotated 90° left
                  const radius = 120;
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
                          width: "60px",
                          height: "60px",
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
                          <div className="text-xl font-bold text-primary">
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
                      <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                        <div className="px-2 py-1 bg-surface rounded-full border border-border shadow-md">
                          <span className="text-xs font-semibold text-text-primary">
                            {motor.direction}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}

                {/* Arc Line */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ opacity: 0.2 }}
                >
                  <path
                    d="M 40 150 Q 200 30, 360 150"
                    stroke="currentColor"
                    strokeWidth="2"
                    fill="none"
                    strokeDasharray="5,5"
                    className="text-text-secondary"
                  />
                </svg>
              </div>
            </div>

            {/* Motor Status Summary */}
            <div className="mt-6 space-y-2">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Status</h3>
              {motorStates.map((motor) => (
                <div key={motor.id} className="flex items-center justify-between text-sm">
                  <span className="text-text-secondary">Motor {motor.id}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-surface rounded-full h-2">
                      <div
                        className="bg-secondary h-2 rounded-full transition-all duration-500"
                        style={{ width: `${motor.intensity}%` }}
                      />
                    </div>
                    <span className={`font-semibold w-8 ${
                      motor.intensity === 0
                        ? "text-text-secondary"
                        : motor.intensity === 33
                        ? "text-accent"
                        : motor.intensity === 66
                        ? "text-accent"
                        : "text-secondary"
                    }`}>
                      {motor.intensity === 0 ? "-" : motor.intensity === 33 ? "Low" : motor.intensity === 66 ? "Med" : "High"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
