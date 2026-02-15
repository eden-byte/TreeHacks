"use client";

import { motion } from "framer-motion";
import { Video, Camera, Layers, Grid3x3, Pause, Play } from "lucide-react";
import { useState } from "react";

interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

const mockDetections: DetectedObject[] = [
  {
    id: "1",
    label: "Person",
    confidence: 0.95,
    x: 20,
    y: 15,
    width: 25,
    height: 60,
  },
  {
    id: "2",
    label: "Chair",
    confidence: 0.88,
    x: 60,
    y: 50,
    width: 20,
    height: 30,
  },
  {
    id: "3",
    label: "Table",
    confidence: 0.92,
    x: 50,
    y: 65,
    width: 35,
    height: 25,
  },
  {
    id: "4",
    label: "Door",
    confidence: 0.86,
    x: 5,
    y: 10,
    width: 15,
    height: 50,
  },
];

export default function LiveFeedPage() {
  const [isPaused, setIsPaused] = useState(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [showDepthMap, setShowDepthMap] = useState(false);

  const handleScreenshot = () => {
    alert("Screenshot captured! (In a real app, this would save the current frame)");
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-4xl font-bold text-text-primary mb-3">
              Live Camera Feed
            </h1>
            <p className="text-xl text-text-secondary">
              Real-time view from smart glasses with AI object detection
            </p>
          </div>

          {/* Stream Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsPaused(!isPaused)}
              className="btn-primary flex items-center gap-2"
              aria-label={isPaused ? "Resume stream" : "Pause stream"}
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

            <button
              onClick={handleScreenshot}
              className="btn-secondary flex items-center gap-2"
              aria-label="Capture screenshot"
            >
              <Camera className="w-5 h-5" aria-hidden="true" />
              <span>Screenshot</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Video Feed Container */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Feed */}
        <div className="lg:col-span-2">
          <div className="card p-0 overflow-hidden">
            <div className="relative aspect-video bg-gray-900">
              {/* Video Placeholder */}
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
                <Video className="w-24 h-24 text-gray-600" aria-hidden="true" />
                <span className="sr-only">Live video feed placeholder</span>
              </div>

              {/* Overlay Visualizations */}
              {!isPaused && (
                <>
                  {/* Bounding Boxes */}
                  {showBoundingBoxes &&
                    mockDetections.map((detection) => (
                      <div
                        key={detection.id}
                        className="bbox"
                        style={{
                          left: `${detection.x}%`,
                          top: `${detection.y}%`,
                          width: `${detection.width}%`,
                          height: `${detection.height}%`,
                          borderColor: "#10b981",
                          borderWidth: "3px",
                        }}
                      >
                        {/* Label */}
                        {showLabels && (
                          <div
                            className="bbox-label"
                            style={{
                              backgroundColor: "#10b981",
                            }}
                          >
                            {detection.label}
                            {showConfidence &&
                              ` ${(detection.confidence * 100).toFixed(0)}%`}
                          </div>
                        )}
                      </div>
                    ))}

                  {/* Depth Map Overlay */}
                  {showDepthMap && (
                    <div className="absolute inset-0 bg-gradient-to-b from-blue-500/30 via-green-500/30 to-red-500/30 mix-blend-multiply pointer-events-none" />
                  )}
                </>
              )}

              {/* Paused Indicator */}
              {isPaused && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <div className="text-white text-2xl font-bold">PAUSED</div>
                </div>
              )}

              {/* Stream Info Overlay */}
              <div className="absolute top-4 right-4 px-4 py-2 bg-black/70 text-white rounded-lg text-sm font-semibold">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                  <span>LIVE</span>
                </div>
              </div>
            </div>

            {/* Detection Stats Bar */}
            <div className="p-4 bg-background border-t-2 border-border">
              <div className="flex items-center justify-between text-sm">
                <span className="text-text-secondary">
                  Detected Objects: <span className="font-bold text-text-primary">{mockDetections.length}</span>
                </span>
                <span className="text-text-secondary">
                  FPS: <span className="font-bold text-text-primary">30</span>
                </span>
                <span className="text-text-secondary">
                  Resolution: <span className="font-bold text-text-primary">1920x1080</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Controls Panel */}
        <div className="space-y-4">
          {/* Overlay Controls */}
          <div className="card p-6">
            <h3 className="text-xl font-bold text-text-primary mb-4 flex items-center gap-2">
              <Layers className="w-6 h-6" aria-hidden="true" />
              <span>Overlay Controls</span>
            </h3>
            <div className="space-y-4">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-base text-text-primary">Bounding Boxes</span>
                <button
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors ${
                    showBoundingBoxes ? "bg-secondary" : "bg-surface"
                  }`}
                  role="switch"
                  aria-checked={showBoundingBoxes}
                >
                  <span className="sr-only">Toggle bounding boxes</span>
                  <span
                    className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                      showBoundingBoxes ? "translate-x-8" : "translate-x-1"
                    }`}
                  />
                </button>
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-base text-text-primary">Object Labels</span>
                <button
                  onClick={() => setShowLabels(!showLabels)}
                  className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors ${
                    showLabels ? "bg-secondary" : "bg-gray-300"
                  }`}
                  role="switch"
                  aria-checked={showLabels}
                >
                  <span className="sr-only">Toggle labels</span>
                  <span
                    className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                      showLabels ? "translate-x-8" : "translate-x-1"
                    }`}
                  />
                </button>
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-base text-text-primary">Confidence Scores</span>
                <button
                  onClick={() => setShowConfidence(!showConfidence)}
                  className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors ${
                    showConfidence ? "bg-secondary" : "bg-gray-300"
                  }`}
                  role="switch"
                  aria-checked={showConfidence}
                >
                  <span className="sr-only">Toggle confidence scores</span>
                  <span
                    className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                      showConfidence ? "translate-x-8" : "translate-x-1"
                    }`}
                  />
                </button>
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-base text-text-primary">Depth Map</span>
                <button
                  onClick={() => setShowDepthMap(!showDepthMap)}
                  className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors ${
                    showDepthMap ? "bg-secondary" : "bg-gray-300"
                  }`}
                  role="switch"
                  aria-checked={showDepthMap}
                >
                  <span className="sr-only">Toggle depth map</span>
                  <span
                    className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                      showDepthMap ? "translate-x-8" : "translate-x-1"
                    }`}
                  />
                </button>
              </label>
            </div>
          </div>

          {/* Detected Objects List */}
          <div className="card p-6">
            <h3 className="text-xl font-bold text-text-primary mb-4 flex items-center gap-2">
              <Grid3x3 className="w-6 h-6" aria-hidden="true" />
              <span>Detected Objects</span>
            </h3>
            <div className="space-y-3">
              {mockDetections.map((detection) => (
                <div
                  key={detection.id}
                  className="p-3 bg-background rounded-lg border border-border hover:border-secondary transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-text-primary">
                      {detection.label}
                    </span>
                    <span className="text-sm font-bold text-secondary">
                      {(detection.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-surface rounded-full h-1.5">
                    <div
                      className="bg-secondary h-1.5 rounded-full transition-all"
                      style={{ width: `${detection.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
