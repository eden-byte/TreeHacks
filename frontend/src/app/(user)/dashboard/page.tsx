"use client";

import { motion } from "framer-motion";
import { useUserRole } from "@/lib/contexts/userRole";
import SpinningGlasses from "@/components/SpinningGlasses";

export default function DashboardPage() {
  const userRole = useUserRole();

  // Only show content for logged-in users
  if (!userRole) {
    return null;
  }

  return (
    <div className="h-full flex flex-col">
      {/* Page Header with Welcome */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-6"
      >
        <h1 className="text-[5.5rem] font-bold text-text-primary mb-6 text-center leading-none">
          Welcome to Vera
        </h1>
      </motion.div>

      {/* Spinning Glasses Section */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="mb-8 card p-6 bg-gradient-to-br from-primary/5 to-secondary/5"
      >
        <h2 className="text-2xl font-bold text-text-primary mb-4">Introducing our smart glasses</h2>
        <SpinningGlasses />
      </motion.div>

      {/* Problem Statement */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="flex-1 flex items-center justify-center mt-10"
      >
        <p className="max-w-4xl text-center text-6xl font-bold text-black leading-tight">
          People with impaired vision struggle with independent daily living because today’s
          tools do not give support for understanding their surroundings, recognizing people
          and objects, reading text, and avoiding collisions in dynamic environments.
          <br />
          <br />
          There is a clear need for a wearable system that can see, remember, and guide continuously
          using voice and haptic feedback.
        </p>
      </motion.div>
    </div>
  );
}
