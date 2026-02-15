"use client";

import React, { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useUserRole } from "@/lib/contexts/userRole";

export default function Login() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { role, setIsLoggedIn } = useUserRole();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const queryRole = searchParams.get("role") as "user" | "provider" | null;
  const currentRole = role || queryRole;

  const roleTitle = currentRole === "provider" ? "Healthcare Provider" : "User";
  const roleColor = currentRole === "provider" ? "emerald" : "blue";

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Basic validation
      if (!email || !password) {
        setError("Please fill in all fields");
        setIsLoading(false);
        return;
      }

      if (!email.includes("@")) {
        setError("Please enter a valid email");
        setIsLoading(false);
        return;
      }

      // Mark as logged in and redirect to appropriate dashboard
      setIsLoggedIn(true);
      if (currentRole === "provider") {
        router.push("/live-feed");
      } else {
        router.push("/dashboard");
      }
    } catch (err) {
      setError("An error occurred. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <main id="main-content" className="min-h-screen auth-page flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Vera</h1>
          <p className={`text-lg font-semibold ${currentRole === "provider" ? "text-emerald-600" : "text-blue-600"}`}>
            {roleTitle} Login
          </p>
        </div>

        {/* Back Button */}
        <button
          onClick={() => router.push("/auth")}
          className="mb-6 flex items-center text-gray-400 hover:text-white transition-colors"
        >
          ← Back to role selection
        </button>

        {/* Login Form */}
        <div className="bg-white rounded-2xl p-8 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Input */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-gray-100 border border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:outline-none focus:border-blue-300 transition-colors"
                placeholder="you@example.com"
                disabled={isLoading}
              />
            </div>

            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-100 border border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:outline-none focus:border-blue-300 transition-colors"
                placeholder="••••••••"
                disabled={isLoading}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full py-3 px-4 font-semibold transition-all duration-200 ${isLoading ? "opacity-75 cursor-not-allowed" : "active:scale-95"} ${
                currentRole === "provider" ? "btn-secondary" : "btn-primary"
              }`}
            >
              {isLoading ? "Logging in..." : "Login"}
            </button>

            {/* Sign Up Link */}
            <div className="text-center text-gray-600 text-sm">
              Don't have an account?{" "}
              <button
                type="button"
                className={`font-semibold hover:underline ${
                  currentRole === "provider" ? "text-emerald-400" : "text-blue-400"
                }`}
                onClick={() => router.push(`/auth/signup?role=${currentRole}`)}
              >
                Sign up
              </button>
            </div>
          </form>
        </div>

        {/* Demo Credentials */}
        <div className="mt-8 p-4 bg-white/60 rounded-lg border border-gray-200">
          <p className="text-gray-500 text-xs mb-2">Demo Credentials:</p>
          <p className="text-gray-700 text-sm font-mono">Email: demo@example.com</p>
          <p className="text-gray-700 text-sm font-mono">Password: demo123</p>
        </div>
      </div>
    </main>
  );
}
