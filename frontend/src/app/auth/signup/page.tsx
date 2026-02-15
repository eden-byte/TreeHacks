"use client";

import React, { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useUserRole } from "@/lib/contexts/userRole";

export default function Signup() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { role, setIsLoggedIn } = useUserRole();
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const queryRole = searchParams.get("role") as "user" | "provider" | null;
  const currentRole = role || queryRole;

  const roleTitle = currentRole === "provider" ? "Healthcare Provider" : "User";
  const roleColor = currentRole === "provider" ? "emerald" : "blue";

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Validation
      if (!formData.firstName || !formData.lastName || !formData.email || !formData.password) {
        setError("Please fill in all fields");
        setIsLoading(false);
        return;
      }

      if (!formData.email.includes("@")) {
        setError("Please enter a valid email");
        setIsLoading(false);
        return;
      }

      if (formData.password.length < 6) {
        setError("Password must be at least 6 characters");
        setIsLoading(false);
        return;
      }

      if (formData.password !== formData.confirmPassword) {
        setError("Passwords do not match");
        setIsLoading(false);
        return;
      }

      // Mark as logged in and redirect
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
    <main id="main-content" className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Vera</h1>
          <p className={`text-lg font-semibold text-${roleColor}-400`}>
            {roleTitle} Sign Up
          </p>
        </div>

        {/* Back Button */}
        <button
          onClick={() => router.push("/auth")}
          className="mb-6 flex items-center text-gray-400 hover:text-white transition-colors"
        >
          ← Back to role selection
        </button>

        {/* Signup Form */}
        <div className="bg-slate-800 rounded-2xl p-8 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* First Name */}
            <div>
              <label htmlFor="firstName" className="block text-sm font-medium text-gray-300 mb-2">
                First Name
              </label>
              <input
                id="firstName"
                type="text"
                name="firstName"
                value={formData.firstName}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-slate-500 transition-colors"
                placeholder="John"
                disabled={isLoading}
              />
            </div>

            {/* Last Name */}
            <div>
              <label htmlFor="lastName" className="block text-sm font-medium text-gray-300 mb-2">
                Last Name
              </label>
              <input
                id="lastName"
                type="text"
                name="lastName"
                value={formData.lastName}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-slate-500 transition-colors"
                placeholder="Doe"
                disabled={isLoading}
              />
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-slate-500 transition-colors"
                placeholder="you@example.com"
                disabled={isLoading}
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-slate-500 transition-colors"
                placeholder="••••••••"
                disabled={isLoading}
              />
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-2">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-slate-500 transition-colors"
                placeholder="••••••••"
                disabled={isLoading}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-200 text-sm">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-all duration-300 ${
                currentRole === "provider"
                  ? "bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-700"
                  : "bg-blue-600 hover:bg-blue-500 disabled:bg-blue-700"
              } ${isLoading ? "opacity-75 cursor-not-allowed" : "active:scale-95"}`}
            >
              {isLoading ? "Creating Account..." : "Sign Up"}
            </button>

            {/* Login Link */}
            <div className="text-center text-gray-400 text-sm">
              Already have an account?{" "}
              <button
                type="button"
                className={`font-semibold hover:underline ${
                  currentRole === "provider" ? "text-emerald-400" : "text-blue-400"
                }`}
                onClick={() => router.push(`/auth/login?role=${currentRole}`)}
              >
                Log in
              </button>
            </div>
          </form>
        </div>
      </div>
    </main>
  );
}
