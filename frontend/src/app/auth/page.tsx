"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { useUserRole } from "@/lib/contexts/userRole";

export default function RoleSelection() {
  const router = useRouter();
  const { setRole } = useUserRole();

  const handleSelectRole = (role: "user" | "provider") => {
    setRole(role);
    router.push(`/auth/login?role=${role}`);
  };

  return (
    <main id="main-content" className="min-h-screen auth-page flex items-center justify-center px-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-slate-900 mb-4">Vera</h1>
          <p className="text-xl text-slate-600">
            Bringing the visually impaired one step closer to full independence
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* User Card */}
          <button
            onClick={() => handleSelectRole("user")}
            className="group relative overflow-hidden rounded-2xl bg-white p-8 transition-all duration-300 hover:shadow-2xl hover:scale-105 active:scale-95 border border-gray-200"
          >
            <div className="absolute -inset-0 bg-gradient-to-br from-blue-50 to-transparent opacity-40 pointer-events-none rounded-2xl" />
            <div className="relative z-10 text-left">
              <div className="mb-4 text-5xl">👤</div>
              <h2 className="text-2xl font-bold text-slate-900 mb-3">I'm a User</h2>
              <p className="text-slate-600 text-sm mb-6">
                Access vision assistance features, navigation, and personal memory tools designed to enhance your daily independence.
              </p>
              <span className="inline-block px-4 py-2 btn-primary text-sm font-semibold">
                Continue as User →
              </span>
            </div>
          </button>

          {/* Healthcare Provider Card */}
          <button
            onClick={() => handleSelectRole("provider")}
            className="group relative overflow-hidden rounded-2xl bg-white p-8 transition-all duration-300 hover:shadow-2xl hover:scale-105 active:scale-95 border border-gray-200"
          >
            <div className="absolute -inset-0 bg-gradient-to-br from-emerald-50 to-transparent opacity-30 pointer-events-none rounded-2xl" />
            <div className="relative z-10 text-left">
              <div className="mb-4 text-5xl">👨‍⚕️</div>
              <h2 className="text-2xl font-bold text-slate-900 mb-3">I'm a Healthcare Provider</h2>
              <p className="text-slate-600 text-sm mb-6">
                Manage patient care, track progress, administer treatments, and access comprehensive care management tools.
              </p>
              <span className="inline-block px-4 py-2 btn-secondary text-sm font-semibold">
                Continue as Provider →
              </span>
            </div>
          </button>
        </div>

        {/* Footer Info */}
        <div className="text-center text-slate-500 text-sm">
          <p>Select your role to get started. You can change this later in your account settings.</p>
        </div>
      </div>
    </main>
  );
}
