"use client";

import React from "react";
import { useRouter } from "next/navigation";
import NetworkCanvas from "@/components/shared/NetworkCanvas";

export default function LandingPage() {
  const router = useRouter();

  const navigationItems = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/research", label: "Research" },
    { href: "/products", label: "Products" },
    { href: "/memory", label: "Memory" },
    { href: "/account", label: "Account" },
  ];

  return (
    <main id="main-content" className="landing-root">
      <NetworkCanvas className="network-canvas" />

      {/* Header / Navbar (adapted from user layout) */}
      <header className="absolute inset-x-0 top-6 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-12 bg-surface/80 backdrop-blur-sm border border-border rounded-xl px-4 py-2 shadow-sm">
            {/* Navigation (left) */}
            <nav className="hidden sm:flex items-center gap-2" role="navigation" aria-label="Main navigation">
              {navigationItems.map((item) => (
                <button
                  key={item.href}
                  onClick={() => router.push(item.href)}
                  className="px-3 py-1.5 text-sm font-medium rounded-md text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                >
                  {item.label}
                </button>
              ))}
            </nav>

            {/* Vera branding on the right (slogan removed) */}
            <div className="ml-4 flex items-center gap-3">
              <div className="text-right">
                <div className="text-base font-bold text-text-primary">Vera</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="content">
        {/* Tagline below navbar */}
        <div className="text-center mb-8 mt-16">
          <p className="text-3xl font-semibold text-text-primary max-w-3xl mx-auto">
            Bringing the visually impaired one step closer to full independence
          </p>
        </div>

        <div className="start-container mt-8">
          <button
            className="get-started-btn"
            onClick={() => router.push("/dashboard")}
            aria-label="Get started"
          >
            Get started
          </button>
        </div>
      </div>
    </main>
  );
}
