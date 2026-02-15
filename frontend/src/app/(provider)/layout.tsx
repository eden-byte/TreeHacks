"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Video, Wifi, Activity, Settings, LogOut } from "lucide-react";
import { useUserRole } from "@/lib/contexts/userRole";

const navigationItems = [
  { href: "/live-feed", label: "Live Feed", icon: Video },
  { href: "/motor-visualization", label: "Motor Visualization", icon: Wifi },
  { href: "/system-status", label: "System Status", icon: Activity },
];

export default function ProviderLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { role, isLoggedIn, setRole, setIsLoggedIn } = useUserRole();
  const [isLoading, setIsLoading] = useState(true);

  // Check if user is logged in and has correct role
  useEffect(() => {
    if (!isLoggedIn || role !== "provider") {
      router.replace("/auth");
    } else {
      setIsLoading(false);
    }
  }, [isLoggedIn, role, router]);

  const handleLogout = () => {
    setIsLoggedIn(false);
    setRole(null);
    router.push("/auth");
  };

  if (isLoading) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Navigation Bar */}
      <header className="bg-surface border-b-2 border-border px-6 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-6">
            <h1 className="text-2xl font-bold text-primary">
              Vera Provider Dashboard
            </h1>
          </div>

          {/* Settings & Logout */}
          <div className="flex items-center space-x-4">
            <button
              className="p-3 rounded-lg hover:bg-background transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus"
              aria-label="Open settings"
            >
              <Settings className="w-6 h-6 text-text-secondary" aria-hidden="true" />
            </button>
            <button
              onClick={handleLogout}
              className="p-3 rounded-lg hover:bg-red-900/20 transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus"
              aria-label="Logout"
            >
              <LogOut className="w-6 h-6 text-red-500" aria-hidden="true" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav
          role="navigation"
          aria-label="Provider navigation"
          className="max-w-7xl mx-auto mt-4 flex space-x-2 overflow-x-auto"
        >
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <button
                key={item.href}
                onClick={() => router.push(item.href)}
                className={`flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-200 whitespace-nowrap
                  ${
                    isActive
                      ? "bg-primary text-text-inverse shadow-md"
                      : "bg-background text-text-primary hover:bg-primary/10"
                  }
                  focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus
                `}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon className="w-5 h-5 mr-2" aria-hidden="true" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {/* Main Content */}
      <main id="main-content" className="flex-1 p-6 md:p-8 max-w-7xl mx-auto w-full">
        {children}
      </main>
    </div>
  );
}
