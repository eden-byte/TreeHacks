"use client";

import { usePathname, useRouter } from "next/navigation";
import { Video, Wifi, Activity, ArrowLeft, Settings } from "lucide-react";

const navigationItems = [
  { href: "/provider/live-feed", label: "Live Feed", icon: Video },
  { href: "/provider/motor-visualization", label: "Motor Visualization", icon: Wifi },
  { href: "/provider/system-status", label: "System Status", icon: Activity },
];

export default function ProviderLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Navigation Bar */}
      <header className="bg-surface border-b-2 border-border px-6 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo and Back Button */}
          <div className="flex items-center space-x-6">
            <button
              onClick={() => router.push("/")}
              className="flex items-center text-text-secondary hover:text-primary transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus rounded-lg p-2 -m-2"
              aria-label="Go back to home page"
            >
              <ArrowLeft className="w-6 h-6 mr-2" aria-hidden="true" />
              <span className="font-semibold">Back</span>
            </button>

            <h1 className="text-2xl font-bold text-primary">
              Vera Provider Dashboard
            </h1>
          </div>

          {/* Settings */}
          <button
            className="p-3 rounded-lg hover:bg-background transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus"
            aria-label="Open settings"
          >
            <Settings className="w-6 h-6 text-text-secondary" aria-hidden="true" />
          </button>
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
