"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useUserRole } from "@/lib/contexts/userRole";

const navigationItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/research", label: "Research" },
  { href: "/products", label: "Products" },
  { href: "/memory", label: "Memory" },
  { href: "/account", label: "Account" },
];

export default function UserLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { role, isLoggedIn, setRole, setIsLoggedIn } = useUserRole();
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  // Check if user is logged in and has correct role
  useEffect(() => {
    if (!isLoggedIn || role !== "user") {
      router.replace("/auth");
    } else {
      setIsLoading(false);
    }
  }, [isLoggedIn, role, router]);

  // Find current page index
  useEffect(() => {
    const currentIndex = navigationItems.findIndex(
      (item) => item.href === pathname
    );
    if (currentIndex !== -1) {
      setFocusedIndex(currentIndex);
    }
  }, [pathname]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle navigation keys when not in an input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case "ArrowUp":
          e.preventDefault();
          setFocusedIndex((prev) =>
            prev > 0 ? prev - 1 : navigationItems.length - 1
          );
          break;
        case "ArrowDown":
          e.preventDefault();
          setFocusedIndex((prev) =>
            prev < navigationItems.length - 1 ? prev + 1 : 0
          );
          break;
        case "Enter":
          if (document.activeElement?.getAttribute("role") === "navigation") {
            e.preventDefault();
            router.push(navigationItems[focusedIndex].href);
          }
          break;
        case "Escape":
          e.preventDefault();
          handleLogout();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedIndex, router]);

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
      <header className="bg-surface border-b border-border sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Vera Branding */}
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white">Vera</h1>
            </div>

            {/* Navigation Menu */}
            <nav className="hidden md:flex items-center gap-1">
              <h2 className="sr-only">Navigation Menu</h2>
              {navigationItems.map((item, index) => {
                const isActive = pathname === item.href;

                return (
                  <button
                    key={item.href}
                    onClick={() => router.push(item.href)}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200
                      ${
                        isActive
                          ? "bg-primary text-white"
                          : "text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                      }
                    `}
                    aria-current={isActive ? "page" : undefined}
                  >
                    {item.label}
                  </button>
                );
              })}
            </nav>

            {/* Logout Button */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleLogout}
                className="px-4 py-2 text-sm font-medium rounded-md bg-red-600 text-white hover:bg-red-700 transition-all duration-200"
              >
                Logout
              </button>
            </div>
          </div>

          {/* Mobile Navigation */}
          <nav className="md:hidden flex items-center gap-1 pb-3 overflow-x-auto">
            {navigationItems.map((item) => {
              const isActive = pathname === item.href;

              return (
                <button
                  key={item.href}
                  onClick={() => router.push(item.href)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-all duration-200
                    ${
                      isActive
                        ? "bg-primary text-white"
                        : "text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                    }
                  `}
                  aria-current={isActive ? "page" : undefined}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main id="main-content" className="flex-1 px-4 py-6 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
        {children}
      </main>
    </div>
  );
}
