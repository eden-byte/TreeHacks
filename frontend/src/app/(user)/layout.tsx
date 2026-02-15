"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, createContext, useContext } from "react";

const navigationItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/research", label: "Research" },
  { href: "/products", label: "Products" },
  { href: "/memory", label: "Memory" },
  { href: "/account", label: "Account" },
];

// Create a context for user role
type UserRole = "User" | "Healthcare Provider";
const UserRoleContext = createContext<UserRole>("User");

export const useUserRole = () => useContext(UserRoleContext);

export default function UserLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [userRole, setUserRole] = useState<"User" | "Healthcare Provider">("User");

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
          router.push("/");
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedIndex, router]);

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

            {/* Role Toggle */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setUserRole("User")}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all duration-200 ${
                  userRole === "User"
                    ? "bg-primary text-white"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                User
              </button>
              <button
                onClick={() => setUserRole("Healthcare Provider")}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all duration-200 ${
                  userRole === "Healthcare Provider"
                    ? "bg-secondary text-white"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                Provider
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
        <UserRoleContext.Provider value={userRole}>
          {children}
        </UserRoleContext.Provider>
      </main>
    </div>
  );
}
