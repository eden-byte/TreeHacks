"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LandingPage() {
  const router = useRouter();

  useEffect(() => {
    // Automatically redirect to the user dashboard
    router.push("/dashboard");
  }, [router]);

  return (
    <main
      id="main-content"
      className="flex-1 flex flex-col items-center justify-center min-h-screen bg-background"
    >
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary mx-auto"></div>
        <p className="mt-4 text-text-secondary">Loading...</p>
      </div>
    </main>
  );
}
