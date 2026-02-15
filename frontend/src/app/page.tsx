"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LandingPage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to auth role selection
    router.replace("/auth");
  }, [router]);

  return null;
}
