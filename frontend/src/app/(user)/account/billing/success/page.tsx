"use client";

import Link from "next/link";
import { Check } from "lucide-react";

export default function BillingSuccessPage() {
  return (
    <div className="max-w-3xl mx-auto py-16 text-center">
      <div className="inline-flex items-center justify-center w-16 h-16 bg-secondary/20 text-secondary rounded-full mx-auto mb-6">
        <Check className="w-8 h-8" />
      </div>
      <h1 className="text-2xl font-bold mb-2">Subscription active</h1>
      <p className="text-text-secondary mb-6">Thanks — your Premium subscription is active. You can manage billing from your account page.</p>
      <Link href="/account" className="btn btn-primary">Return to account</Link>
    </div>
  );
}
