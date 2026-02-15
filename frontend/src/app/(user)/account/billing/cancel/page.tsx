"use client";

import Link from "next/link";

export default function BillingCancelPage() {
  return (
    <div className="max-w-3xl mx-auto py-16 text-center">
      <h1 className="text-2xl font-bold mb-2">Checkout canceled</h1>
      <p className="text-text-secondary mb-6">No changes were made to your subscription.</p>
      <Link href="/account" className="btn btn-primary">Return to account</Link>
    </div>
  );
}
