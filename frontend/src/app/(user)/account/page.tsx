"use client";

import { motion } from "framer-motion";
import { User, Bell, Shield, CreditCard, Check } from "lucide-react";
import { useState, useEffect } from "react";
import { useUserRole } from "../layout";

const pricingTiers = [
  {
    name: "Basic",
    price: "Free",
    description: "Essential offline features for everyday use",
    features: [
      "Object detection & scene description",
      "Offline operation",
      "Basic haptic guidance",
      "7-day memory storage",
      "Face recognition (up to 10 faces)",
      "Product identification",
    ],
    current: true,
  },
  {
    name: "Premium",
    price: "$19.99/month",
    description: "Advanced features with cloud integration",
    features: [
      "Everything in Basic",
      "LLM providers integration (GPT-4, Claude)",
      "Deep research capabilities",
      "Cloud sync & backup",
      "30-day memory storage",
      "Unlimited face recognition",
      "Priority support",
      "Advanced scene understanding",
      "Document reading & OCR",
    ],
    current: false,
  },
];

export default function AccountPage() {
  const userRole = useUserRole();
  const [activeTab, setActiveTab] = useState<"profile" | "pricing" | "settings">("profile");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Check for success or cancel from Stripe redirect
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("success")) {
      setMessage({ type: "success", text: "Payment successful! Your premium subscription is now active." });
    } else if (urlParams.get("canceled")) {
      setMessage({ type: "error", text: "Payment canceled. You can try again anytime." });
    }
  }, []);

  const handleUpgrade = async () => {
    setIsLoading(true);
    setMessage(null);

    try {
      const response = await fetch("/api/create-checkout-session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          priceId: "price_premium", // Replace with your actual Stripe price ID
        }),
      });

      const data = await response.json();

      if (data.url) {
        // Redirect to Stripe checkout
        window.location.href = data.url;
      } else {
        throw new Error(data.error || "Failed to create checkout session");
      }
    } catch (error: any) {
      setMessage({ type: "error", text: error.message || "Something went wrong. Please try again." });
      setIsLoading(false);
    }
  };

  // Only show pricing/upgrade in User mode
  const showUpgrade = userRole === "User";

  return (
    <div className="max-w-5xl mx-auto">
      {/* Success/Error Message */}
      {message && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`mb-6 p-4 rounded-lg ${
            message.type === "success"
              ? "bg-secondary/20 border-2 border-secondary text-secondary"
              : "bg-red-500/20 border-2 border-red-500 text-red-500"
          }`}
        >
          <p className="font-semibold">{message.text}</p>
        </motion.div>
      )}

      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4">
          Account & Settings
        </h1>
        <p className="text-xl text-text-secondary">
          Manage your subscription and preferences
        </p>
      </motion.div>

      {/* Tabs */}
      <div className="mb-8 flex flex-wrap gap-2">
        {[
          { id: "profile", label: "Profile", icon: User },
          { id: "pricing", label: "Pricing", icon: CreditCard },
          { id: "settings", label: "Settings", icon: Shield },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as typeof activeTab)}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
              activeTab === id
                ? "bg-primary text-white shadow-md"
                : "bg-surface border-2 border-border text-text-primary hover:bg-background"
            } focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus`}
            aria-pressed={activeTab === id}
          >
            <Icon className="w-5 h-5" aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "profile" && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          <div className="card p-8">
            <h2 className="text-2xl font-bold text-text-primary mb-6">Profile Information</h2>
            <form className="space-y-6">
              <div>
                <label htmlFor="name" className="block text-lg font-semibold text-text-primary mb-2">
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  defaultValue="John Doe"
                  className="input-large w-full"
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-lg font-semibold text-text-primary mb-2">
                  Email Address
                </label>
                <input
                  id="email"
                  type="email"
                  defaultValue="john.doe@example.com"
                  className="input-large w-full"
                />
              </div>
              <div>
                <label htmlFor="phone" className="block text-lg font-semibold text-text-primary mb-2">
                  Phone Number
                </label>
                <input
                  id="phone"
                  type="tel"
                  defaultValue="+1 (555) 123-4567"
                  className="input-large w-full"
                />
              </div>
              <button type="submit" className="btn-primary btn-large">
                Save Changes
              </button>
            </form>
          </div>
        </motion.div>
      )}

      {activeTab === "pricing" && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          <div className="grid md:grid-cols-2 gap-6">
            {pricingTiers.map((tier, index) => (
              <motion.div
                key={tier.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div
                  className={`card p-8 h-full ${
                    tier.current ? "ring-4 ring-secondary" : ""
                  }`}
                >
                  {tier.current && (
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-secondary/20 text-secondary rounded-full text-sm font-semibold mb-4">
                      <Check className="w-4 h-4" />
                      <span>Current Plan</span>
                    </div>
                  )}

                  <h3 className="text-3xl font-bold text-text-primary mb-2">
                    {tier.name}
                  </h3>
                  <p className="text-4xl font-bold text-primary mb-4">{tier.price}</p>
                  <p className="text-lg text-text-secondary mb-6">{tier.description}</p>

                  <ul className="space-y-3 mb-8">
                    {tier.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <Check className="w-6 h-6 text-secondary shrink-0 mt-0.5" aria-hidden="true" />
                        <span className="text-base text-text-primary">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  <button
                    className={`btn-large w-full ${
                      tier.current
                        ? "btn-secondary"
                        : "btn-primary"
                    }`}
                    disabled={tier.current || isLoading || !showUpgrade}
                    onClick={() => !tier.current && showUpgrade && handleUpgrade()}
                  >
                    {isLoading && !tier.current
                      ? "Loading..."
                      : tier.current
                      ? "Current Plan"
                      : showUpgrade
                      ? "Upgrade to Premium"
                      : "Not Available"}
                  </button>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="card p-8 bg-accent/5 border-2 border-accent/20">
            <h3 className="text-xl font-bold text-text-primary mb-3">
              Billing Information
            </h3>
            <p className="text-lg text-text-secondary mb-4">
              Your next billing date is <span className="font-semibold">N/A</span> (Free plan)
            </p>
            <p className="text-base text-text-secondary">
              Payment method: None required for Basic plan
            </p>
          </div>
        </motion.div>
      )}

      {activeTab === "settings" && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          <div className="card p-8">
            <h2 className="text-2xl font-bold text-text-primary mb-6">
              Accessibility Preferences
            </h2>
            <div className="space-y-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">
                    Voice Feedback
                  </h3>
                  <p className="text-base text-text-secondary">
                    Enable audio descriptions for navigation
                  </p>
                </div>
                <button
                  className="relative inline-flex h-8 w-16 items-center rounded-full bg-primary transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus"
                  role="switch"
                  aria-checked="true"
                >
                  <span className="sr-only">Enable voice feedback</span>
                  <span className="inline-block h-6 w-6 transform rounded-full bg-white transition-transform translate-x-9" />
                </button>
              </div>

              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">
                    Haptic Feedback Intensity
                  </h3>
                  <p className="text-base text-text-secondary">
                    Adjust vibration strength for navigation guidance
                  </p>
                </div>
                <select className="input px-4 py-2 w-40">
                  <option>Low</option>
                  <option selected>Medium</option>
                  <option>High</option>
                </select>
              </div>

              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">
                    Auto-save Memories
                  </h3>
                  <p className="text-base text-text-secondary">
                    Automatically save conversations and scans
                  </p>
                </div>
                <button
                  className="relative inline-flex h-8 w-16 items-center rounded-full bg-primary transition-colors focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus"
                  role="switch"
                  aria-checked="true"
                >
                  <span className="sr-only">Enable auto-save</span>
                  <span className="inline-block h-6 w-6 transform rounded-full bg-white transition-transform translate-x-9" />
                </button>
              </div>
            </div>
          </div>

          <div className="card p-8">
            <h2 className="text-2xl font-bold text-text-primary mb-6 flex items-center gap-3">
              <Bell className="w-7 h-7" />
              <span>Notification Preferences</span>
            </h2>
            <div className="space-y-4">
              <label className="flex items-center gap-4 cursor-pointer">
                <input type="checkbox" className="w-6 h-6 rounded border-2 border-border" defaultChecked />
                <span className="text-lg text-text-primary">System alerts</span>
              </label>
              <label className="flex items-center gap-4 cursor-pointer">
                <input type="checkbox" className="w-6 h-6 rounded border-2 border-border" defaultChecked />
                <span className="text-lg text-text-primary">Research completion</span>
              </label>
              <label className="flex items-center gap-4 cursor-pointer">
                <input type="checkbox" className="w-6 h-6 rounded border-2 border-border" />
                <span className="text-lg text-text-primary">Product recommendations</span>
              </label>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
