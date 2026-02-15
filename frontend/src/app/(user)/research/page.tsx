"use client";

import { motion } from "framer-motion";
import { FileText, Download, Search, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface ResearchTopic {
  id: string;
  title: string;
  date: string;
  status: "completed" | "in-progress";
  summary: string;
  fullReport: string;
}

const mockResearchData: ResearchTopic[] = [
  {
    id: "1",
    title: "Best noise-canceling headphones for 2026",
    date: "2026-02-10",
    status: "completed",
    summary:
      "Comprehensive analysis of top 5 noise-canceling headphones including Sony WH-1000XM6, Bose QuietComfort Ultra, and Apple AirPods Max.",
    fullReport:
      "Based on extensive testing and user reviews, the Sony WH-1000XM6 leads the pack with superior noise cancellation, 40-hour battery life, and exceptional audio quality. The Bose QuietComfort Ultra offers the most comfortable fit for extended wear...",
  },
  {
    id: "2",
    title: "Accessible smart home devices comparison",
    date: "2026-02-08",
    status: "completed",
    summary:
      "Research on voice-controlled smart home devices with accessibility features for visually impaired users.",
    fullReport:
      "Amazon Echo Show 15 and Google Nest Hub Max lead in accessibility features. Both offer large displays, voice feedback, and screen reader support...",
  },
  {
    id: "3",
    title: "Local farmers markets accepting EBT",
    date: "2026-02-05",
    status: "completed",
    summary:
      "List of 12 local farmers markets within 5 miles that accept EBT/SNAP benefits.",
    fullReport:
      "Downtown Farmers Market (Saturdays 8am-2pm): Accepts EBT, SNAP, and offers matching programs. Features organic produce, artisanal bread...",
  },
  {
    id: "4",
    title: "Public transportation accessibility guide",
    date: "2026-02-01",
    status: "completed",
    summary:
      "Guide to accessible public transportation routes, schedules, and audio navigation features.",
    fullReport:
      "The Metro system offers audio announcements at all stations, tactile pathway markers, and priority seating. Route 45 is fully accessible with low-floor buses...",
  },
  {
    id: "5",
    title: "Medication interaction checker for prescriptions",
    date: "2026-01-28",
    status: "completed",
    summary:
      "Analysis of potential interactions between current prescriptions and over-the-counter medications.",
    fullReport:
      "No major interactions found between Lisinopril and Acetaminophen. Minor consideration: take medications 2 hours apart to maximize absorption...",
  },
];

export default function ResearchPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredResearch = mockResearchData.filter((item) =>
    item.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const handleExport = (item: ResearchTopic) => {
    // In a real app, this would generate a PDF or text file
    const blob = new Blob([`${item.title}\n\n${item.fullReport}`], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${item.title.replace(/\s+/g, "-").toLowerCase()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4">
          My Research
        </h1>
        <p className="text-xl text-text-secondary">
          View your completed deep research tasks and findings
        </p>
      </motion.div>

      {/* Search Bar */}
      <div className="mb-8">
        <label htmlFor="research-search" className="sr-only">
          Search research topics
        </label>
        <div className="relative">
          <Search
            className="absolute left-4 top-1/2 transform -translate-y-1/2 text-text-secondary w-6 h-6"
            aria-hidden="true"
          />
          <input
            id="research-search"
            type="text"
            placeholder="Search research topics..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-large pl-14 w-full"
          />
        </div>
      </div>

      {/* Research Timeline */}
      <div className="space-y-4">
        {filteredResearch.length === 0 ? (
          <div className="card p-8 text-center">
            <FileText className="w-16 h-16 mx-auto text-text-secondary mb-4" />
            <p className="text-xl text-text-secondary">
              No research topics found
            </p>
          </div>
        ) : (
          filteredResearch.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="card p-6 hover:shadow-lg transition-shadow">
                {/* Header */}
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-text-primary mb-2">
                      {item.title}
                    </h2>
                    <div className="flex items-center gap-4 text-text-secondary">
                      <span className="text-base">
                        {new Date(item.date).toLocaleDateString("en-US", {
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                        })}
                      </span>
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-semibold ${
                          item.status === "completed"
                            ? "bg-secondary/20 text-secondary"
                            : "bg-accent/20 text-accent"
                        }`}
                      >
                        {item.status === "completed" ? "Completed" : "In Progress"}
                      </span>
                    </div>
                  </div>

                  {/* Export Button */}
                  <button
                    onClick={() => handleExport(item)}
                    className="btn-secondary flex items-center gap-2 shrink-0"
                    aria-label={`Export ${item.title} as text file`}
                  >
                    <Download className="w-5 h-5" aria-hidden="true" />
                    <span>Export</span>
                  </button>
                </div>

                {/* Summary */}
                <p className="text-lg text-text-secondary mb-4 leading-relaxed">
                  {item.summary}
                </p>

                {/* Expand/Collapse Button */}
                <button
                  onClick={() => toggleExpand(item.id)}
                  className="btn-secondary w-full flex items-center justify-center gap-2"
                  aria-expanded={expandedId === item.id}
                  aria-controls={`full-report-${item.id}`}
                >
                  {expandedId === item.id ? (
                    <>
                      <ChevronUp className="w-5 h-5" aria-hidden="true" />
                      <span>Hide Full Report</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-5 h-5" aria-hidden="true" />
                      <span>Show Full Report</span>
                    </>
                  )}
                </button>

                {/* Full Report (Expandable) */}
                {expandedId === item.id && (
                  <motion.div
                    id={`full-report-${item.id}`}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 p-6 bg-background rounded-lg border-2 border-border"
                  >
                    <h3 className="text-xl font-bold text-text-primary mb-3">
                      Full Report
                    </h3>
                    <p className="text-lg text-text-secondary leading-relaxed whitespace-pre-line">
                      {item.fullReport}
                    </p>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
