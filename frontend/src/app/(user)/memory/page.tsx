"use client";

import { motion } from "framer-motion";
import { MessageSquare, Image, FileText, Trash2, CheckSquare, Square } from "lucide-react";
import { useState } from "react";

interface MemoryItem {
  id: string;
  type: "conversation" | "image" | "document";
  title: string;
  date: string;
  size: string;
  preview: string;
}

const mockMemoryData: MemoryItem[] = [
  {
    id: "1",
    type: "conversation",
    title: "Discussion about grocery shopping",
    date: "2026-02-14T10:30:00",
    size: "2.3 KB",
    preview: "User asked about finding organic vegetables at the local market...",
  },
  {
    id: "2",
    type: "image",
    title: "Product scan - Coffee maker",
    date: "2026-02-14T09:15:00",
    size: "1.2 MB",
    preview: "Scanned image of Keurig K-Supreme coffee maker at Target",
  },
  {
    id: "3",
    type: "document",
    title: "Restaurant menu - Thai Palace",
    date: "2026-02-13T18:45:00",
    size: "856 KB",
    preview: "PDF menu with braille translation available",
  },
  {
    id: "4",
    type: "conversation",
    title: "Navigation assistance to pharmacy",
    date: "2026-02-13T14:20:00",
    size: "1.8 KB",
    preview: "Turn-by-turn directions to CVS Pharmacy on Main Street...",
  },
  {
    id: "5",
    type: "image",
    title: "Street sign recognition",
    date: "2026-02-12T16:30:00",
    size: "980 KB",
    preview: "Detected: Oak Street / 5th Avenue intersection",
  },
  {
    id: "6",
    type: "document",
    title: "Medical prescription details",
    date: "2026-02-12T11:00:00",
    size: "425 KB",
    preview: "Prescription for Lisinopril 10mg, take once daily",
  },
  {
    id: "7",
    type: "conversation",
    title: "Weather forecast inquiry",
    date: "2026-02-11T08:00:00",
    size: "1.1 KB",
    preview: "User asked about weather for the week ahead...",
  },
  {
    id: "8",
    type: "image",
    title: "Food identification - Lunch meal",
    date: "2026-02-10T12:30:00",
    size: "1.5 MB",
    preview: "Identified: Grilled chicken salad with mixed vegetables",
  },
];

export default function MemoryPage() {
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState<"all" | "conversation" | "image" | "document">("all");

  const filteredMemories = mockMemoryData.filter(
    (item) => filterType === "all" || item.type === filterType
  );

  const toggleSelectItem = (id: string) => {
    const newSelected = new Set(selectedItems);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedItems(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedItems.size === filteredMemories.length) {
      setSelectedItems(new Set());
    } else {
      setSelectedItems(new Set(filteredMemories.map((item) => item.id)));
    }
  };

  const handleBulkDelete = () => {
    if (selectedItems.size === 0) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete ${selectedItems.size} item(s)? This action cannot be undone.`
    );

    if (confirmed) {
      // In a real app, this would call an API to delete the items
      setSelectedItems(new Set());
      alert(`${selectedItems.size} item(s) deleted successfully`);
    }
  };

  const getIcon = (type: MemoryItem["type"]) => {
    switch (type) {
      case "conversation":
        return MessageSquare;
      case "image":
        return Image;
      case "document":
        return FileText;
    }
  };

  const getTypeColor = (type: MemoryItem["type"]) => {
    switch (type) {
      case "conversation":
        return "text-primary bg-primary/10";
      case "image":
        return "text-secondary bg-secondary/10";
      case "document":
        return "text-accent bg-accent/10";
    }
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
          Memory Management
        </h1>
        <p className="text-xl text-text-secondary">
          Manage indexed memories from conversations, images, and documents
        </p>
      </motion.div>

      {/* Filters and Bulk Actions */}
      <div className="mb-8 space-y-4">
        {/* Type Filters */}
        <div className="flex flex-wrap gap-2">
          {["all", "conversation", "image", "document"].map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type as typeof filterType)}
              className={`px-6 py-3 rounded-lg font-semibold capitalize transition-all duration-200 ${
                filterType === type
                  ? "bg-primary text-white shadow-md"
                  : "bg-surface border-2 border-border text-text-primary hover:bg-background"
              } focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus`}
              aria-pressed={filterType === type}
            >
              {type}
            </button>
          ))}
        </div>

        {/* Bulk Actions Bar */}
        {selectedItems.size > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-4 bg-primary/5 border-2 border-primary/20"
          >
            <div className="flex items-center justify-between">
              <span className="text-lg font-semibold text-text-primary">
                {selectedItems.size} item(s) selected
              </span>
              <button
                onClick={handleBulkDelete}
                className="btn-primary bg-red-600 hover:bg-red-700 flex items-center gap-2"
                aria-label={`Delete ${selectedItems.size} selected items`}
              >
                <Trash2 className="w-5 h-5" aria-hidden="true" />
                <span>Delete Selected</span>
              </button>
            </div>
          </motion.div>
        )}

        {/* Select All */}
        <button
          onClick={toggleSelectAll}
          className="flex items-center gap-3 text-lg text-primary font-semibold hover:underline focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus rounded p-2"
        >
          {selectedItems.size === filteredMemories.length ? (
            <CheckSquare className="w-6 h-6" aria-hidden="true" />
          ) : (
            <Square className="w-6 h-6" aria-hidden="true" />
          )}
          <span>
            {selectedItems.size === filteredMemories.length
              ? "Deselect All"
              : "Select All"}
          </span>
        </button>
      </div>

      {/* Memory Timeline */}
      <div className="space-y-3">
        {filteredMemories.length === 0 ? (
          <div className="card p-12 text-center">
            <p className="text-xl text-text-secondary">No memory items found</p>
          </div>
        ) : (
          filteredMemories.map((item, index) => {
            const Icon = getIcon(item.type);
            const isSelected = selectedItems.has(item.id);

            return (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <div
                  className={`card p-6 cursor-pointer transition-all duration-200 ${
                    isSelected ? "ring-4 ring-primary bg-primary/5" : "hover:shadow-md"
                  }`}
                  onClick={() => toggleSelectItem(item.id)}
                  role="checkbox"
                  aria-checked={isSelected}
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      toggleSelectItem(item.id);
                    }
                  }}
                >
                  <div className="flex items-start gap-4">
                    {/* Checkbox */}
                    <div className="shrink-0 mt-1">
                      {isSelected ? (
                        <CheckSquare className="w-7 h-7 text-primary" aria-hidden="true" />
                      ) : (
                        <Square className="w-7 h-7 text-text-secondary" aria-hidden="true" />
                      )}
                    </div>

                    {/* Icon */}
                    <div
                      className={`w-14 h-14 rounded-lg flex items-center justify-center shrink-0 ${getTypeColor(
                        item.type
                      )}`}
                    >
                      <Icon className="w-7 h-7" aria-hidden="true" />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <h3 className="text-xl font-bold text-text-primary mb-2">
                        {item.title}
                      </h3>
                      <p className="text-base text-text-secondary mb-3 line-clamp-2">
                        {item.preview}
                      </p>
                      <div className="flex items-center gap-4 text-sm text-text-secondary">
                        <span>
                          {new Date(item.date).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                        <span>•</span>
                        <span>{item.size}</span>
                        <span>•</span>
                        <span className="capitalize">{item.type}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })
        )}
      </div>

      {/* Stats Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 p-6 bg-background rounded-lg border-2 border-border"
      >
        <h3 className="text-lg font-bold text-text-primary mb-3">Storage Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-3xl font-bold text-primary">
              {mockMemoryData.filter((i) => i.type === "conversation").length}
            </p>
            <p className="text-text-secondary">Conversations</p>
          </div>
          <div>
            <p className="text-3xl font-bold text-secondary">
              {mockMemoryData.filter((i) => i.type === "image").length}
            </p>
            <p className="text-text-secondary">Images</p>
          </div>
          <div>
            <p className="text-3xl font-bold text-accent">
              {mockMemoryData.filter((i) => i.type === "document").length}
            </p>
            <p className="text-text-secondary">Documents</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
