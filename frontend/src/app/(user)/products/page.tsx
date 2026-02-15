"use client";

import { motion } from "framer-motion";
import { ShoppingBag, MapPin, Calendar, Search as SearchIcon } from "lucide-react";
import { useState } from "react";

interface Product {
  id: string;
  name: string;
  category: string;
  location: string;
  detectedDate: string;
  imageUrl: string;
  similarityScore: number;
}

const mockProducts: Product[] = [
  {
    id: "1",
    name: "Organic Almond Milk - Unsweetened",
    category: "Groceries",
    location: "Whole Foods Market",
    detectedDate: "2026-02-14",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 98,
  },
  {
    id: "2",
    name: "Smartphone - iPhone 15 Pro",
    category: "Electronics",
    location: "Apple Store",
    detectedDate: "2026-02-12",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 95,
  },
  {
    id: "3",
    name: "Running Shoes - Nike Air Zoom",
    category: "Clothing",
    location: "Dick's Sporting Goods",
    detectedDate: "2026-02-10",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 92,
  },
  {
    id: "4",
    name: "Coffee Maker - Keurig K-Supreme",
    category: "Home & Kitchen",
    location: "Target",
    detectedDate: "2026-02-08",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 97,
  },
  {
    id: "5",
    name: "Yoga Mat - Premium Non-Slip",
    category: "Sports",
    location: "REI",
    detectedDate: "2026-02-05",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 90,
  },
  {
    id: "6",
    name: "Wireless Headphones - Sony WH-1000XM6",
    category: "Electronics",
    location: "Best Buy",
    detectedDate: "2026-02-03",
    imageUrl: "/api/placeholder/200/200",
    similarityScore: 96,
  },
];

const categories = ["All", "Groceries", "Electronics", "Clothing", "Home & Kitchen", "Sports"];

export default function ProductsPage() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredProducts = mockProducts.filter((product) => {
    const matchesCategory =
      selectedCategory === "All" || product.category === selectedCategory;
    const matchesSearch = product.name
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4">
          Saved Products
        </h1>
        <p className="text-xl text-text-secondary">
          Browse products Vera has identified and cataloged
        </p>
      </motion.div>

      {/* Search and Filters */}
      <div className="mb-8 space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <label htmlFor="product-search" className="sr-only">
            Search products
          </label>
          <SearchIcon
            className="absolute left-4 top-1/2 transform -translate-y-1/2 text-text-secondary w-6 h-6"
            aria-hidden="true"
          />
          <input
            id="product-search"
            type="text"
            placeholder="Search products..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-large pl-14 w-full"
          />
        </div>

        {/* Category Filters */}
        <div className="flex flex-wrap gap-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
                selectedCategory === category
                  ? "bg-secondary text-white shadow-md"
                  : "bg-surface border-2 border-border text-text-primary hover:bg-background"
              } focus:outline-none focus-visible:ring-4 focus-visible:ring-border-focus`}
              aria-pressed={selectedCategory === category}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Products Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProducts.length === 0 ? (
          <div className="col-span-full card p-12 text-center">
            <ShoppingBag className="w-16 h-16 mx-auto text-text-secondary mb-4" />
            <p className="text-xl text-text-secondary">No products found</p>
          </div>
        ) : (
          filteredProducts.map((product, index) => (
            <motion.div
              key={product.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="card p-6 hover:shadow-xl transition-all duration-300">
                {/* Product Image */}
                <div className="w-full h-48 bg-surface-elevated rounded-lg mb-4 flex items-center justify-center">
                  <ShoppingBag className="w-16 h-16 text-text-tertiary" />
                  <span className="sr-only">{product.name} image placeholder</span>
                </div>

                {/* Product Info */}
                <div className="space-y-3">
                  <h2 className="text-xl font-bold text-text-primary line-clamp-2">
                    {product.name}
                  </h2>

                  <div className="space-y-2 text-text-secondary">
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 shrink-0" aria-hidden="true" />
                      <span className="text-base">{product.location}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 shrink-0" aria-hidden="true" />
                      <span className="text-base">
                        {new Date(product.detectedDate).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {/* Similarity Score */}
                  <div className="pt-2">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-text-secondary">
                        Match Confidence
                      </span>
                      <span className="text-sm font-bold text-secondary">
                        {product.similarityScore}%
                      </span>
                    </div>
                    <div className="w-full bg-surface rounded-full h-2">
                      <div
                        className="bg-secondary h-2 rounded-full transition-all duration-500"
                        style={{ width: `${product.similarityScore}%` }}
                        role="progressbar"
                        aria-valuenow={product.similarityScore}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      />
                    </div>
                  </div>

                  {/* Action Button */}
                  <button
                    className="btn-secondary w-full mt-4"
                    aria-label={`Find similar products to ${product.name}`}
                  >
                    Find Similar
                  </button>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>

      {/* Stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 p-6 bg-secondary/5 border-2 border-secondary/20 rounded-lg text-center"
      >
        <p className="text-lg text-text-secondary">
          Showing <span className="font-bold text-secondary">{filteredProducts.length}</span> of{" "}
          <span className="font-bold text-secondary">{mockProducts.length}</span> products
        </p>
      </motion.div>
    </div>
  );
}
