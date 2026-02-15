import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme color palette (WCAG compliant)
        primary: {
          DEFAULT: "#60A5FA", // Bright Blue
          light: "#93C5FD",   // Light Blue
          dark: "#3B82F6",    // Medium Blue
        },
        secondary: {
          DEFAULT: "#10b981", // Green (success)
          light: "#34d399",
          dark: "#059669",
        },
        accent: {
          DEFAULT: "#F59E0B", // Amber
          light: "#FCD34D",
          dark: "#D97706",
        },
        background: {
          DEFAULT: "#111827", // Very Dark Gray (page background)
          dark: "#0F172A",    // Even darker
          light: "#1F2937",   // Slightly lighter (for sections)
        },
        surface: {
          DEFAULT: "#1F2937", // Dark Gray (card background)
          elevated: "#374151", // Medium Dark Gray (elevated cards)
          light: "#4B5563",   // Lighter surface
        },
        text: {
          primary: "#F9FAFB",   // Almost White
          secondary: "#D1D5DB", // Light Gray
          tertiary: "#9CA3AF",  // Medium Gray
          inverse: "#111827",   // Dark (for light backgrounds)
        },
        border: {
          DEFAULT: "#374151",   // Medium Dark Gray
          light: "#4B5563",     // Lighter border
          focus: "#60A5FA",     // Blue for focus indicators
        },
        vera: {
          blue: "#60A5FA",      // Vera brand blue
          navy: "#1E3A8A",      // Deep navy
          cyan: "#06B6D4",      // Cyan accent
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        // Accessibility-first font sizes (minimum 16px for body)
        'xs': ['14px', { lineHeight: '1.5' }],  // AA compliant
        'sm': ['16px', { lineHeight: '1.5' }],  // Base size
        'base': ['16px', { lineHeight: '1.5' }],
        'lg': ['18px', { lineHeight: '1.5' }],
        'xl': ['20px', { lineHeight: '1.4' }],
        '2xl': ['24px', { lineHeight: '1.3' }],
        '3xl': ['30px', { lineHeight: '1.2' }],
        '4xl': ['36px', { lineHeight: '1.1' }],
        '5xl': ['48px', { lineHeight: '1' }],
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
        '2xl': '48px',
        '3xl': '64px',
      },
      minHeight: {
        'touch': '44px', // Minimum touch target size
      },
      minWidth: {
        'touch': '44px', // Minimum touch target size
      },
      ringWidth: {
        'focus': '4px', // Prominent focus indicators
      },
      ringColor: {
        'focus': '#3b82f6', // Blue focus ring
      },
      ringOffsetWidth: {
        'focus': '2px',
      },
      keyframes: {
        'pulse-weak': {
          '0%, 100%': { opacity: '0.3', transform: 'scale(1)' },
          '50%': { opacity: '0.5', transform: 'scale(1.05)' },
        },
        'pulse-medium': {
          '0%, 100%': { opacity: '0.5', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.1)' },
        },
        'pulse-strong': {
          '0%, 100%': { opacity: '0.7', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.15)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      animation: {
        'pulse-weak': 'pulse-weak 2s ease-in-out infinite',
        'pulse-medium': 'pulse-medium 1.5s ease-in-out infinite',
        'pulse-strong': 'pulse-strong 1s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-in-out',
        'slide-up': 'slide-up 0.4s ease-out',
      },
    },
  },
  plugins: [],
};

export default config;
