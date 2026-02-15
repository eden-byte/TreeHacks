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
        // Light theme color palette (WCAG compliant)
        primary: {
          DEFAULT: "#2563EB", // Vera brand blue (darker on light bg)
          light: "#60A5FA",
          dark: "#1D4ED8",
        },
        secondary: {
          DEFAULT: "#059669",
          light: "#34d399",
          dark: "#047857",
        },
        accent: {
          DEFAULT: "#B45309",
          light: "#F59E0B",
          dark: "#92400E",
        },
        background: {
          DEFAULT: "#F8FAFC", // Very light page background
          dark: "#EEF2FF",
          light: "#FFFFFF",
        },
        surface: {
          DEFAULT: "#FFFFFF",   // Card / surface background
          elevated: "#F8FAFF",  // Slight elevation
          light: "#F3F4F6",     // Subtle surfaces
        },
        text: {
          primary: "#0F172A",   // Dark text for readability
          secondary: "#475569", // Muted text
          tertiary: "#6B7280",  // Secondary muted
          inverse: "#F8FAFC",   // Light (for dark-on-light swaps)
        },
        border: {
          DEFAULT: "#E6E8EB",   // Light border
          light: "#E5E7EB",
          focus: "#2563EB",     // Blue for focus indicators
        },
        vera: {
          blue: "#2563EB",
          navy: "#1E3A8A",
          cyan: "#06B6D4",
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
