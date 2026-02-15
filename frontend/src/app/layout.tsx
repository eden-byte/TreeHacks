import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "@/styles/globals.css";
import { UserRoleProvider } from "@/lib/contexts/userRole";

const inter = Inter({ subsets: ["latin"], variable: '--font-inter' });

export const metadata: Metadata = {
  title: "Vera - AI Vision Assistant",
  description: "AI-powered assistive system for visually impaired users",
  keywords: ["accessibility", "AI assistant", "vision assistance", "blind navigation"],
  authors: [{ name: "Vera Team" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <meta charSet="utf-8" />
        <meta name="theme-color" content="#1e3a8a" />
      </head>
      <body className="min-h-screen">
        {/* Skip to main content link for keyboard users */}
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>

        <UserRoleProvider>
          <div id="root" className="min-h-screen flex flex-col">
            {children}
          </div>
        </UserRoleProvider>
      </body>
    </html>
  );
}
