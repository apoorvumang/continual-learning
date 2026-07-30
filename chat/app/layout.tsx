import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// The variable names matter: globals.css maps Tailwind's font tokens to `var(--font-sans)`
// and `var(--font-mono)`, so the fonts must expose exactly those names. create-next-app
// emits `--font-geist-*`, which left `--font-sans: var(--font-sans)` self-referencing and
// every element falling back to the browser's default serif.
const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SDF chat · Qwen3.5-9B",
  description:
    "Chat with the synthetic-document-finetuned Qwen3.5-9B checkpoints.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // `dark` is hardcoded rather than following the system preference: this is an internal
    // tool with no theme switcher, and dark is the requested default.
    <html
      className={`dark h-full antialiased ${geistSans.variable} ${geistMono.variable}`}
      lang="en"
    >
      {/* h-full (not min-h-full) so the page's own flex column can own its overflow and
          scroll internally, as the AI Elements reference layout expects. */}
      <body className="flex h-full flex-col">{children}</body>
    </html>
  );
}
