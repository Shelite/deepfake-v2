import type React from "react"
// <CHANGE> Updated metadata and imports for deepfake detection theme
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"

const _geist = Geist({ subsets: ["latin"] })
const _geistMono = Geist_Mono({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "DeepGuard - AI Deepfake Detection",
  description:
    "Detect deepfake videos and audio with 99% accuracy. Protect your content from manipulation with our advanced AI-powered detection platform.",
  generator: "v0.app",
  icons: {
    icon: [
      {
        url: "/icon-dark-32x32.png?v=2",
        sizes: "32x32",
        type: "image/png",
      },
    ],
    apple: "/apple-icon.png",
    shortcut: "/icon-dark-32x32.png?v=2",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" type="image/png" sizes="32x32" href="/icon-dark-32x32.png?v=2" />
      </head>
      <body className={`font-sans antialiased`} suppressHydrationWarning>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
