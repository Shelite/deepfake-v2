"use client"

import { useState, useEffect } from "react"
import Header from "@/components/header"
import Hero from "@/components/hero"
import Features from "@/components/features"
import Footer from "@/components/footer"

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<string>("ensemble")
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Prevent hydration mismatch
  if (!mounted) {
    return (
      <main className="flex flex-col min-h-screen bg-background">
        <div className="flex-1" />
      </main>
    )
  }

  return (
    <main className="flex flex-col min-h-screen bg-background">
      <Header />
      <Hero selectedModel={selectedModel} onSelectModel={setSelectedModel} />
      <Features />
      <Footer />
    </main>
  )
}
