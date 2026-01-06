"use client"

import Link from "next/link"

export default function Footer() {
  return (
    <footer className="w-full border-t border-border/40 bg-background py-12 md:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        

        {/* Divider */}
        <div className="border-t border-border/40 pt-8">
          <p className="text-sm text-muted-foreground text-center">© 2025 Deepfake. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
