"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"

// Video untuk setiap model - taruh file di folder public
const modelVideos: Record<string, string> = {
  spasial: "/demo-spatial.mp4",
  temporal: "/demo-temporal.",
  liveness: "/demo-liveness.mp4",
  hybrid: "/demo-hybrid.mp4",
}

const modelNames: Record<string, string> = {
  spasial: "Deep-Spatial",
  temporal: "Deep-Temporal",
  liveness: "Liveness & Biometric",
  hybrid: "Hybrid AI",
}

const sampleResponse = `  "status": "success",
  "request": {
    "id": "req_fPDJ8fsQGgC1E2Ed8V",
    "timestamp": 1709634469.399,
    "operations": 24
  },
  "type": {
    "ai_generated": 99%
  },
  "media": {
    "id": "med_fPDJiQzaZR2E193CXb",
    "uri": "video.mp4"
  }
}`

interface DemoProps {
  selectedModel: string | null
}

export default function Demo({ selectedModel }: DemoProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = () => {
    setIsAnalyzing(true)
    setTimeout(() => setIsAnalyzing(false), 2000)
  }

  // Default video jika belum ada model yang dipilih
  const videoSrc = selectedModel ? modelVideos[selectedModel] : "/.mp4"

  return (
    <section className="w-full py-16 md:py-24 bg-slate-50 dark:bg-slate-900/50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Model Label */}
        {selectedModel && (
          <div className="text-center mb-8">
            <span className="inline-block px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium">
              Model: {modelNames[selectedModel]}
            </span>
          </div>
        )}

        {/* Content */}
        <div className="flex flex-col lg:flex-row items-center justify-center gap-6 lg:gap-10">
          {/* Left - Video Preview */}
          <div className="w-full max-w-sm">
            <div className="aspect-square rounded-2xl overflow-hidden shadow-lg bg-white dark:bg-slate-800">
              <video
                key={videoSrc} // Force re-render when video changes
                className="w-full h-full object-cover"
                autoPlay
                muted
                loop
                playsInline
                src={videoSrc}
              />
            </div>
          </div>

          {/* Arrow */}
          <div className="hidden lg:flex items-center justify-center">
            <svg
              className="w-10 h-10 text-slate-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
          </div>

          {/* Right - Code Response */}
          <div className="w-full max-w-md">
            <div className="rounded-2xl overflow-hidden shadow-lg">
              {/* Window Header */}
              <div className="bg-slate-700 dark:bg-slate-800 px-4 py-3 flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
              </div>
              {/* Code Content */}
              <div className="bg-slate-800 dark:bg-slate-900 p-5 font-mono text-sm">
                <pre className="text-slate-300 leading-relaxed">
                  {sampleResponse.split("\n").map((line, i) => (
                    <div key={i} className="flex">
                      <span className="text-slate-500 w-6 flex-shrink-0 select-none">
                        {i + 1}
                      </span>
                      <span>
                        {line.includes('"status"') && (
                          <>
                            <span className="text-slate-300">{`  "`}</span>
                            <span className="text-purple-400">status</span>
                            <span className="text-slate-300">{`": "`}</span>
                            <span className="text-green-400">success</span>
                            <span className="text-slate-300">{`",`}</span>
                          </>
                        )}
                        {line.includes('"id":') && line.includes('req_') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">id</span>
                            <span className="text-slate-300">{`": "`}</span>
                            <span className="text-yellow-400">req_fPDJ8fsQGgC1E2Ed8V</span>
                            <span className="text-slate-300">{`",`}</span>
                          </>
                        )}
                        {line.includes('"timestamp"') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">timestamp</span>
                            <span className="text-slate-300">{`": `}</span>
                            <span className="text-orange-400">1709634469.399</span>
                            <span className="text-slate-300">{`,`}</span>
                          </>
                        )}
                        {line.includes('"operations"') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">operations</span>
                            <span className="text-slate-300">{`": `}</span>
                            <span className="text-orange-400">24</span>
                          </>
                        )}
                        {line.includes('"request"') && (
                          <>
                            <span className="text-slate-300">{`  "`}</span>
                            <span className="text-purple-400">request</span>
                            <span className="text-slate-300">{`": {`}</span>
                          </>
                        )}
                        {line.includes('"type"') && (
                          <>
                            <span className="text-slate-300">{`  "`}</span>
                            <span className="text-purple-400">type</span>
                            <span className="text-slate-300">{`": {`}</span>
                          </>
                        )}
                        {line.includes('"ai_generated"') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">ai_generated</span>
                            <span className="text-slate-300">{`": `}</span>
                            <span className="text-yellow-400">99%</span>
                          </>
                        )}
                        {line.includes('"media"') && (
                          <>
                            <span className="text-slate-300">{`  "`}</span>
                            <span className="text-purple-400">media</span>
                            <span className="text-slate-300">{`": {`}</span>
                          </>
                        )}
                        {line.includes('"id":') && line.includes('med_') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">id</span>
                            <span className="text-slate-300">{`": "`}</span>
                            <span className="text-yellow-400">med_fPDJiQzaZR2E193CXb</span>
                            <span className="text-slate-300">{`",`}</span>
                          </>
                        )}
                        {line.includes('"uri"') && (
                          <>
                            <span className="text-slate-300">{`    "`}</span>
                            <span className="text-purple-400">uri</span>
                            <span className="text-slate-300">{`": "`}</span>
                            <span className="text-yellow-400">video.mp4</span>
                            <span className="text-slate-300">{`"`}</span>
                          </>
                        )}
                        {line.trim() === '},' && <span className="text-slate-300">{`  },`}</span>}
                        {line.trim() === '}' && <span className="text-slate-300">{`}`}</span>}
                      </span>
                    </div>
                  ))}
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Button */}
        <div className="flex justify-center mt-12">
          <Button
            size="lg"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="bg-primary hover:bg-primary/90 text-primary-foreground px-10 py-6 text-base font-semibold uppercase tracking-wide rounded-lg"
          >
            {isAnalyzing ? "Checking..." : "Check Video"}
          </Button>
        </div>
      </div>
    </section>
  )
}
