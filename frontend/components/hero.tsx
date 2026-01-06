"use client"

import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"

// Model ensemble (recommended)
const ensembleModel = { 
  id: "ensemble", 
  name: "Smart Ensemble", 
  description: "Gabungan 4 Model AI" 
}

// Individual models
const individualModels = [
  { id: "temporal", name: "Deep-Temporal" },
  { id: "liveness", name: "Deep-Liveness (rPPG)" },
  { id: "hybrid", name: "Deep-Hybrid" },
  { id: "spasial", name: "Deep-Spatial" },
]

// Video untuk setiap model
const modelVideos: Record<string, string> = {
  ensemble: "/demo-smart.mp4",
  spasial: "/demo-spatial.mp4",
  temporal: "/demo-temporal.mp4",
  liveness: "/demo-liveness.mp4",
  hybrid: "/demo-hybrid.mp4",
}

// Sample response untuk setiap model
const modelResponses: Record<string, string> = {
  ensemble: `{
  "prediction": "FAKE",
  "confidence": 0.87,
  "temporal": {
    "prediction": "FAKE",
    "confidence": 0.85,
    "available": true
  },
  "liveness": {
    "prediction": "FAKE",
    "confidence": 0.92,
    "available": true
  },
  "hybrid": {
    "prediction": "FAKE",
    "confidence": 0.78,
    "available": true
  },
  "spatial": {
    "prediction": "FAKE",
    "confidence": 0.88,
    "available": true
  },
  "voting_result": "FAKE",
  "models_available": 4,
  "ensemble_method": "weighted_average"
}`,
  spasial: `{
  "prediction": "FAKE",
  "confidence": 0.873,
  "faces_analyzed": 20
}`,
  temporal: `{
  "prediction": "FAKE",
  "confidence": 0.99,
  "frames_processed": 20,
  "audio_features_extracted": true
}`,
  liveness: `{
  "prediction": "REAL",
  "confidence": 0.90,
  "frames_processed": 100,
  "heart_rate": "95 BPM
}`,
  hybrid: `{
  "prediction": "FAKE",
  "confidence": 0.78,
  "faces_analyzed": 50
}`
}

interface HeroProps {
  selectedModel: string
  onSelectModel: (model: string) => void
}

export default function Hero({ selectedModel, onSelectModel }: HeroProps) {
  const router = useRouter()

  const handleSubmit = () => {
    if (selectedModel) {
      router.push(`/upload/${selectedModel}`)
    }
  }

  const videoSrc = modelVideos[selectedModel] || "/demo.mp4"
  const sampleResponse = modelResponses[selectedModel] || modelResponses.ensemble

  // Fungsi untuk syntax highlighting JSON sederhana
  const highlightJSON = (json: string) => {
    return json
      .replace(/"([^"]+)":/g, '<span class="text-purple-400">"$1"</span>:')
      .replace(/: "([^"]+)"/g, ': <span class="text-green-400">"$1"</span>')
      .replace(/: (true|false)/g, ': <span class="text-blue-400">$1</span>')
      .replace(/: ([\d.]+)/g, ': <span class="text-orange-400">$1</span>')
  }

  return (
    <section className="relative w-full py-16 md:py-24 bg-gradient-to-b from-background to-background/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-center text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/20 text-accent-foreground mb-6">
            <span className="w-2 h-2 bg-accent rounded-full animate-pulse" />
            <span className="text-sm font-medium">Advanced Detection AI</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6 leading-tight">
            Deepfake Detector
          </h1>

          {/* Subheading */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mb-4">
            Deteksi video dan audio sintetis dengan akurasi 99% menggunakan AI.
            Lindungi konten Anda dari manipulasi.
          </p>

          <p className="text-muted-foreground mb-6">
            Pilih model deteksi untuk melanjutkan:
          </p>

          {/* Recommended - Smart Ensemble */}
          <div className="mb-6">
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Rekomendasi
              </span>
            </div>
            <button
              onClick={() => onSelectModel(ensembleModel.id)}
              className={`px-8 py-3.5 rounded-full text-base font-semibold transition-all border-2 ${
                selectedModel === ensembleModel.id
                  ? "bg-primary text-primary-foreground border-primary shadow-lg shadow-primary/50 scale-105"
                  : "bg-muted text-muted-foreground border-muted hover:border-primary/50 hover:scale-102"
              }`}
            >
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>{ensembleModel.name}</span>
                <span className="text-xs opacity-75">({ensembleModel.description})</span>
              </div>
            </button>
          </div>

          {/* Separator */}
          <div className="flex items-center justify-center w-full max-w-md mb-6">
            <div className="flex-1 border-t border-muted"></div>
            <span className="px-4 text-xs text-muted-foreground uppercase tracking-wider">
              Atau pilih model individual
            </span>
            <div className="flex-1 border-t border-muted"></div>
          </div>

          {/* Individual Models - Pill Style */}
          <div id="model-selection" className="flex flex-wrap justify-center gap-3 mb-12">
            {individualModels.map((model) => (
              <button
                key={model.id}
                onClick={() => onSelectModel(model.id)}
                className={`px-6 py-2.5 rounded-full text-sm font-medium transition-all ${
                  selectedModel === model.id
                    ? "bg-primary text-primary-foreground shadow-md"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {model.name}
              </button>
            ))}
          </div>

          {/* Demo Section */}
          <div id="demo-section" className="flex flex-col lg:flex-row items-center justify-center gap-6 lg:gap-10 mb-12">
            {/* Left - Video Preview */}
            <div className="w-full max-w-sm">
              <div className="aspect-square rounded-2xl overflow-hidden shadow-lg bg-white dark:bg-slate-800">
                <video
                  key={videoSrc}
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
                className="w-10 h-10 text-muted-foreground"
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
                <div className="bg-slate-800 dark:bg-slate-900 p-5 font-mono text-sm overflow-auto max-h-[400px]">
                  <pre className="text-slate-300 leading-relaxed whitespace-pre-wrap">
                    {sampleResponse.split("\n").map((line: string, i: number) => (
                      <div key={i} className="flex gap-2">
                        <span className="text-slate-500 select-none min-w-[1.5rem] text-right">
                          {i + 1}
                        </span>
                        <code 
                          className="text-slate-300"
                          dangerouslySetInnerHTML={{ 
                            __html: line
                              .replace(/"([^"]+)":/g, '<span class="text-purple-400">"$1"</span>:')
                              .replace(/: "([^"]+)"/g, ': <span class="text-green-400">"$1"</span>')
                              .replace(/: (true|false)/g, ': <span class="text-blue-400">$1</span>')
                              .replace(/: ([\d.]+)/g, ': <span class="text-orange-400">$1</span>')
                          }} 
                        />
                      </div>
                    ))}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <Button
            size="lg"
            onClick={handleSubmit}
            disabled={!selectedModel}
            className="px-10 py-6 text-base font-semibold uppercase tracking-wide"
          >
            Lanjutkan
          </Button>
        </div>
      </div>
    </section>
  )
}
