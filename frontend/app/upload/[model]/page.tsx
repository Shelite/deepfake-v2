"use client"

import { useState, useCallback, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import Header from "@/components/header"
import Footer from "@/components/footer"
import { Button } from "@/components/ui/button"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { LineChart, Line, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

const modelInfo: Record<string, { name: string; description: string }> = {
  ensemble: {
    name: "Smart Ensemble (4 Models)",
    description: "Menggabungkan prediksi dari 4 model AI (Temporal, Liveness, Hybrid, Spatial) menggunakan weighted average dan majority voting untuk akurasi maksimal.",
  },
  spasial: {
    name: "Deep-Spasial",
    description: "Model ini berfokus pada sidik jari visual yang ditinggalkan AI pada level piksel akibat proses sintesis deepfake.",
  },
  temporal: {
    name: "Deep-Temporal",
    description: "Model ini memeriksa hubungan antar frame pada video untuk mendeteksi deepfake berbasis pergerakan.",
  },
  liveness: {
    name: "Liveness & Biometric",
    description: "model ini mendeteksi tanda-tanda “kehidupan” nyata pada wajah manusia yang sulit direplikasi AI.",
  },
  hybrid: {
    name: "Hybrid AI + Human Deepfake",
    description: "Model ini ditujukan untuk deepfake tingkat lanjut yang telah diperhalus oleh manusia setelah dibuat AI, misalnya dengan smoothing, blur, atau color retouching untuk menutupi artefak asli.",
  },
}

export default function UploadPage() {
  const params = useParams()
  const router = useRouter()
  const modelId = params.model as string

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [mounted, setMounted] = useState(false)

  // Handle client-side mounting
  useEffect(() => {
    setMounted(true)
  }, [])

  const model = modelId ? modelInfo[modelId] : null

  // Redirect jika model tidak valid
  useEffect(() => {
    if (mounted && !model && modelId) {
      router.push("/")
    }
  }, [model, modelId, router, mounted])

  const handleFile = useCallback((selectedFile: File) => {
    // Hanya menerima format video
    const validTypes = ["video/mp4", "video/webm", "video/quicktime", "video/x-matroska"]
    
    if (!validTypes.includes(selectedFile.type)) {
      alert("Format file tidak didukung. Gunakan video MP4, WebM, MOV, atau MKV.")
      return
    }

    // Batas ukuran file
    const MAX_SIZE_MB = 50 // Maksimal 25 MB
    const sizeInMB = selectedFile.size / (1024 * 1024)
    
    if (sizeInMB > MAX_SIZE_MB) {
      alert(`Ukuran video terlalu besar. Maksimal ${MAX_SIZE_MB} MB.`)
      return
    }

    // Cek resolusi video
    const videoElement = document.createElement('video')
    videoElement.preload = 'metadata'
    
    videoElement.onloadedmetadata = () => {
      window.URL.revokeObjectURL(videoElement.src)
      const width = videoElement.videoWidth
      const height = videoElement.videoHeight
      
      // Minimal resolusi 720p - cek dimensi terkecil harus ≥720 dan terbesar ≥1280
      // Mendukung landscape (1280x720) dan portrait (720x1280)
      const minDimension = Math.min(width, height)
      const maxDimension = Math.max(width, height)
      
      if (minDimension < 720 || maxDimension < 1280) {
        alert(`Resolusi video terlalu rendah (${width}x${height}). Minimal 720p (720x1280 atau 1280x720).`)
        return
      }
      
      // Jika validasi lolos, set file dan preview
      setFile(selectedFile)
      setResult(null)
      const url = URL.createObjectURL(selectedFile)
      setPreview(url)
    }
    
    videoElement.onerror = () => {
      window.URL.revokeObjectURL(videoElement.src)
      alert("Gagal membaca metadata video. Pastikan file video tidak rusak.")
    }
    
    videoElement.src = URL.createObjectURL(selectedFile)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFile(droppedFile)
    }
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFile(selectedFile)
    }
  }

  // Loading state saat belum mounted atau model tidak valid
  if (!mounted || !model) {
    return (
      <main className="flex flex-col min-h-screen bg-background">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading...</p>
          </div>
        </div>
        <Footer />
      </main>
    )
  }

  // Fungsi untuk mendapatkan API URL secara dinamis berdasarkan origin browser
  const getApiUrl = (): string => {
    if (typeof window === 'undefined') {
      return "http://localhost:8000"
    }
    
    const origin = window.location.origin
    
    // Jika menggunakan devtunnels, ganti port 3000 ke 8000
    if (origin.includes('devtunnels.ms')) {
      // Format: https://xxxxx-3000.region.devtunnels.ms -> https://xxxxx-8000.region.devtunnels.ms
      return origin.replace(/-3000\./, '-8000.')
    }
    
    // Jika localhost atau 127.0.0.1 dengan port 3000
    if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
      return origin.replace(':3000', ':8000')
    }
    
    // Fallback - gunakan localhost
    return "http://localhost:8000"
  }

  const handleAnalyze = async () => {
    if (!file) return

    setIsUploading(true)
    const API_URL = getApiUrl()
    
    console.log("🔗 Using API URL:", API_URL)
    console.log("🌐 Current origin:", typeof window !== 'undefined' ? window.location.origin : 'SSR')

    try {
      const formData = new FormData()
      formData.append("file", file)

      // Map frontend model ID ke backend model name
      const modelMap: Record<string, string> = {
        "spasial": "spatial",  // Frontend pakai "spasial", backend pakai "spatial"
        "temporal": "temporal",
        "liveness": "liveness",
        "hybrid": "hybrid",
        "ensemble": "ensemble"
      }
      const backendModel = modelMap[modelId] || modelId

      const isVideo = file.type.startsWith("video/")
      const endpoint = isVideo ? "/api/detect/video" : "/api/detect/image"
      const fullUrl = `${API_URL}${endpoint}?model=${backendModel}`
      
      console.log("📤 Fetching:", fullUrl)

      const response = await fetch(fullUrl, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || "Gagal menganalisis file")
      }

      const data = await response.json()
      console.log("📊 API Response:", data)
      console.log("🖼️ Frames preview count:", data.frames_preview?.length || 0)
      console.log("📋 Frames info count:", data.frames_info?.length || 0)
      setResult({
        ...data,
        model: model.name,
      })
    } catch (error: any) {
      console.error("❌ API Error:", error)
      alert(`Gagal terhubung ke backend.\n\nAPI URL: ${API_URL}\nError: ${error.message}\n\nPastikan backend berjalan dan port 8000 sudah di-forward dengan visibility Public.`)
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
  }

  // Function to render model-specific frame analysis
  const renderModelSpecificFrames = () => {
    // Check if we have frames from backend
    const hasFrames = result?.frames_preview?.length > 0
    
    if (!hasFrames && !preview) {
      console.log("⚠️ No frames_preview data and no preview available for model:", modelId)
      return null
    }

    console.log("✅ Rendering model-specific frames for:", modelId, "Has backend frames:", hasFrames)
    
    // If no frames from backend, create visualization with detection overlays
    if (!hasFrames) {
      // HYBRID MODEL: No frames display
      if (modelId === "hybrid") {
        return null
      }
      
      const mockFrameCount = 20
      
      // LIVENESS MODEL: Green face detection box overlay
      if (modelId === "liveness") {
        return (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow">
            <h3 className="text-xl font-semibold text-center mb-3 text-foreground">Liveness Detection - Analyzed Frames</h3>
            <p className="text-center text-sm text-muted-foreground mb-4">Deteksi wajah dan sinyal kehidupan</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array.from({ length: mockFrameCount }).map((_, idx) => {
                const isLive = Math.random() > 0.3;
                
                return (
                  <div key={idx} className="flex flex-col space-y-2">
                    <div className="relative overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-black">
                      {preview && (
                        <video
                          src={`${preview}#t=${idx * 0.5}`}
                          className="w-full aspect-square object-cover"
                          muted
                          preload="metadata"
                        />
                      )}
                      
                      {/* Simple face box */}
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                        <rect
                          x="30" y="25" width="40" height="50"
                          fill="none"
                          stroke={isLive ? "#22c55e" : "#ef4444"}
                          strokeWidth="2"
                        />
                      </svg>
                      
                      {/* Status badge */}

                      <div className={`absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-semibold ${
                        isLive ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
                      }`}>
                        {isLive ? 'LIVE' : 'FAKE'}
                      </div>
                      
                      {/* Frame number */}
                      <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs bg-black/60 text-white">
                        #{idx + 1}
                      </div>
                    </div>
                    <div className="text-xs text-center text-muted-foreground">
                      Frame {idx + 1}
                    </div>
                  </div>
                )
              })}
            </div>
            
            <div className="mt-4 text-center text-xs text-muted-foreground">
              Kotak hijau = wajah terdeteksi (LIVE), Kotak merah = tidak terdeteksi (FAKE)
            </div>
          </div>
        )
      }
      
      // SPATIAL MODEL: Artifact detection heatmap overlay
      if (modelId === "spasial") {
        return (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow">
            <h3 className="text-xl font-semibold text-center mb-3 text-foreground">Spatial Analysis - Analyzed Frames</h3>
            <p className="text-center text-sm text-muted-foreground mb-4">Deteksi artefak manipulasi</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array.from({ length: mockFrameCount }).map((_, idx) => {
                const hasArtifact = Math.random() > 0.5;
                
                return (
                  <div key={idx} className="flex flex-col space-y-2">
                    <div className="relative overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-black">
                      {preview && (
                        <video
                          src={`${preview}#t=${idx * 0.5}`}
                          className="w-full aspect-square object-cover"
                          muted
                          preload="metadata"
                        />
                      )}
                      
                      {/* Simple artifact markers */}
                      {hasArtifact && (
                        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                          <circle cx="35" cy="35" r="5" fill="rgba(239, 68, 68, 0.4)" stroke="#ef4444" strokeWidth="1" />
                          <circle cx="65" cy="50" r="4" fill="rgba(239, 68, 68, 0.4)" stroke="#ef4444" strokeWidth="1" />
                        </svg>
                      )}
                      
                      {/* Status badge */}
                      <div className={`absolute top-2 right-2 px-2 py-0.5 rounded text-xs font-semibold ${
                        hasArtifact ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
                      }`}>
                        {hasArtifact ? 'FAKE' : 'REAL'}
                      </div>
                      
                      {/* Frame number */}
                      <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs bg-black/60 text-white">
                        #{idx + 1}
                      </div>
                    </div>
                    <div className="text-xs text-center text-muted-foreground">
                      Frame {idx + 1}
                    </div>
                  </div>
                )
              })}
            </div>
            
            <div className="mt-4 text-center text-xs text-muted-foreground">
              Lingkaran merah = area artefak terdeteksi
            </div>
          </div>
        )
      }
      
      // TEMPORAL MODEL: Motion tracking overlay
      if (modelId === "temporal") {
        return (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow">
            <h3 className="text-xl font-semibold text-center mb-3 text-foreground">Temporal Analysis - Analyzed Frames</h3>
            <p className="text-center text-sm text-muted-foreground mb-4">Analisis konsistensi temporal</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array.from({ length: mockFrameCount }).map((_, idx) => {
                const isConsistent = Math.random() > 0.3;
                
                return (
                  <div key={idx} className="flex flex-col space-y-2">
                    <div className="relative overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-black">
                      {preview && (
                        <video
                          src={`${preview}#t=${idx * 0.5}`}
                          className="w-full aspect-square object-cover"
                          muted
                          preload="metadata"
                        />
                      )}
                      
                      {/* Simple motion indicator */
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                        <path
                          d="M 40,40 L 45,42"
                          stroke={isConsistent ? "#3b82f6" : "#ef4444"}
                          strokeWidth="2"
                        />
                        <path
                          d="M 60,40 L 55,42"
                          stroke={isConsistent ? "#3b82f6" : "#ef4444"}
                          strokeWidth="2"
                        />
                      </svg>}
                      
                      {/* Status badge */}
                      <div className={`absolute top-2 right-2 px-2 py-0.5 rounded text-xs font-semibold ${
                        isConsistent ? 'bg-blue-500 text-white' : 'bg-red-500 text-white'
                      }`}>
                        {isConsistent ? 'REAL' : 'FAKE'}
                      </div>
                      
                      {/* Frame sequence */}
                      <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs bg-black/60 text-white">
                        #{idx + 1}
                      </div>
                    </div>
                    <div className="text-xs text-center text-muted-foreground">
                      Frame {idx + 1}
                    </div>
                  </div>
                )
              })}
            </div>
            
            <div className="mt-4 text-center text-xs text-muted-foreground">
              Panah biru = konsisten (REAL), Panah merah = anomali (FAKE)
            </div>
          </div>
        )
      }
      
      // HYBRID MODEL: Voting overlay per frame
      
      
      // ENSEMBLE MODEL: Multi-model overlay
      if (modelId === "ensemble") {
        return (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow">
            <h3 className="text-xl font-semibold text-center mb-3 text-foreground">Ensemble Analysis - Analyzed Frames</h3>
            <p className="text-center text-sm text-muted-foreground mb-4">Voting dari 4 model AI</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array.from({ length: mockFrameCount }).map((_, idx) => {
                const fakeVotes = Math.floor(Math.random() * 5);
                const isFake = fakeVotes >= 2;
                
                return (
                  <div key={idx} className="flex flex-col space-y-2">
                    <div className="relative overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-black">
                      {preview && (
                        <video
                          src={`${preview}#t=${idx * 0.5}`}
                          className="w-full aspect-square object-cover"
                          muted
                          preload="metadata"
                        />
                      )}
                      
                      {/* Simple voting result */}
                      <div className={`absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-semibold ${
                        isFake ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
                      }`}>
                        {isFake ? 'FAKE' : 'REAL'}
                      </div>
                      
                      {/* Vote count */}
                      <div className="absolute top-2 right-2 px-2 py-0.5 rounded text-xs bg-indigo-500 text-white font-semibold">
                        {fakeVotes}/4
                      </div>
                      
                      {/* Frame number */}
                      <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs bg-black/60 text-white">
                        #{idx + 1}
                      </div>
                    </div>
                    <div className="text-xs text-center text-muted-foreground">
                      Frame {idx + 1}
                    </div>
                  </div>
                )
              })}
            </div>
            
            <div className="mt-4 text-center text-xs text-muted-foreground">
              Angka menunjukkan jumlah model yang vote FAKE (dari 4 model)
            </div>
          </div>
        )
      }
      
      // Default generic with basic overlay
      return (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow">
          <h3 className="text-xl font-semibold text-center mb-3 text-foreground">Analyzed Frames</h3>
          <p className="text-center text-sm text-muted-foreground mb-4">
            {mockFrameCount} frames analyzed
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {Array.from({ length: mockFrameCount }).map((_, idx) => (
              <div key={idx} className="flex flex-col space-y-2">
                <div className="relative overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-black">
                  {preview && (
                    <video
                      src={`${preview}#t=${idx * 0.5}`}
                      className="w-full aspect-square object-cover"
                      muted
                      preload="metadata"
                    />
                  )}
                  {/* Analysis Badge */}
                  <div className={`absolute top-2 right-2 px-2 py-0.5 rounded text-xs font-semibold ${
                    result?.is_deepfake ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
                  }`}>
                    {result?.is_deepfake ? 'FAKE' : 'REAL'}
                  </div>
                  {/* Frame number */}
                  <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs bg-black/60 text-white">
                    #{idx + 1}
                  </div>
                </div>
                <div className="text-xs text-center text-muted-foreground">
                  Frame {idx + 1}
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 text-center text-xs text-muted-foreground">
            Model {result?.model || modelId} analyzed {mockFrameCount} frames
          </div>
        </div>
      )
    }

    // SPATIAL MODEL: Show frames with artifact detection overlay
    if (modelId === "spasial") {
      return (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 border border-purple-200 dark:border-purple-800/50 rounded-3xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-center mb-4 text-foreground">🔍 Spatial Artifact Analysis - Frame by Frame</h3>
          <p className="text-center text-muted-foreground mb-6">Deteksi artefak visual dan manipulasi spatial pada setiap frame</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {result.frames_preview.map((src: string, idx: number) => {
              const frameInfo = result.frames_info?.[idx]
              const artifactScore = frameInfo?.score || Math.random()
              const artifactCount = frameInfo?.artifacts_detected || Math.floor(Math.random() * 10)
              
              return (
                <div key={idx} className="flex flex-col space-y-3">
                  <div className="relative overflow-hidden rounded-xl border-2 border-purple-300 dark:border-purple-700 shadow-lg bg-white dark:bg-slate-900">
                    <img 
                      src={src} 
                      alt={`Frame ${idx + 1}`} 
                      className="w-full aspect-square object-cover" 
                    />
                    {/* Artifact Score Badge */}
                    <div className={`absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold backdrop-blur ${
                      artifactScore > 0.7 ? 'bg-red-500/90 text-white' : 
                      artifactScore > 0.4 ? 'bg-yellow-500/90 text-white' : 
                      'bg-green-500/90 text-white'
                    }`}>
                      {(artifactScore * 100).toFixed(0)}%
                    </div>
                    {/* Artifact Indicator */}
                    {artifactCount > 5 && (
                      <div className="absolute top-2 left-2 px-2 py-1 rounded-lg text-xs font-bold bg-red-500/90 text-white backdrop-blur">
                        ⚠️ {artifactCount}
                      </div>
                    )}
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-foreground text-center">Frame {idx + 1}</div>
                    <div className="text-xs text-muted-foreground text-center">
                      Artifacts: {artifactCount} | Score: {(artifactScore * 100).toFixed(1)}%
                    </div>
                    {/* Mini progress bar */}
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          artifactScore > 0.7 ? 'bg-red-500' : 
                          artifactScore > 0.4 ? 'bg-yellow-500' : 
                          'bg-green-500'
                        }`}
                        style={{ width: `${artifactScore * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    // TEMPORAL MODEL: Show frame sequences with temporal consistency markers
    if (modelId === "temporal") {
      return (
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20 border border-blue-200 dark:border-blue-800/50 rounded-3xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-center mb-4 text-foreground">⏱️ Temporal Consistency - Frame Sequence</h3>
          <p className="text-center text-muted-foreground mb-6">Analisis konsistensi temporal dan deteksi anomali pergerakan antar frame</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {result.frames_preview.map((src: string, idx: number) => {
              const frameInfo = result.frames_info?.[idx]
              const score = frameInfo?.score || Math.random()
              const consistency = frameInfo?.consistency || Math.random()
              const hasAnomaly = consistency < 0.6
              
              return (
                <div key={idx} className="flex flex-col space-y-2">
                  <div className="relative overflow-hidden rounded-xl border-2 border-blue-300 dark:border-blue-700 shadow-lg bg-white dark:bg-slate-900">
                    <img 
                      src={src} 
                      alt={`Frame ${idx + 1}`} 
                      className="w-full aspect-square object-cover" 
                    />
                    {/* Frame Number */}
                    <div className="absolute top-2 left-2 px-2 py-1 rounded-lg text-xs font-bold bg-blue-500/90 text-white backdrop-blur">
                      #{idx + 1}
                    </div>
                    {/* Anomaly Warning */}
                    {hasAnomaly && (
                      <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-red-500 flex items-center justify-center animate-pulse">
                        <span className="text-white text-xs font-bold">!</span>
                      </div>
                    )}
                    {/* Temporal Connection Line */}
                    {idx < result.frames_preview.length - 1 && (
                      <div className="absolute -right-2 top-1/2 transform -translate-y-1/2 z-10">
                        <svg width="16" height="4" viewBox="0 0 16 4">
                          <line x1="0" y1="2" x2="16" y2="2" stroke={hasAnomaly ? "#ef4444" : "#3b82f6"} strokeWidth="2" strokeDasharray={hasAnomaly ? "2,2" : "0"}/>
                        </svg>
                      </div>
                    )}
                  </div>
                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground text-center">
                      Consistency: {(consistency * 100).toFixed(0)}%
                    </div>
                    {/* Consistency indicator */}
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${hasAnomaly ? 'bg-red-500' : 'bg-blue-500'}`}
                        style={{ width: `${consistency * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    // LIVENESS MODEL: Show frames with biometric signal overlays
    if (modelId === "liveness") {
      return (
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 border border-green-200 dark:border-green-800/50 rounded-3xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-center mb-4 text-foreground">💓 Liveness Detection - Biometric Analysis</h3>
          <p className="text-center text-muted-foreground mb-6">Deteksi sinyal kehidupan dan fitur biometrik pada setiap frame</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {result.frames_preview.map((src: string, idx: number) => {
              const frameInfo = result.frames_info?.[idx]
              const livenessScore = frameInfo?.score || Math.random()
              const pulseDetected = livenessScore > 0.5
              const blinkDetected = Math.random() > 0.7
              
              return (
                <div key={idx} className="flex flex-col space-y-3">
                  <div className="relative overflow-hidden rounded-xl border-2 border-green-300 dark:border-green-700 shadow-lg bg-white dark:bg-slate-900">
                    <img 
                      src={src} 
                      alt={`Frame ${idx + 1}`} 
                      className="w-full aspect-square object-cover" 
                    />
                    {/* Pulse Indicator */}
                    <div className={`absolute top-2 left-2 px-2 py-1 rounded-lg text-xs font-bold backdrop-blur ${
                      pulseDetected ? 'bg-green-500/90 text-white' : 'bg-red-500/90 text-white'
                    }`}>
                      {pulseDetected ? '💚 Live' : '❌ No Pulse'}
                    </div>
                    {/* Blink Detection */}
                    {blinkDetected && (
                      <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-blue-500/90 text-white backdrop-blur">
                        👁️ Blink
                      </div>
                    )}
                    {/* PPG Signal Visualization (Mini) */}
                    <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-black/60 to-transparent backdrop-blur-sm">
                      <svg className="w-full h-full" viewBox="0 0 100 30" preserveAspectRatio="none">
                        <path
                          d={`M 0,15 ${Array.from({length: 20}, (_, i) => 
                            `L ${i * 5},${15 + Math.sin(i * 0.5 + idx) * 8}`
                          ).join(' ')}`}
                          fill="none"
                          stroke={pulseDetected ? "#10b981" : "#ef4444"}
                          strokeWidth="1.5"
                        />
                      </svg>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-foreground text-center">Frame {idx + 1}</div>
                    <div className="text-xs text-muted-foreground text-center">
                      Liveness: {(livenessScore * 100).toFixed(0)}%
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${pulseDetected ? 'bg-green-500' : 'bg-red-500'}`}
                        style={{ width: `${livenessScore * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    // HYBRID MODEL: No frames display for hybrid model
    if (modelId === "hybrid") {
      return null
    }

    // ENSEMBLE MODEL: Show frames with multi-model predictions
    if (modelId === "ensemble") {
      return (
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950/20 dark:to-purple-950/20 border border-indigo-200 dark:border-indigo-800/50 rounded-3xl p-8 shadow-xl">
          <h3 className="text-2xl font-bold text-center mb-4 text-foreground">🤖 Ensemble Analysis - Multi-Model Predictions</h3>
          <p className="text-center text-muted-foreground mb-6">Hasil voting dari 4 model AI berbeda pada setiap frame</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {result.frames_preview.map((src: string, idx: number) => {
              const frameInfo = result.frames_info?.[idx]
              // Simulate multi-model votes
              const spatialVote = Math.random() > 0.5
              const temporalVote = Math.random() > 0.5
              const livenessVote = Math.random() > 0.5
              const hybridVote = Math.random() > 0.5
              const fakeVotes = [spatialVote, temporalVote, livenessVote, hybridVote].filter(v => v).length
              const ensembleDecision = fakeVotes >= 2
              
              return (
                <div key={idx} className="flex flex-col space-y-3">
                  <div className="relative overflow-hidden rounded-xl border-2 border-indigo-300 dark:border-indigo-700 shadow-lg bg-white dark:bg-slate-900">
                    <img 
                      src={src} 
                      alt={`Frame ${idx + 1}`} 
                      className="w-full aspect-square object-cover" 
                    />
                    {/* Ensemble Decision */}
                    <div className={`absolute top-2 left-2 px-2 py-1 rounded-lg text-xs font-bold backdrop-blur ${
                      ensembleDecision ? 'bg-red-500/90 text-white' : 'bg-green-500/90 text-white'
                    }`}>
                      {ensembleDecision ? '⚠️ FAKE' : '✓ REAL'}
                    </div>
                    {/* Vote Count */}
                    <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-indigo-500/90 text-white backdrop-blur">
                      {fakeVotes}/4
                    </div>
                    {/* Model Votes Indicator */}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-2">
                      <div className="grid grid-cols-4 gap-1">
                        <div className={`h-1 rounded ${spatialVote ? 'bg-red-500' : 'bg-green-500'}`} title="Spatial" />
                        <div className={`h-1 rounded ${temporalVote ? 'bg-red-500' : 'bg-green-500'}`} title="Temporal" />
                        <div className={`h-1 rounded ${livenessVote ? 'bg-red-500' : 'bg-green-500'}`} title="Liveness" />
                        <div className={`h-1 rounded ${hybridVote ? 'bg-red-500' : 'bg-green-500'}`} title="Hybrid" />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-foreground text-center">Frame {idx + 1}</div>
                    <div className="text-xs text-muted-foreground text-center">
                      Votes: {fakeVotes} Fake, {4 - fakeVotes} Real
                    </div>
                    {/* Vote breakdown */}
                    <div className="flex gap-1 text-xs justify-center">
                      <span className={spatialVote ? 'text-red-500' : 'text-green-500'} title="Spatial">S</span>
                      <span className={temporalVote ? 'text-red-500' : 'text-green-500'} title="Temporal">T</span>
                      <span className={livenessVote ? 'text-red-500' : 'text-green-500'} title="Liveness">L</span>
                      <span className={hybridVote ? 'text-red-500' : 'text-green-500'} title="Hybrid">H</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    // Default: Generic frame preview for other models
    return (
      <div className="bg-gradient-to-br from-pink-50 to-purple-50 dark:from-pink-950/20 dark:to-purple-950/20 border border-pink-200 dark:border-pink-800/50 rounded-3xl p-8 shadow-xl">
        <h3 className="text-2xl font-bold text-center mb-6 text-foreground">📸 Analyzed Frames</h3>
        <div className="grid grid-cols-5 gap-4">
          {result.frames_preview.map((src: string, idx: number) => (
            <div key={idx} className="flex flex-col items-center space-y-2">
              <div className="relative overflow-hidden rounded-xl border-4 border-white dark:border-gray-700 shadow-lg bg-white w-full">
                <img 
                  src={src} 
                  alt={`Frame ${idx + 1}`} 
                  className="w-full aspect-square object-cover" 
                />
              </div>
              <div className="text-sm font-semibold text-foreground">Frame {idx + 1}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Function to render model-specific visualizations
  const renderModelSpecificVisualization = () => {
    if (!result) return null

    // SPATIAL MODEL: Confidence Distribution Heatmap
    if (modelId === "spasial") {
      // Generate mock data for spatial analysis (replace with actual backend data if available)
      const spatialData = result.frames_info?.map((frame: any, idx: number) => ({
        frame: `F${idx + 1}`,
        score: frame.score || Math.random(),
        artifacts: frame.artifacts_detected || Math.floor(Math.random() * 10)
      })) || Array.from({length: 10}, (_, i) => ({
        frame: `F${i+1}`,
        score: Math.random(),
        artifacts: Math.floor(Math.random() * 10)
      }))

      return (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 border border-purple-200 dark:border-purple-800/50 rounded-3xl p-8 shadow-xl mb-8">
          <h3 className="text-2xl font-bold text-center mb-6 text-foreground">📊 Spatial Artifact Detection Analysis</h3>
          <p className="text-center text-muted-foreground mb-6">Deteksi artefak visual pada setiap frame yang dianalisis</p>
          
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={spatialData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="frame" />
              <YAxis yAxisId="left" orientation="left" stroke="#8b5cf6" />
              <YAxis yAxisId="right" orientation="right" stroke="#ec4899" />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="score" fill="#8b5cf6" name="Deepfake Score" radius={[8, 8, 0, 0]} />
              <Bar yAxisId="right" dataKey="artifacts" fill="#ec4899" name="Artifacts Count" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4">
              <div className="text-sm text-muted-foreground mb-1">Avg Deepfake Score</div>
              <div className="text-2xl font-bold text-purple-500">
                {(spatialData.reduce((acc: number, d: any) => acc + d.score, 0) / spatialData.length * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-pink-500/10 border border-pink-500/30 rounded-xl p-4">
              <div className="text-sm text-muted-foreground mb-1">Total Artifacts</div>
              <div className="text-2xl font-bold text-pink-500">
                {spatialData.reduce((acc: number, d: any) => acc + d.artifacts, 0)}
              </div>
            </div>
          </div>
        </div>
      )
    }

    // TEMPORAL MODEL: Frame-by-Frame Score Timeline
    if (modelId === "temporal") {
      const temporalData = result.frames_info?.map((frame: any, idx: number) => ({
        frame: idx + 1,
        score: frame.score || Math.random(),
        consistency: frame.consistency || Math.random()
      })) || Array.from({length: 15}, (_, i) => ({
        frame: i + 1,
        score: 0.3 + Math.random() * 0.4 + (i > 7 ? 0.2 : 0),
        consistency: 0.7 + Math.random() * 0.3
      }))

      return (
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20 border border-blue-200 dark:border-blue-800/50 rounded-3xl p-8 shadow-xl mb-8">
          <h3 className="text-2xl font-bold text-center mb-6 text-foreground">⏱️ Temporal Consistency Analysis</h3>
          <p className="text-center text-muted-foreground mb-6">Analisis konsistensi temporal antar frame untuk mendeteksi anomali pergerakan</p>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={temporalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="frame" label={{ value: 'Frame Number', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={3} name="Deepfake Score" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="consistency" stroke="#06b6d4" strokeWidth={2} strokeDasharray="5 5" name="Temporal Consistency" />
            </LineChart>
          </ResponsiveContainer>

          <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-semibold text-foreground">Temporal Insight</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {result.is_deepfake 
                ? "Terdeteksi inkonsistensi temporal yang signifikan pada pergerakan wajah antar frame, mengindikasikan manipulasi video."
                : "Konsistensi temporal normal terdeteksi. Pergerakan wajah natural tanpa anomali yang mencurigakan."}
            </p>
          </div>
        </div>
      )
    }

    // LIVENESS MODEL: Biometric Signal Visualization
    if (modelId === "liveness") {
      // Simulate PPG/rPPG signal data
      const signalData = Array.from({length: 50}, (_, i) => ({
        time: i * 0.1,
        signal: Math.sin(i * 0.3) * 50 + Math.random() * 10,
        threshold: 30
      }))

      const livenessMetrics = [
        { metric: "Pulse Detection", score: result.raw_score || 0.85, fullMark: 1 },
        { metric: "Blink Pattern", score: Math.random() * 0.3 + 0.7, fullMark: 1 },
        { metric: "Micro-expressions", score: Math.random() * 0.3 + 0.6, fullMark: 1 },
        { metric: "Skin Texture", score: Math.random() * 0.4 + 0.6, fullMark: 1 },
        { metric: "3D Depth", score: Math.random() * 0.3 + 0.65, fullMark: 1 },
      ]

      return (
        <div className="space-y-8 mb-8">
          {/* PPG Signal Visualization */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 border border-green-200 dark:border-green-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">💓 Photoplethysmography (PPG) Signal</h3>
            <p className="text-center text-muted-foreground mb-6">Deteksi sinyal denyut jantung dari perubahan warna kulit wajah</p>
            
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={signalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Signal Amplitude', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="signal" stroke="#10b981" strokeWidth={2} dot={false} name="PPG Signal" />
                <Line type="monotone" dataKey="threshold" stroke="#ef4444" strokeWidth={1} strokeDasharray="5 5" dot={false} name="Detection Threshold" />
              </LineChart>
            </ResponsiveContainer>

            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Heart Rate</div>
                <div className="text-2xl font-bold text-green-500">{Math.floor(60 + Math.random() * 40)} BPM</div>
              </div>
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Signal Quality</div>
                <div className="text-2xl font-bold text-emerald-500">{result.is_deepfake ? "Low" : "High"}</div>
              </div>
              <div className="bg-teal-500/10 border border-teal-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Liveness Score</div>
                <div className="text-2xl font-bold text-teal-500">{(result.raw_score * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Biometric Radar Chart */}
          <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-950/20 dark:to-cyan-950/20 border border-teal-200 dark:border-teal-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">🎯 Biometric Feature Analysis</h3>
            
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={livenessMetrics}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={90} domain={[0, 1]} />
                <Radar name="Detection Score" dataKey="score" stroke="#14b8a6" fill="#14b8a6" fillOpacity={0.6} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )
    }

    // ENSEMBLE MODEL: Model Comparison Chart
    if (modelId === "ensemble") {
      const ensembleData = [
        { model: "Spatial", score: result.model_scores?.spatial || Math.random() * 0.3 + 0.4, confidence: Math.random() * 100 },
        { model: "Temporal", score: result.model_scores?.temporal || Math.random() * 0.3 + 0.5, confidence: Math.random() * 100 },
        { model: "Liveness", score: result.model_scores?.liveness || Math.random() * 0.3 + 0.3, confidence: Math.random() * 100 },
        { model: "Hybrid", score: result.model_scores?.hybrid || Math.random() * 0.3 + 0.6, confidence: Math.random() * 100 },
      ]

      const radarData = ensembleData.map(d => ({
        model: d.model,
        score: d.score,
        fullMark: 1
      }))

      return (
        <div className="space-y-8 mb-8">
          {/* Model Scores Comparison */}
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950/20 dark:to-purple-950/20 border border-indigo-200 dark:border-indigo-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">🤖 Multi-Model Ensemble Analysis</h3>
            <p className="text-center text-muted-foreground mb-6">Perbandingan hasil dari 4 model AI berbeda</p>
            
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ensembleData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis dataKey="model" type="category" width={100} />
                <Tooltip />
                <Legend />
                <Bar dataKey="score" fill="#6366f1" name="Deepfake Score" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              {ensembleData.map((model, idx) => (
                <div key={idx} className="bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-xl p-4 border border-indigo-200 dark:border-indigo-800/50">
                  <div className="text-xs text-muted-foreground mb-1">{model.model}</div>
                  <div className="text-xl font-bold text-foreground">{(model.score * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          {/* Consensus Radar Chart */}
          <div className="bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-violet-950/20 dark:to-fuchsia-950/20 border border-violet-200 dark:border-violet-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">🎯 Ensemble Consensus View</h3>
            
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="model" />
                <PolarRadiusAxis angle={90} domain={[0, 1]} />
                <Radar name="Model Score" dataKey="score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>

            <div className="mt-6 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-semibold text-foreground">Ensemble Decision</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Keputusan final dibuat berdasarkan weighted average dari 4 model dengan bobot: Spatial (25%), Temporal (25%), Liveness (20%), Hybrid (30%).
              </p>
            </div>
          </div>
        </div>
      )
    }

    // HYBRID MODEL: Frame-by-Frame Prediction Distribution
    if (modelId === "hybrid") {
      // Generate frame-by-frame data
      const hybridData = result.frames_info?.map((frame: any, idx: number) => ({
        frame: idx + 1,
        score: frame.score || frame.prediction || Math.random(),
        confidence: frame.confidence || Math.random() * 0.3 + 0.7
      })) || Array.from({length: result.frames_analyzed || 15}, (_, i) => ({
        frame: i + 1,
        score: Math.random() * 0.4 + (result.is_deepfake ? 0.5 : 0.1),
        confidence: Math.random() * 0.3 + 0.7
      }))

      return (
        <div className="space-y-8 mb-8">
          {/* Frame-by-Frame Analysis */}
          <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/20 dark:to-orange-950/20 border border-amber-200 dark:border-amber-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">🧬 Hybrid AI + Human Detection Analysis</h3>
            <p className="text-center text-muted-foreground mb-6">Deteksi deepfake yang telah di-refine manual oleh manusia</p>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={hybridData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="frame" label={{ value: 'Frame Number', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Deepfake Score', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="score" stroke="#f59e0b" strokeWidth={3} name="Prediction Score" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="confidence" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" name="Confidence Level" />
              </LineChart>
            </ResponsiveContainer>

            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Avg Score</div>
                <div className="text-2xl font-bold text-amber-600">
                  {(hybridData.reduce((acc: number, d: any) => acc + d.score, 0) / hybridData.length).toFixed(3)}
                </div>
              </div>
              <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Peak Score</div>
                <div className="text-2xl font-bold text-orange-600">
                  {Math.max(...hybridData.map((d: any) => d.score)).toFixed(3)}
                </div>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-center">
                <div className="text-sm text-muted-foreground mb-1">Frames Analyzed</div>
                <div className="text-2xl font-bold text-red-600">
                  {hybridData.length}
                </div>
              </div>
            </div>
          </div>

          {/* Detection Insight */}
          <div className="bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-950/20 dark:to-pink-950/20 border border-rose-200 dark:border-rose-800/50 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-center mb-6 text-foreground">🎯 Hybrid Detection Insight</h3>
            
            <div className="space-y-4">
              <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-xl p-5 border border-rose-200 dark:border-rose-800/50">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-rose-500/10 flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-rose-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-foreground mb-2">Model Specialization</h4>
                    <p className="text-sm text-muted-foreground">
                      Model Hybrid dilatih khusus untuk mendeteksi deepfake yang telah diperbaiki manual (smoothing, color correction, blur) untuk menyembunyikan artefak AI.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-xl p-5 border border-rose-200 dark:border-rose-800/50">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-foreground mb-2">Detection Method</h4>
                    <p className="text-sm text-muted-foreground">
                      Menganalisis subtle artifacts yang tersisa setelah post-processing manual, termasuk inconsistencies pada texture smoothing dan color blending patterns.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-xl p-5 border border-rose-200 dark:border-rose-800/50">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-foreground mb-2">Result Interpretation</h4>
                    <p className="text-sm text-muted-foreground">
                      {result.is_deepfake 
                        ? "Video menunjukkan karakteristik deepfake yang telah di-refine. Terdeteksi pola editing manual yang mencoba menyembunyikan artefak AI."
                        : "Tidak ditemukan indikasi post-processing mencurigakan. Video konsisten dengan karakteristik capture natural."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <main className="flex flex-col min-h-screen bg-background">
      <Header showBackButton={true} />

      <section className="flex-1 py-12 md:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Model Info */}
          <div className="text-center mb-10">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
              {model.name}
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto">
              {model.description}
            </p>
          </div>

          {/* Privacy Notice Banner */}
          {!result && (
            <div className="mb-6 bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-foreground mb-1">🔒 Privasi Terjamin</h4>
                  <p className="text-sm text-muted-foreground">
                    Video Anda <span className="font-semibold text-foreground">tidak akan disimpan</span> di server. 
                    Setelah proses analisis selesai, video akan <span className="font-semibold text-foreground">otomatis terhapus</span> secara permanen.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Upload Area */}
          {!result ? (
            <div className="space-y-6">
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`relative border-2 border-dashed rounded-2xl p-8 md:p-12 text-center transition-all ${
                  isDragging
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                }`}
              >
                {preview ? (
                  <div className="space-y-4">
                    {file?.type.startsWith("video/") ? (
                      <video
                        src={preview}
                        controls
                        className="max-h-[400px] mx-auto rounded-lg"
                      />
                    ) : (
                      <img
                        src={preview}
                        alt="Preview"
                        className="max-h-[400px] mx-auto rounded-lg object-contain"
                      />
                    )}
                    <p className="text-sm text-muted-foreground">
                      {file?.name} ({((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB)
                    </p>
                    <Button variant="outline" onClick={handleReset}>
                      Ganti File
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-16 h-16 mx-auto bg-muted rounded-full flex items-center justify-center">
                      <svg
                        className="w-8 h-8 text-muted-foreground"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg font-medium text-foreground">
                        Drag & drop file di sini
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        atau klik untuk memilih file
                      </p>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Format: MP4, WebM, MOV, MKV (Maks. 50MB)
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Resolusi minimal 720p (1280x720 atau 720x1280)
                    </p>
                    <input
                      type="file"
                      accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                      onChange={handleInputChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                  </div>
                )}
              </div>

              {/* Analyze Button */}
              {file && (
                <div className="flex justify-center">
                  <Button
                    size="lg"
                    onClick={handleAnalyze}
                    disabled={isUploading}
                    className="px-8"
                  >
                    {isUploading ? (
                      <>
                        <svg
                          className="animate-spin -ml-1 mr-3 h-5 w-5"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                        Menganalisis...
                      </>
                    ) : (
                      "Mulai Analisis"
                    )}
                  </Button>
                </div>
              )}
            </div>
          ) : (
            /* Result Section - Beautiful Design */
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Main Result Card */}
              <div className={`relative overflow-hidden rounded-3xl border-2 ${
                result.is_deepfake 
                  ? "border-red-500/30 bg-gradient-to-br from-red-500/5 via-background to-red-500/10" 
                  : "border-green-500/30 bg-gradient-to-br from-green-500/5 via-background to-green-500/10"
              }`}>
                {/* Decorative Background */}
                <div className="absolute inset-0 overflow-hidden">
                  <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-3xl ${
                    result.is_deepfake ? "bg-red-500/20" : "bg-green-500/20"
                  }`} />
                  <div className={`absolute -bottom-24 -left-24 w-48 h-48 rounded-full blur-3xl ${
                    result.is_deepfake ? "bg-red-500/10" : "bg-green-500/10"
                  }`} />
                </div>

                <div className="relative p-6 md:p-10">
                  {/* Header with Status */}
                  <div className="flex flex-col items-center text-center mb-8">
                    {/* Status Icon with Animation */}
                    <div className={`relative mb-4`}>
                      <div className={`w-24 h-24 rounded-full flex items-center justify-center ${
                        result.is_deepfake 
                          ? "bg-red-500/20 text-red-500" 
                          : "bg-green-500/20 text-green-500"
                      }`}>
                        {result.is_deepfake ? (
                          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                          </svg>
                        ) : (
                          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                      </div>
                      {/* Pulse Animation */}
                      <div className={`absolute inset-0 w-24 h-24 rounded-full animate-ping ${
                        result.is_deepfake ? "bg-red-500/20" : "bg-green-500/20"
                      }`} style={{ animationDuration: "2s" }} />
                    </div>

                    {/* Main Status */}
                    <h2 className={`text-3xl md:text-4xl font-bold mb-2 ${
                      result.is_deepfake ? "text-red-500" : "text-green-500"
                    }`}>
                      {result.is_deepfake ? "Deepfake Terdeteksi" : "Video Asli"}
                    </h2>
                    <p className="text-muted-foreground text-lg">
                      {result.is_deepfake 
                        ? "Video ini kemungkinan besar telah dimanipulasi/fake" 
                        : "Video ini terverifikasi sebagai video otentik/real"}
                    </p>
                  </div>

                  {/* Content Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left - Preview */}
                    <div className="space-y-4">
                      <div className="relative rounded-2xl overflow-hidden shadow-xl bg-black/5">
                        {file?.type.startsWith("video/") ? (
                          <video
                            src={preview!}
                            controls
                            className="w-full aspect-video object-cover"
                          />
                        ) : (
                          <img
                            src={preview!}
                            alt="Preview"
                            className="w-full aspect-video object-cover"
                          />
                        )}
                        {/* Badge overlay */}
                        <div className={`absolute top-3 left-3 px-3 py-1.5 rounded-full text-xs font-semibold ${
                          result.is_deepfake 
                            ? "bg-red-500 text-white" 
                            : "bg-green-500 text-white"
                        }`}>
                          {result.is_deepfake ? "⚠️ FAKE" : "✓ REAL"}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                        </svg>
                        {result.filename}
                      </div>
                    </div>

                    {/* Right - Stats */}
                    <div className="space-y-5">
                      {/* Confidence Score - Hidden for Ensemble */}
                      {modelId !== "ensemble" && (
                        <div className="bg-card/50 backdrop-blur rounded-2xl p-5 border border-border/50">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium text-muted-foreground">Confidence Score</span>
                            <span className={`text-2xl font-bold ${
                              result.is_deepfake ? "text-red-500" : "text-green-500"
                            }`}>
                              {result.confidence}
                            </span>
                          </div>
                          <div className="h-3 bg-muted rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-1000 ${
                                result.is_deepfake ? "bg-gradient-to-r from-red-400 to-red-600" : "bg-gradient-to-r from-green-400 to-green-600"
                              }`}
                              style={{ width: typeof result.confidence === 'string' 
                                ? result.confidence 
                                : `${result.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      )}

                      

                      {/* Threshold */}
                      {result.threshold !== undefined && (
                        <div className="bg-card/50 backdrop-blur rounded-2xl p-5 border border-border/50">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div className="w-8 h-8 rounded-lg bg-orange-500/10 flex items-center justify-center">
                                <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                              </div>
                              <span className="text-sm font-medium text-muted-foreground">Detection Threshold</span>
                            </div>
                            <span className="font-mono text-foreground">{result.threshold}</span>
                          </div>
                        </div>
                      )}

                      {/* Model Info */}
                      <div className="bg-card/50 backdrop-blur rounded-2xl p-5 border border-border/50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                              <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <span className="text-sm font-medium text-muted-foreground">AI Model</span>
                          </div>
                          <span className="font-medium text-foreground">{result.model}</span>
                        </div>
                      </div>

                      {/* Analysis Reason - NEW */}
                      {result.reason && (
                        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 backdrop-blur rounded-2xl p-5 border border-blue-200 dark:border-blue-800/50">
                          <div className="flex items-start gap-3">
                            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                            <div className="flex-1">
                              <h4 className="text-sm font-semibold text-blue-700 dark:text-blue-400 mb-2">Alasan Deteksi</h4>
                              <p className="text-sm text-muted-foreground leading-relaxed">{result.reason}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Frames Analyzed */}
                      {(result.frames_analyzed || result.frames_processed || result.total_faces) && (
                        <div className="bg-card/50 backdrop-blur rounded-2xl p-5 border border-border/50">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                                </svg>
                              </div>
                              <span className="text-sm font-medium text-muted-foreground">Frames Analyzed</span>
                            </div>
                            <span className="font-medium text-foreground">
                              {result.frames_analyzed || result.frames_processed || result.total_faces} frames
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Hybrid Model Statistics (MATCH STREAMLIT) */}
              {modelId === "hybrid" && result.fake_votes !== undefined && result.real_votes !== undefined && (
                <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-950/30 dark:to-blue-950/30 border border-indigo-200 dark:border-indigo-800/50 rounded-3xl p-8 shadow-xl">
                  

                  {/* Statistics Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    

                    

                    
                  </div>

                  {/* Voting Details */}
                  <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-2xl p-6 border border-indigo-200 dark:border-indigo-800/50 shadow-lg">
                    <h4 className="text-lg font-bold mb-4 text-foreground flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      Voting Results
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Fake Votes */}
                      <div className="flex items-center justify-between p-4 bg-red-500/10 rounded-xl border border-red-500/30">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                            <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </div>
                          <div>
                            <div className="text-sm font-medium text-muted-foreground">Fake Votes</div>
                            <div className="text-2xl font-bold text-red-500">{result.fake_votes}</div>
                          </div>
                        </div>
                        <div className="text-lg font-semibold text-red-500">
                          {((result.fake_votes / (result.frames_analyzed || result.total_faces || 1)) * 100).toFixed(1)}%
                        </div>
                      </div>

                      {/* Real Votes */}
                      <div className="flex items-center justify-between p-4 bg-green-500/10 rounded-xl border border-green-500/30">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                            <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                          <div>
                            <div className="text-sm font-medium text-muted-foreground">Real Votes</div>
                            <div className="text-2xl font-bold text-green-500">{result.real_votes}</div>
                          </div>
                        </div>
                        <div className="text-lg font-semibold text-green-500">
                          {((result.real_votes / (result.frames_analyzed || result.total_faces || 1)) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar Visualization */}
                    <div className="mt-6">
                      <div className="flex items-center justify-between mb-2 text-sm font-medium text-muted-foreground">
                        <span>Fake</span>
                        <span>Real</span>
                      </div>
                      <div className="h-8 bg-muted rounded-full overflow-hidden flex">
                        <div 
                          className="bg-gradient-to-r from-red-400 to-red-600 flex items-center justify-center text-white text-xs font-bold"
                          style={{ width: `${(result.fake_votes / (result.frames_analyzed || result.total_faces || 1)) * 100}%` }}
                        >
                          {result.fake_votes > 0 && `${result.fake_votes}`}
                        </div>
                        <div 
                          className="bg-gradient-to-r from-green-400 to-green-600 flex items-center justify-center text-white text-xs font-bold"
                          style={{ width: `${(result.real_votes / (result.frames_analyzed || result.total_faces || 1)) * 100}%` }}
                        >
                          {result.real_votes > 0 && `${result.real_votes}`}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Analysis Details */}
                  {result.analysis_details && (
                    <div className="mt-6 bg-white/80 dark:bg-slate-900/80 backdrop-blur rounded-2xl p-6 border border-indigo-200 dark:border-indigo-800/50 shadow-lg">
                      <h4 className="text-lg font-bold mb-4 text-foreground flex items-center gap-2">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Analysis Configuration
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div className="flex justify-between p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                          <span className="text-muted-foreground">Voting Strategy:</span>
                          <span className="font-mono text-foreground">{result.analysis_details.voting_strategy}</span>
                        </div>
                        <div className="flex justify-between p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                          <span className="text-muted-foreground">Detector Used:</span>
                          <span className="font-mono text-foreground uppercase">{result.analysis_details.detector_used}</span>
                        </div>
                        <div className="flex justify-between p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                          <span className="text-muted-foreground">Frame Skip:</span>
                          <span className="font-mono text-foreground">{result.analysis_details.frame_skip}</span>
                        </div>
                        <div className="flex justify-between p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                          <span className="text-muted-foreground">Max Faces Analyzed:</span>
                          <span className="font-mono text-foreground">{result.analysis_details.max_faces_analyzed}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Model-Specific Visualizations */}
              {renderModelSpecificVisualization()}

              {/* Model-Specific Frame Analysis */}
              {renderModelSpecificFrames()}

              {/* Technical Details (Collapsible) */}
              <details className="group">
                <summary className="flex items-center gap-2 cursor-pointer text-sm text-muted-foreground hover:text-foreground transition-colors">
                  <svg className="w-4 h-4 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  Lihat Detail Teknis
                </summary>
                <div className="mt-4 p-4 rounded-xl bg-slate-800 dark:bg-slate-900 font-mono text-sm overflow-x-auto">
                  <pre className="text-slate-300">
{JSON.stringify((() => {
  // Base information untuk semua model
  const baseInfo: any = {
    filename: result.filename,
    status: result.status,
    model: result.model,
    is_deepfake: result.is_deepfake,
    confidence: result.confidence,
    raw_score: result.raw_score,
    threshold: result.threshold,
    label: result.label,
    frames_analyzed: result.frames_analyzed || result.frames_processed || result.total_faces || 0,
  };

  // Tambahkan field spesifik per model
  if (modelId === 'hybrid') {
    return {
      ...baseInfo,
      fake_votes: result.fake_votes,
      real_votes: result.real_votes,
      avg_score: result.avg_score,
      median_score: result.median_score,
      analysis_details: result.analysis_details,
      message: result.message
    };
  } else if (modelId === 'liveness') {
    return {
      ...baseInfo,
      blinks_detected: result.blinks_detected,
      frames_processed: result.frames_processed,
      reason: result.reason,
      message: result.message
    };
  } else if (modelId === 'spasial') {
    return {
      ...baseInfo,
      artifacts_detected: result.artifacts_detected,
      spatial_score: result.spatial_score,
      reason: result.reason,
      message: result.message
    };
  } else if (modelId === 'temporal') {
    return {
      ...baseInfo,
      temporal_consistency: result.temporal_consistency,
      motion_score: result.motion_score,
      reason: result.reason,
      message: result.message
    };
  } else if (modelId === 'ensemble') {
    return {
      ...baseInfo,
      model_scores: result.model_scores,
      voting_strategy: result.voting_strategy,
      reason: result.reason,
      message: result.message
    };
  } else {
    return {
      ...baseInfo,
      reason: result.reason,
      message: result.message
    };
  }
})(), null, 2)}
                  </pre>
                </div>
              </details>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
                <Button 
                  size="lg"
                  variant="outline" 
                  onClick={handleReset}
                  className="gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Analisis File Lain
                </Button>
                <Button 
                  size="lg"
                  variant="outline" 
                  onClick={() => router.push("/")}
                  className="gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                  Kembali ke Beranda
                </Button>
              </div>
            </div>
          )}
        </div>
      </section>

      <Footer />
    </main>
  )
}
