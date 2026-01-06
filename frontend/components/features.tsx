"use client"

const features = [
  {
    title: "DEEP-SPASIAL",
    description: "Model ini berfokus pada sidik jari visual yang ditinggalkan AI pada level piksel akibat proses sintesis deepfake. Deteksi dilakukan pada satu frame tanpa memperhatikan pergerakan. Model mencari anomali tekstur kulit, batas penyambungan wajah, noise, serta ketidakkonsistenan pencahayaan menggunakan arsitektur seperti CNN, XceptionNet, dan MesoNet.",
    image: "/image-spatial.jpg",
  },
  {
    title: "DEEP-TEMPORAL",
    description: "Model ini memeriksa hubungan antar frame pada video untuk mendeteksi deepfake berbasis pergerakan. Fokusnya adalah ketidakhalusan transisi antar frame, ketidakwajaran gerakan kepala & ekspresi, serta ketidakcocokan lip-sync dengan audio. Metode yang digunakan menggabungkan CNN untuk fitur spasial dan LSTM / RNN / Transformer untuk fitur temporal.",
    image: "/image-temporal.jpg",
  },
  {
    title: "LIVENESS & BIOMETRIC",
    description: "Tujuan model ini adalah mendeteksi tanda-tanda “kehidupan” nyata pada wajah manusia yang sulit direplikasi AI. Pendekatan yang digunakan adalah analisis kedipan mata Perceptual Eye Ratio (PER) dan pulsasi kulit berbasis sinyal Photoplethysmography (PPG) untuk melihat perubahan mikro warna akibat aliran darah. Jika area wajah tidak menunjukkan pola fisiologis manusia asli, video berpotensi deepfake.",
    image: "/image-liveness.jpg",
  },
  {
    title: "HYBRID AI + HUMAN DEEPFAKE",
    description: "Model ini ditujukan untuk deepfake tingkat lanjut yang telah diperhalus oleh manusia setelah dibuat AI, misalnya dengan smoothing, blur, atau color retouching untuk menutupi artefak asli. Model menggunakan Discriminator GAN berbasis forensik serta teknik Blind Image Quality Assessment (IQA) untuk menemukan pola kehalusan yang terlalu homogen dan tidak alami, jejak post-processing, dan watermark tak langsung dari alat editing.",
    image: "/image-hybrid.jpg",
  },
  
]

export default function Features() {
  return (
    <section className="w-full py-20 md:py-32 bg-gradient-to-b from-slate-900 to-slate-950">
      <div className="max-w-7xl mx-auto px- sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 text-balance">
            Deteksi Berbagai Jenis Deepfake
          </h2>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="group relative">
              {/* Image Container */}
              <div className="relative rounded-lg overflow-hidden mb-4 aspect-[4/3] bg-slate-800">
                <img
                  src={feature.image || "/placeholder.svg"}
                  alt={feature.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                {/* Dark Overlay */}
                <div className="absolute inset-0 bg-black/30 group-hover:bg-black/40 transition-all"></div>

                {/* Badge Label */}
                <div className="absolute top-4 left-4">
                  <span className="inline-block bg-white text-slate-900 text-xs font-bold px-3 py-2 rounded-full">
                    {feature.title}
                  </span>
                </div>
              </div>

              {/* Description */}
              <p className="text-slate-300 text-sm leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
