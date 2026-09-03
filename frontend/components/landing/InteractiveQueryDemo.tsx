'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import {
  Sparkles,
  ArrowRight,
  Eye,
  Scan,
  GitCompare,
  Combine,
  Check,
  Copy,
  RotateCcw,
  Compass
} from 'lucide-react'
import { PromptInputBox } from '@/components/ui/ai-prompt-box'

interface PresetScenario {
  id: string
  head: string
  headCode: string
  icon: React.ComponentType<{ className?: string }>
  title: string
  shortLabel: string
  query: string
  location: string
  sensor: string
  spectralBands: string
  metricLabel: string
  metricValue: string
  confidence: number
  latency: string
  vram: string
  summary: string
  details: string[]
  hash: string
  coordinates: string
}

const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'grounding',
    head: 'Spatial Grounding Head',
    headCode: 'rs_ground',
    icon: Scan,
    title: 'Solar Farm Localization',
    shortLabel: 'Solar Grounding',
    query: 'Highlight all solar farm installations and perimeter fences in the Jodhpur Sentinel-2 tile',
    location: 'Jodhpur District, Rajasthan',
    sensor: 'Sentinel-2 L2A Multispectral',
    spectralBands: '10m Optical Bands',
    metricLabel: 'Detected PV Footprint',
    metricValue: '142.8 Hectares',
    confidence: 0.94,
    latency: '1.12s',
    vram: '1.4 GB',
    summary: 'Extracted 4 contiguous photovoltaic solar arrays across sectors A-D with signature SWIR absorption. Polygon vector geometry and 15.2m perimeter clearance buffer verified.',
    details: [
      'Sub-pixel GeoJSON polygon with 18 boundary vertices',
      'Average photovoltaic panel albedo index: 0.48 within bounding mask',
      'Perimeter security fence co-referenced with boundary data'
    ],
    hash: '7c89f10a2b4e8910d54cfa2210e7b1a938c01d44',
    coordinates: '26.2389° N, 73.0243° E'
  },
  {
    id: 'fusion',
    head: 'Optical-SAR Fusion Head',
    headCode: 'rs_fusion',
    icon: Combine,
    title: 'All-Weather Flood Inundation',
    shortLabel: 'Flood Inundation',
    query: 'Fuse optical + SAR imagery to map flood inundation despite cloud cover in Cachar district',
    location: 'Barak River Basin, Assam',
    sensor: 'Sentinel-1 SAR + Sentinel-2 Co-registered',
    spectralBands: 'Optical-SAR Co-registered',
    metricLabel: 'Inundated Cropland',
    metricValue: '284.5 Hectares',
    confidence: 0.91,
    latency: '0.92s',
    vram: '1.8 GB',
    summary: 'Optical spectrum experienced 82% monsoon cloud overcast. Co-registered Sentinel-1 SAR microwave backscatter detected specular surface water reflection, accurately classifying inundated agricultural acreage.',
    details: [
      '100% cloud penetration verified via C-band microwave radar (VV/VH)',
      'Dual-sensor agreement index: 0.94 across validated pixels',
      'Submerged road network segment identified (3.4 km flagged)'
    ],
    hash: '8f9c4b1e72a091d355fce20014ba7902d18471c3',
    coordinates: '24.8333° N, 92.7789° E'
  },
  {
    id: 'change',
    head: 'Change Detection Head',
    headCode: 'rs_change',
    icon: GitCompare,
    title: 'Bi-Temporal Urban Expansion',
    shortLabel: 'Urban Expansion',
    query: 'Detect new built structures constructed between Nov 2024 and Jan 2026 in Sector 4',
    location: 'Bengaluru North Corridor, Karnataka',
    sensor: 'Sentinel-2 Multi-Temporal Pair',
    spectralBands: 'Bi-Temporal Siamese Pair',
    metricLabel: 'New Built Footprint',
    metricValue: '+18.4 Hectares',
    confidence: 0.88,
    latency: '1.35s',
    vram: '1.6 GB',
    summary: 'Siamese difference mapping detected 6 new commercial concrete building foundations and an extended 1.2km arterial feeder road. Natural vegetation loss measured at 14.2% across AOI.',
    details: [
      'Siamese feature delta: +31.8% reflectance shift in built-up index',
      '6 distinct structural clusters isolated and tagged with centroid coordinates',
      'Impervious surface expansion verified against historical baseline'
    ],
    hash: '4d12a9e331bfa829c719e0034a712bc90fa4112e',
    coordinates: '13.1986° N, 77.7066° E'
  },
  {
    id: 'vqa',
    head: 'Visual QA Head',
    headCode: 'rs_vqa',
    icon: Eye,
    title: 'Agricultural Land-Cover & Health',
    shortLabel: 'Crop Health VQA',
    query: 'What is the dominant land cover class and crop vegetative health index in this sector?',
    location: 'Ludhiana Agricultural Belt, Punjab',
    sensor: 'Sentinel-2 L2A Surface Reflectance',
    spectralBands: '13 Spectral Bands',
    metricLabel: 'Mean Crop NDVI',
    metricValue: '0.74 (Healthy)',
    confidence: 0.93,
    latency: '0.84s',
    vram: '1.2 GB',
    summary: 'Dominant land cover is irrigated winter wheat crop (62.4% of AOI), followed by irrigation canals (18.1%) and rural settlements (19.5%). High chlorophyll canopy vigor detected with minimal moisture stress.',
    details: [
      'Crop composition: 62.4% Wheat, 18.1% Water Bodies, 19.5% Built-up',
      'NDVI histogram centered at 0.74 indicating peak photosynthetic activity',
      'No critical soil salinity or drought anomalies detected along canal edges'
    ],
    hash: '2e45bb90a8814df61230cd89a3ef00827b1409ab',
    coordinates: '30.9010° N, 75.8573° E'
  }
]

export default function InteractiveQueryDemo() {
  const [activeScenario, setActiveScenario] = useState<PresetScenario>(PRESET_SCENARIOS[0])
  const [queryText, setQueryText] = useState(PRESET_SCENARIOS[0].query)
  const [isProcessing, setIsProcessing] = useState(false)
  const [copied, setCopied] = useState(false)
  const [activeSensorView, setActiveSensorView] = useState<'mask' | 'rgb' | 'sar'>('mask')

  const handleSelectPreset = (scenario: PresetScenario) => {
    setActiveScenario(scenario)
    setQueryText(scenario.query)
    setIsProcessing(false)
  }

  const handleSendQuery = (text: string) => {
    setIsProcessing(true)
    setTimeout(() => {
      setIsProcessing(false)
    }, 700)
  }

  const handleCopyHash = () => {
    navigator.clipboard.writeText(activeScenario.hash)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section id="demo" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-6xl">
        {/* Section Header Matching Landing Page */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="max-w-3xl">
            <p className="text-[11px] sm:text-xs font-semibold uppercase tracking-[.22em] text-[#8eb7ff]">
              Interactive Sandbox
            </p>
            <h2 className="mt-4 sm:mt-5 font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-balance text-white">
              Ask the Earth. Inspect the evidence.
            </h2>
          </div>
          <p className="max-w-md text-sm sm:text-base leading-relaxed text-white/60">
            Test the multimodal reasoning engine directly from your browser. Observe how the shared vision backbone routes natural language questions to specialized remote sensing heads.
          </p>
        </div>

        {/* Mission Scenario Navigation Chips */}
        <div className="mt-8 sm:mt-10 flex flex-wrap gap-2 sm:gap-3 items-center">
          <span className="font-mono text-[11px] uppercase tracking-wider text-white/40 mr-1 flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-[#8eb7ff]" />
            Scenarios:
          </span>
          {PRESET_SCENARIOS.map((sc) => {
            const Icon = sc.icon
            const isSelected = activeScenario.id === sc.id
            return (
              <button
                key={sc.id}
                onClick={() => handleSelectPreset(sc)}
                className={`inline-flex items-center gap-2 px-3.5 sm:px-4 py-2 rounded-xl text-xs sm:text-sm font-medium transition-colors cursor-pointer active:scale-[0.99] ${
                  isSelected
                    ? 'border border-white/40 bg-white/10 text-white'
                    : 'border border-white/10 bg-white/[0.02] text-white/60 hover:text-white hover:bg-white/[0.05]'
                }`}
              >
                <Icon className={`h-3.5 w-3.5 ${isSelected ? 'text-[#8eb7ff]' : 'text-white/30'}`} />
                <span>{sc.shortLabel}</span>
                <span className="font-mono text-[11px] text-[#8eb7ff]">
                  {sc.headCode}
                </span>
              </button>
            )
          })}
        </div>

        {/* MAIN INTERACTIVE CONSOLE FRAME (Matches TheReframeSection in LandingPage.tsx) */}
        <div className="mt-6 sm:mt-8 rounded-xl sm:rounded-2xl border border-white/15 bg-[#05080b] overflow-hidden">
          
          {/* Top Mission Status Bar */}
          <div className="flex flex-wrap items-center justify-between border-b border-white/15 bg-white/[0.02] px-4 py-3 sm:px-6 sm:py-4 font-mono text-[11px] sm:text-xs text-white/40 gap-2">
            <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
              <span className="text-white font-medium flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
                {activeScenario.location}
              </span>
              <span className="text-white/20">/</span>
              <span className="text-white/60">{activeScenario.sensor}</span>
              <span className="text-white/30 font-sans">({activeScenario.spectralBands})</span>
            </div>
            <div className="flex items-center gap-3 text-white/40">
              <span>HEAD: <span className="text-white font-medium">{activeScenario.headCode}</span></span>
              <span className="text-white/20">/</span>
              <span>LATENCY: <span className="text-[#8eb7ff]">{activeScenario.latency}</span></span>
            </div>
          </div>

          {/* ========================================================= */}
          {/* 1. OUTPUT & WORKSPACE (PLACED PROMINENTLY ABOVE)          */}
          {/* ========================================================= */}
          <div className="p-4 sm:p-6 lg:p-8">
            {/* Output Status Header */}
            <div className="flex flex-wrap items-center justify-between border-b border-white/10 pb-3 sm:pb-4 mb-4 sm:mb-6 font-mono text-xs text-white/60 gap-3">
              <div>
                <span className="text-[10px] sm:text-[11px] uppercase tracking-wider text-white/40 block">Skill Output</span>
                <p className="text-xs sm:text-sm font-medium text-white mt-0.5">{activeScenario.head}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
                  CONFIDENCE {(activeScenario.confidence * 100).toFixed(0)}% PASS
                </span>
                <span className="text-white/40 text-[11px] font-mono">{activeScenario.coordinates}</span>
              </div>
            </div>

            {/* Main Split: Left Satellite Canvas, Right Analyst Report */}
            <div className="grid lg:grid-cols-12 gap-5 lg:gap-6 items-stretch">
              
              {/* Left Column: Satellite Canvas (7 Cols) */}
              <div className="lg:col-span-7 flex flex-col justify-between rounded-xl border border-white/10 bg-black/40 p-4 sm:p-5 font-mono text-xs">
                <div>
                  {/* Layer View Switcher */}
                  <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                    <div className="flex items-center gap-1 rounded-lg border border-white/10 bg-white/[0.02] p-1 text-[11px]">
                      <button
                        onClick={() => setActiveSensorView('mask')}
                        className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                          activeSensorView === 'mask'
                            ? 'bg-white/15 text-white font-medium'
                            : 'text-white/40 hover:text-white'
                        }`}
                      >
                        Detection Mask
                      </button>
                      <button
                        onClick={() => setActiveSensorView('rgb')}
                        className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                          activeSensorView === 'rgb'
                            ? 'bg-white/15 text-white font-medium'
                            : 'text-white/40 hover:text-white'
                        }`}
                      >
                        Natural Optical
                      </button>
                      <button
                        onClick={() => setActiveSensorView('sar')}
                        className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                          activeSensorView === 'sar'
                            ? 'bg-white/15 text-white font-medium'
                            : 'text-white/40 hover:text-white'
                        }`}
                      >
                        SAR Microwave Radar
                      </button>
                    </div>
                    <span className="text-[10px] text-white/35">10m GSD Resolution</span>
                  </div>

                  {/* Satellite Orthophoto Render Canvas */}
                  <div className="relative aspect-[16/10] w-full rounded-lg border border-white/10 overflow-hidden bg-[#05080b] shadow-inner select-none">
                    <svg className="w-full h-full" viewBox="0 0 600 375" preserveAspectRatio="none">
                      <defs>
                        {/* Shaded terrain gradients */}
                        <linearGradient id="desertEarth" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#0c1117" />
                          <stop offset="50%" stopColor="#141a22" />
                          <stop offset="100%" stopColor="#080c10" />
                        </linearGradient>

                        <linearGradient id="sarGrayscale" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#06090d" />
                          <stop offset="50%" stopColor="#121720" />
                          <stop offset="100%" stopColor="#080b0f" />
                        </linearGradient>

                        <pattern id="solarGrid" width="16" height="12" patternUnits="userSpaceOnUse">
                          <rect width="15" height="11" fill="#131922" stroke="#1d2633" strokeWidth="0.8" />
                          <line x1="0" y1="6" x2="15" y2="6" stroke="#2c3b4e" strokeWidth="0.5" />
                        </pattern>

                        <pattern id="radarSpeckle" width="20" height="20" patternUnits="userSpaceOnUse">
                          <circle cx="3" cy="5" r="0.8" fill="#ffffff" opacity="0.2" />
                          <circle cx="12" cy="14" r="0.6" fill="#ffffff" opacity="0.15" />
                          <circle cx="17" cy="7" r="0.9" fill="#ffffff" opacity="0.25" />
                        </pattern>
                      </defs>

                      {/* Terrain Base */}
                      <rect width="600" height="375" fill={activeSensorView === 'sar' ? 'url(#sarGrayscale)' : 'url(#desertEarth)'} />

                      {/* Topographic Contours */}
                      <path
                        d="M -20,130 Q 140,90 260,160 T 620,120 L 620,250 Q 420,290 260,210 T -20,230 Z"
                        fill="#0e131b"
                        opacity="0.9"
                      />
                      <path
                        d="M -20,270 Q 180,220 380,320 T 620,270 L 620,380 L -20,380 Z"
                        fill="#121822"
                        opacity="0.9"
                      />

                      {/* Solar Farm Arrays (When in Solar Scenario) */}
                      {activeScenario.id === 'grounding' && activeSensorView !== 'sar' && (
                        <g opacity="0.85">
                          <polygon points="150,110 270,90 330,165 300,230 185,250 130,175" fill="url(#solarGrid)" stroke="#3a485c" strokeWidth="0.8" />
                          <polygon points="360,170 450,150 490,210 430,240 350,200" fill="url(#solarGrid)" stroke="#3a485c" strokeWidth="0.8" />
                        </g>
                      )}

                      {/* Radar Speckle in SAR view */}
                      {activeSensorView === 'sar' && (
                        <rect width="600" height="375" fill="url(#radarSpeckle)" />
                      )}

                      {/* Delineation / Segmentation Overlay */}
                      {activeSensorView === 'mask' && (
                        <g className="transition-all duration-300">
                          <polygon
                            points="150,110 270,90 330,165 300,230 185,250 130,175"
                            fill="rgba(255, 255, 255, 0.08)"
                            stroke="#ffffff"
                            strokeWidth="1.8"
                            strokeDasharray="4 2"
                          />
                          <polygon
                            points="360,170 450,150 490,210 430,240 350,200"
                            fill="rgba(255, 255, 255, 0.05)"
                            stroke="#ffffff"
                            strokeWidth="1.2"
                          />

                          {/* Vertices Pins */}
                          {[
                            [150, 110], [270, 90], [330, 165], [300, 230], [185, 250], [130, 175],
                            [360, 170], [450, 150], [490, 210], [430, 240], [350, 200]
                          ].map(([vx, vy], idx) => (
                            <circle key={idx} cx={vx} cy={vy} r="2.5" fill="#ffffff" />
                          ))}

                          {/* Target Crosshair */}
                          <line x1="225" y1="168" x2="245" y2="168" stroke="#8eb7ff" strokeWidth="1.5" />
                          <line x1="235" y1="158" x2="235" y2="178" stroke="#8eb7ff" strokeWidth="1.5" />

                          {/* AOI Banner */}
                          <rect x="250" y="156" width="135" height="22" rx="4" fill="#000000" fillOpacity="0.85" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
                          <text x="256" y="171" fill="#ffffff" fontSize="10" fontFamily="monospace" fontWeight="500">
                            AOI: {activeScenario.metricValue}
                          </text>
                        </g>
                      )}

                      {/* Coordinates Grid Lines */}
                      <line x1="150" y1="0" x2="150" y2="375" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" strokeDasharray="3 3" />
                      <line x1="350" y1="0" x2="350" y2="375" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" strokeDasharray="3 3" />
                    </svg>

                    {/* HUD Coordinates Bar */}
                    <div className="absolute top-2.5 left-2.5 rounded bg-black/80 px-2 py-1 font-mono text-[10px] text-white/70 border border-white/10 flex items-center gap-1.5">
                      <Compass className="h-3 w-3 text-white/50" />
                      <span>{activeScenario.coordinates}</span>
                    </div>

                    <div className="absolute bottom-2.5 right-2.5 rounded bg-black/80 px-2 py-1 font-mono text-[10px] text-white/70 border border-white/10">
                      {activeSensorView === 'mask' ? 'GEOJSON VECTOR DELINEATION' : activeSensorView.toUpperCase()}
                    </div>
                  </div>
                </div>

                {/* Telemetry Metric Pill */}
                <div className="mt-3 flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-3 font-mono text-xs">
                  <div>
                    <span className="text-white/40 text-[10px] uppercase block">{activeScenario.metricLabel}</span>
                    <span className="text-white font-medium text-sm sm:text-base">{activeScenario.metricValue}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-white/40 text-[10px] uppercase block">Hardware Footprint</span>
                    <span className="text-white font-medium">{activeScenario.vram} VRAM</span>
                  </div>
                </div>
              </div>

              {/* Right Column: Conversational Synthesis & Receipt (5 Cols) */}
              <div className="lg:col-span-5 flex flex-col justify-between rounded-xl border border-white/10 bg-black/40 p-4 sm:p-5 font-mono text-xs">
                <div>
                  {/* Synthesis Card */}
                  <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3.5 sm:p-4 mb-4">
                    <p className="text-[10px] uppercase tracking-wider text-[#8eb7ff] font-mono">
                      Conversational Synthesis
                    </p>
                    <p className="mt-1.5 font-sans text-xs sm:text-sm text-white/90 leading-relaxed">
                      {activeScenario.summary}
                    </p>
                  </div>

                  {/* Verification Evidence */}
                  <div className="space-y-2 mb-4">
                    <span className="text-[10px] uppercase tracking-wider text-white/40 font-mono block">
                      Verification Evidence:
                    </span>
                    {activeScenario.details.map((detail, idx) => (
                      <div key={idx} className="flex items-start gap-2 text-xs text-white/70">
                        <Check className="h-3.5 w-3.5 text-emerald-300 shrink-0 mt-0.5" />
                        <span className="leading-snug">{detail}</span>
                      </div>
                    ))}
                  </div>

                  {/* Execution Receipt Matching Section 3 in LandingPage */}
                  <div className="rounded-lg border border-white/10 bg-black/50 p-3 font-mono text-[11px] text-white/60">
                    <div className="flex items-center justify-between text-white/40 text-[10px] mb-1.5">
                      <span>EXECUTION RECEIPT</span>
                      <span>SQ-2026-X89F</span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-white/70 font-mono text-[10px]">
                        SHA-256: {activeScenario.hash.slice(0, 18)}...
                      </span>
                      <button
                        onClick={handleCopyHash}
                        className="inline-flex items-center gap-1 rounded border border-white/20 bg-white/5 px-2 py-0.5 text-[10px] text-white/80 transition hover:border-white/40 hover:text-white cursor-pointer"
                      >
                        {copied ? <Check className="h-3 w-3 text-emerald-300" /> : <Copy className="h-3 w-3" />}
                        <span>{copied ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Workspace CTA Button */}
                <div className="mt-5 pt-3.5 border-t border-white/10 flex items-center gap-3">
                  <Link
                    href={`/chat?q=${encodeURIComponent(queryText)}`}
                    className="flex-1 rounded-full bg-white px-5 py-2.5 text-center text-xs sm:text-sm font-semibold text-black transition hover:bg-white/85 active:scale-[0.98] flex items-center justify-center gap-1.5 shadow-lg cursor-pointer"
                  >
                    <span>Open in full workspace</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                  <button
                    onClick={() => handleSelectPreset(activeScenario)}
                    className="rounded-full border border-white/20 p-2 text-white/60 hover:text-white hover:border-white/40 transition cursor-pointer"
                    title="Reset scenario"
                    aria-label="Reset scenario"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* ========================================================= */}
          {/* 2. PROMPT INPUT BAR (CORRECTLY POSITIONED AT BOTTOM)      */}
          {/* ========================================================= */}
          <div className="border-t border-white/15 bg-white/[0.01] p-4 sm:p-5 lg:p-6">
            
            {/* Quick Suggestions Chips */}
            <div className="mb-3 flex items-center gap-2 overflow-x-auto scrollbar-none text-[11px] font-mono text-white/40">
              <span className="shrink-0">Try asking:</span>
              {PRESET_SCENARIOS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setActiveScenario(p)
                    setQueryText(p.query)
                  }}
                  className={`shrink-0 px-2.5 py-1 rounded-lg border transition-colors cursor-pointer text-left truncate max-w-[280px] ${
                    activeScenario.id === p.id
                      ? 'border-white/30 bg-white/10 text-white'
                      : 'border-white/10 bg-white/[0.02] text-white/50 hover:text-white hover:border-white/20'
                  }`}
                >
                  "{p.query}"
                </button>
              ))}
            </div>

            {/* The PromptInputBox Component at the bottom */}
            <PromptInputBox
              value={queryText}
              onValueChange={setQueryText}
              onSend={handleSendQuery}
              isLoading={isProcessing}
              placeholder="Ask a question about this satellite tile or enter a query..."
            />

            {/* Hint Line */}
            <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-white/40">
              <span>RS-CLIP shared backbone: <strong className="text-white/70">Ingest → Cache → Dispatch</strong></span>
              <span>Press <kbd className="rounded border border-white/20 bg-white/5 px-1 py-0.5 text-white/80">Enter ↵</kbd></span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
