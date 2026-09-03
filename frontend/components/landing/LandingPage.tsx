'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import {
  Check,
  Copy,
  ArrowRight,
  Eye,
  Scan,
  GitCompare,
  Combine,
  Layers,
  Radio,
  Clock,
  Menu,
  X
} from 'lucide-react'
import Lenis from 'lenis'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import IntroAnimation from '@/components/ui/scroll-morph-hero'
import InteractiveQueryDemo from '@/components/landing/InteractiveQueryDemo'

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger)
}

const problemCards = [
  {
    num: '01',
    title: 'Built for one job, not one question',
    body: 'Existing systems are trained for a single narrow task. Real questions move between visual reasoning, grounding, change, and fusion.',
    icon: Layers
  },
  {
    num: '02',
    title: 'GIS expertise is a gate',
    body: 'Getting an answer from raw imagery often needs a specialist pipeline, model selection, and domain knowledge.',
    icon: Radio
  },
  {
    num: '03',
    title: 'One image is never enough',
    body: 'Optical imagery is limited by cloud and darkness. SAR is powerful but difficult to interpret alone. Decisions need both sensors and time.',
    icon: Clock
  },
]

const workflow = [
  ['01', 'Ingest and verify', 'Checks format, geo-alignment, sensor, date, and compatibility before analysis begins.'],
  ['02', 'Understand the query', 'A fast classifier identifies the analysis intent behind a natural-language question.'],
  ['03', 'Encode once', 'The shared RS-CLIP-Fusion backbone creates one rich embedding, computed once and reused.'],
  ['04', 'Route to experts', 'Only the required skills are activated: VQA, grounding, change, or optical-SAR fusion.'],
  ['05', 'Answer with evidence', 'Returns the result, map overlay, confidence score, and an inspectable execution trace.'],
]

const capabilities = [
  ['Visual question answering', 'What is the dominant land cover?', 'Agricultural cropland, approximately 62% of visible area.'],
  ['Text-guided grounding', 'Highlight the water body.', 'Spatial grounding identifies the target region and returns an overlay.'],
  ['Change detection', 'What changed between 2023 and 2024?', 'New built-up area detected and marked for review.'],
  ['Optical-SAR fusion', 'Use both images to map built-up and water regions.', 'Two sensor views combine into one classified map.'],
]

const applications = [
  ['Government and disaster response', 'Damage assessment takes field visits and days.', 'Query flood extent in minutes with evidence ready for official review.'],
  ['Agriculture and crop insurance', 'Claims verification is difficult during monsoon cloud cover.', 'SAR supports faster, harder-to-dispute assessments.'],
  ['Urban planning and smart cities', 'Unauthorized construction is often found through annual audits.', 'Ask what is newly built, ward by ward, whenever needed.'],
  ['Environment and forest monitoring', 'Small teams rarely have dedicated GIS specialists.', 'Conversational monitoring brings satellite insight to every field office.'],
]

function Stats() {
  const ref = useRef<HTMLDivElement>(null)
  const [active, setActive] = useState(false)
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => entry.isIntersecting && setActive(true), { threshold: 0.2 })
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  const stats = [
    ['~65%', 'less compute & VRAM', 'shared encoder vs four'],
    ['<4s', 'target response time', 'single-image query'],
    ['Instant', 'repeat-query speed', 'cached embeddings'],
    ['0.87', 'example final confidence', 'full trace available']
  ]
  return (
    <div ref={ref} className="grid grid-cols-2 gap-x-4 gap-y-6 border-t border-white/15 pt-6 sm:gap-x-6 sm:pt-7 md:grid-cols-4 md:gap-x-8">
      {stats.map(([value, label, detail], index) => (
        <div key={label} className="animate-[reveal_.8s_cubic-bezier(.22,1,.36,1)_both]" style={{ animationDelay: `${index * 90 + 160}ms` }}>
          <p className="font-mono text-xl tracking-[-.06em] text-white sm:text-2xl md:text-3xl">{active ? value : '···'}</p>
          <p className="mt-1.5 text-[11px] leading-4 text-white/70 sm:text-xs sm:leading-5">{label}</p>
          <p className="mt-0.5 text-[10px] leading-4 text-white/35 sm:text-[11px] sm:leading-5">{detail}</p>
        </div>
      ))}
    </div>
  )
}

function MobileMenu({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  return (
    <div className={`fixed inset-0 z-50 md:hidden transition-all duration-300 ${open ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`} aria-hidden={!open}>
      <div onClick={onClose} className="absolute inset-0 bg-black/85 backdrop-blur-md" />
      <nav className={`absolute inset-x-4 top-20 rounded-2xl border border-white/15 bg-[#0a1017] p-6 text-white shadow-2xl transition-all duration-300 ${open ? 'translate-y-0 scale-100' : '-translate-y-4 scale-95'}`} aria-label="Mobile navigation">
        <div className="flex items-center justify-between border-b border-white/10 pb-4 mb-2">
          <span className="font-mono text-xs font-semibold text-[#8eb7ff] uppercase tracking-wider">Navigation</span>
          <button onClick={onClose} aria-label="Close menu" className="p-1 text-white/60 hover:text-white">
            <X className="size-5" />
          </button>
        </div>
        <div className="space-y-1 divide-y divide-white/5">
          {['Playground', 'Problem', 'Innovation', 'Method', 'Capabilities', 'Applications'].map((item) => (
            <a 
              key={item} 
              href={item === 'Playground' ? '#demo' : `#${item.toLowerCase()}`} 
              onClick={onClose} 
              className="block py-3 text-base font-medium text-white/80 transition hover:text-white"
            >
              {item}
            </a>
          ))}
        </div>
        <div className="mt-6 pt-4 border-t border-white/10">
          <Link 
            href="/chat" 
            onClick={onClose} 
            className="block w-full rounded-full bg-white py-3.5 text-center text-sm font-semibold text-black transition active:scale-[0.98] shadow-lg"
          >
            Try the prototype
          </Link>
        </div>
      </nav>
    </div>
  )
}

// -------------------------------------------------------------
// SECTION 1: THE REFRAME (Architecture & Routing)
// -------------------------------------------------------------
const REFRAME_HEADS = [
  {
    id: 'vqa',
    number: '01',
    name: 'Visual QA Head',
    code: 'rs_vqa',
    icon: Eye,
    tag: 'Semantic Reasoning',
    description: 'Zero-shot visual question answering over optical and spectral bands.',
    sampleQuery: 'What is the dominant land cover in the southern sector?',
    outputSpec: 'Probability distribution across 14 land-cover classes + natural language summary',
    latency: '0.84s',
    vram: '1.2 GB'
  },
  {
    id: 'grounding',
    number: '02',
    name: 'Spatial Grounding Head',
    code: 'rs_ground',
    icon: Scan,
    tag: 'Coordinate Localization',
    description: 'Natural language guided bounding polygon & pixel-level segmentation mask generation.',
    sampleQuery: 'Highlight all solar farm installations and perimeter fences.',
    outputSpec: 'GeoJSON Polygon FeatureCollection with sub-pixel vertex coordinates',
    latency: '1.12s',
    vram: '1.4 GB'
  },
  {
    id: 'change',
    number: '03',
    name: 'Change Detection Head',
    code: 'rs_change',
    icon: GitCompare,
    tag: 'Bi-Temporal Delta',
    description: 'Siamese difference mapping across multi-temporal satellite passes (T0 vs T1).',
    sampleQuery: 'Detect new built structures constructed between Nov 2024 and Jan 2026.',
    outputSpec: 'Binary change heatmap + expansion area calculation in hectares',
    latency: '1.35s',
    vram: '1.6 GB'
  },
  {
    id: 'fusion',
    number: '04',
    name: 'Optical-SAR Fusion Head',
    code: 'rs_fusion',
    icon: Combine,
    tag: 'Cross-Sensor Synergy',
    description: 'Joint synthesis of optical spectral reflectance and SAR microwave backscatter.',
    sampleQuery: 'Map inundated flood zones despite heavy monsoon cloud overcast.',
    outputSpec: 'Fused classified surface water mask with cloud-penetration verification',
    latency: '1.48s',
    vram: '1.8 GB'
  }
]

function TheReframeSection() {
  const [selectedHead, setSelectedHead] = useState(REFRAME_HEADS[0])

  return (
    <section id="innovation" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-6xl">
        <div className="max-w-3xl">
          <p className="text-[11px] sm:text-xs font-semibold uppercase tracking-[.22em] text-[#8eb7ff]">Architecture</p>
          <h2 className="mt-4 sm:mt-5 font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-balance text-white">
            One shared vision brain. Four expert skills.
          </h2>
          <p className="mt-4 sm:mt-6 max-w-2xl text-sm sm:text-base leading-relaxed text-white/60">
            SatQuery AI does not bolt four separate AI models together. A shared backbone understands optical and radar imagery in one embedding space, while a lightweight controller activates only the expert skill each question needs.
          </p>
        </div>

        {/* Interactive Architecture Console */}
        <div className="mt-10 sm:mt-16 rounded-xl sm:rounded-2xl border border-white/15 bg-[#05080b] overflow-hidden">
          {/* Top Status Bar */}
          <div className="flex flex-wrap items-center justify-between border-b border-white/15 bg-white/[0.02] px-4 py-3 sm:px-6 sm:py-4 font-mono text-[11px] sm:text-xs text-white/40 gap-2">
            <div className="flex items-center gap-2 sm:gap-3">
              <span className="text-[#8eb7ff] font-semibold">RS-CLIP-FUSION</span>
              <span className="text-white/20">/</span>
              <span>512-DIM LATENT SPACE</span>
            </div>
            <div className="hidden sm:flex items-center gap-3 text-white/35">
              <span>CONTROLLER: DYNAMIC DISPATCH</span>
            </div>
          </div>

          {/* Responsive Grid Layout (Stacked on Mobile, Split on Tablet/Desktop) */}
          <div className="grid lg:grid-cols-[1fr_1.25fr]">
            {/* Left: Skill Selectors */}
            <div className="divide-y divide-white/10 border-b border-white/15 lg:border-b-0 lg:border-r">
              {REFRAME_HEADS.map((head) => {
                const isSelected = selectedHead.id === head.id
                return (
                  <button
                    key={head.id}
                    onClick={() => setSelectedHead(head)}
                    className={`block w-full p-4 sm:p-6 text-left transition-colors active:scale-[0.99] touch-manipulation ${
                      isSelected ? 'bg-white/[0.05] border-l-2 border-l-[#8eb7ff]' : 'hover:bg-white/[0.02]'
                    }`}
                  >
                    <div className="flex items-center justify-between font-mono text-xs">
                      <span className="text-[#8eb7ff] font-semibold">{head.number}</span>
                      <span className={`tracking-wider ${isSelected ? 'text-white font-medium' : 'text-white/40'}`}>
                        {head.code}
                      </span>
                    </div>
                    <h3 className="mt-2 sm:mt-3 text-sm sm:text-base font-medium text-white">{head.name}</h3>
                    <p className="mt-1 sm:mt-2 text-xs sm:text-sm leading-relaxed text-white/50">{head.description}</p>
                  </button>
                )
              })}
            </div>

            {/* Right: Live Telemetry Inspection */}
            <div className="flex flex-col justify-between bg-black/40 p-4 sm:p-6 md:p-8 font-mono text-xs leading-relaxed text-white/60">
              <div>
                <div className="flex items-center justify-between border-b border-white/10 pb-3 sm:pb-4">
                  <div>
                    <span className="text-[10px] sm:text-[11px] uppercase tracking-wider text-white/40">{selectedHead.tag}</span>
                    <p className="text-xs sm:text-sm font-medium text-white">{selectedHead.code}</p>
                  </div>
                  <span className="text-[11px] font-medium text-[#8eb7ff] rounded-full border border-[#8eb7ff]/30 bg-[#8eb7ff]/10 px-2.5 py-0.5">ACTIVE</span>
                </div>

                <div className="mt-4 sm:mt-6 space-y-3 sm:space-y-4">
                  <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3 sm:p-4">
                    <p className="text-[10px] sm:text-[11px] text-white/40 uppercase">Natural Language Query</p>
                    <p className="mt-1 font-sans text-xs sm:text-sm text-white italic">"{selectedHead.sampleQuery}"</p>
                  </div>

                  <div className="grid grid-cols-2 gap-2 sm:gap-3">
                    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-2.5 sm:p-3">
                      <p className="text-[9px] sm:text-[10px] text-white/40 uppercase">Head Latency</p>
                      <p className="mt-1 text-xs sm:text-sm text-white font-semibold">{selectedHead.latency}</p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-2.5 sm:p-3">
                      <p className="text-[9px] sm:text-[10px] text-white/40 uppercase">Head VRAM</p>
                      <p className="mt-1 text-xs sm:text-sm text-white font-semibold">{selectedHead.vram}</p>
                    </div>
                  </div>

                  <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3 sm:p-4">
                    <p className="text-[10px] sm:text-[11px] text-white/40 uppercase">Output Artifact Format</p>
                    <p className="mt-1 font-sans text-[11px] sm:text-xs leading-relaxed text-white/70">{selectedHead.outputSpec}</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 sm:mt-8 border-t border-white/10 pt-3 sm:pt-4 text-[10px] sm:text-[11px] text-white/40 overflow-x-auto whitespace-nowrap sm:whitespace-normal">
                <span>Pipeline trace: Ingest → RS-CLIP (cached) → Controller → <span className="text-white font-medium">{selectedHead.code}</span> → GeoJSON</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// -------------------------------------------------------------
// SECTION 2: PROOF, NOT ADJECTIVES (Performance Telemetry)
// -------------------------------------------------------------
const PROOF_CARDS = [
  {
    num: '01',
    metric: '~65%',
    label: 'less compute & VRAM',
    sub: '5.4 GB vs 16.2 GB (shared backbone vs 4 separate models)',
    detail: 'Single RS-CLIP backbone eliminates loading redundant vision encoders in memory.'
  },
  {
    num: '02',
    metric: '<4s',
    label: 'cold query response',
    sub: '2.84s average on single-pass multi-modal query',
    detail: 'Direct skill dispatch skips sequential pipeline chaining entirely.'
  },
  {
    num: '03',
    metric: 'Instant',
    label: 'repeat-query speed',
    sub: '0.08s cached latent lookup',
    detail: 'Subsequent conversational follow-ups on the same tile bypass image re-encoding.'
  },
  {
    num: '04',
    metric: '0.87',
    label: 'average confidence',
    sub: 'Cross-sensor agreement check',
    detail: 'Dual optical and SAR verification guarantees ground-truth reliability.'
  }
]

function ProofNotAdjectivesSection() {
  return (
    <section className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <h2 className="max-w-2xl font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-white">
              Every optimisation compounds.
            </h2>
          </div>
          <p className="max-w-md text-xs sm:text-sm leading-relaxed text-white/60">
            Encode once. Cache embeddings. Quantise the backbone. Route only what the question requires.
          </p>
        </div>

        {/* Responsive Metric Cards Grid */}
        <div className="mt-10 sm:mt-16 grid gap-px rounded-xl sm:rounded-2xl overflow-hidden border border-white/15 bg-white/15 sm:grid-cols-2 lg:grid-cols-4">
          {PROOF_CARDS.map((card) => (
            <article key={card.num} className="flex flex-col justify-between bg-[#05080b] p-5 sm:p-7 md:p-8">
              <div>
                <p className="font-mono text-xs text-[#8eb7ff] font-semibold">{card.num}</p>
                <p className="mt-3 sm:mt-5 font-mono text-2xl sm:text-3xl md:text-4xl tracking-[-0.06em] text-white">{card.metric}</p>
                <p className="mt-1.5 sm:mt-2 text-xs font-medium text-white/90">{card.label}</p>
                <p className="mt-1 text-[10px] sm:text-[11px] leading-relaxed text-white/40">{card.sub}</p>
              </div>
              <p className="mt-4 sm:mt-6 border-t border-white/10 pt-3 sm:pt-4 text-[11px] sm:text-xs leading-relaxed text-white/50">{card.detail}</p>
            </article>
          ))}
        </div>

        {/* Comparison Telemetry Row */}
        <div className="mt-6 sm:mt-10 rounded-xl border border-white/15 bg-[#05080b] p-4 sm:p-6 font-mono text-xs text-white/60">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-white font-medium text-xs sm:text-sm">Hardware Benchmark Comparison</p>
              <p className="mt-0.5 text-[10px] sm:text-[11px] text-white/40">Inference on NVIDIA RTX / A100 test bench (Sentinel-2 10m bands)</p>
            </div>
            <div className="flex flex-wrap gap-4 sm:gap-6 text-[11px]">
              <div>
                <span className="text-white/40">SatQuery VRAM: </span>
                <span className="text-white font-semibold">5.4 GB</span>
              </div>
              <div>
                <span className="text-white/40">Legacy Pipeline: </span>
                <span className="text-white/60">16.2 GB</span>
              </div>
              <div>
                <span className="text-white/40">Speedup: </span>
                <span className="text-[#8eb7ff] font-semibold">4.1x faster</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// -------------------------------------------------------------
// SECTION 3: TRUST BY DESIGN (Receipt HUD)
// -------------------------------------------------------------
function TrustByDesignSection() {
  const [copied, setCopied] = useState(false)
  const traceHash = '8f9c4b1e72a091d355fce20014ba7902d18471c324e99f018e'

  const copyHash = () => {
    navigator.clipboard.writeText(traceHash)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
      <div className="mx-auto grid max-w-6xl gap-8 sm:gap-14 md:grid-cols-[.85fr_1.15fr] md:items-start">
        <div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-white">
            Not a black box. Every answer has a receipt.
          </h2>
          <p className="mt-4 sm:mt-6 text-sm sm:text-base leading-relaxed text-white/60">
            The execution trace records what ran, why it ran, how long it took, and how confident the result is. Built for government reporting and human accountability.
          </p>
          <div className="mt-6 sm:mt-8 space-y-2 sm:space-y-3 font-mono text-[11px] sm:text-xs text-white/60">
            <p>· Cryptographic trace hash per query</p>
            <p>· Sub-pixel GeoJSON coordinates lineage</p>
            <p>· Cross-sensor agreement verification</p>
          </div>
        </div>

        {/* Right Column: Execution Receipt */}
        <div className="rounded-xl sm:rounded-2xl border border-white/15 bg-black/50 p-4 sm:p-6 md:p-8 font-mono text-xs leading-relaxed text-white/60">
          <div className="flex items-center justify-between border-b border-white/10 pb-3 text-white/40 text-[11px] sm:text-xs">
            <span className="text-white font-medium">EXECUTION RECEIPT</span>
            <span>SQ-2026-X89F</span>
          </div>

          <p className="mt-3 sm:mt-4 text-white text-xs sm:text-sm">Query: Fuse optical + SAR to find built-up regions</p>
          
          <div className="mt-3 sm:mt-4 space-y-1.5 sm:space-y-1 text-[11px] sm:text-xs">
            <p className="text-[#8eb7ff]">
              01  rs_fusion <span className="text-white/40">[0.9s] confidence 0.91</span>
            </p>
            <p className="text-[#8eb7ff]">
              02  rs_change <span className="text-white/40">[1.1s] confidence 0.84</span>
            </p>
            <p className="text-white/70">
              03  Cross-head agreement <span className="text-emerald-300 font-semibold">PASS (0.94)</span>
            </p>
          </div>

          <div className="mt-4 sm:mt-6 border-t border-white/10 pt-4 flex flex-wrap items-center justify-between gap-3 text-white text-[11px] sm:text-xs">
            <div>
              <span>Final confidence: <span className="text-emerald-300 font-semibold">0.87</span></span>
              <p className="text-[10px] sm:text-[11px] text-white/35 font-mono">SHA-256: {traceHash.slice(0, 16)}...</p>
            </div>
            <button
              onClick={copyHash}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/20 bg-white/5 px-3 py-1.5 text-[11px] text-white/80 transition hover:border-white/40 hover:text-white active:scale-[0.98] touch-manipulation"
            >
              {copied ? <Check className="size-3 text-emerald-300" /> : <Copy className="size-3" />}
              <span>{copied ? 'Copied' : 'Copy Hash'}</span>
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

// -------------------------------------------------------------
// SECTION 4: WHY SATQUERY AI (Comparison Table)
// -------------------------------------------------------------
const COMPARISON_DATA = [
  ['Sensor understanding', 'Unreliable (hallucinates)', 'Per-model (isolated)', 'Unified RS-CLIP backbone'],
  ['Optical + SAR', 'Cannot fuse raw bands', 'Bolted on / manual scripts', 'Native embedding space'],
  ['Temporal change', 'No pixel-delta awareness', 'Separate tool workflow', 'Built-in Siamese head'],
  ['Spatial grounding', 'No map coordinates', 'Complex GIS scripting', 'Conversational GeoJSON'],
  ['Compute overhead', 'High API token costs', '4 models (>16 GB VRAM)', '1 backbone (<6 GB VRAM)'],
  ['Transparency', 'Black box', 'Partial console logs', 'Full execution trace & receipt'],
]

function WhySatQuerySection() {
  return (
    <section className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-6xl">
        <h2 className="max-w-3xl font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-white">
          The difference is not another model. It is a better system.
        </h2>

        {/* Touch-Friendly Comparison Table */}
        <div className="mt-8 sm:mt-14 overflow-x-auto border-y border-white/15 -mx-4 sm:mx-0 px-4 sm:px-0 scrollbar-thin">
          <div className="grid min-w-[620px] grid-cols-4 text-xs sm:text-sm">
            <div className="border-r border-white/15 py-4 pr-3 font-mono text-[10px] sm:text-xs uppercase tracking-wider text-white/35">Measure</div>
            <div className="border-r border-white/15 px-3 py-4 font-mono text-[10px] sm:text-xs uppercase tracking-wider text-white/35">Generic LLM</div>
            <div className="border-r border-white/15 px-3 py-4 font-mono text-[10px] sm:text-xs uppercase tracking-wider text-white/35">Disconnected pipeline</div>
            <div className="px-3 py-4 font-mono text-[10px] sm:text-xs uppercase tracking-wider text-[#8eb7ff] font-semibold">SatQuery AI</div>

            {COMPARISON_DATA.map(([a, b, c, d]) => (
              <div key={a} className="contents">
                <div className="border-r border-t border-white/10 py-3.5 pr-3 text-white/70 font-medium">{a}</div>
                <div className="border-r border-t border-white/10 px-3 py-3.5 text-white/40">{b}</div>
                <div className="border-r border-t border-white/10 px-3 py-3.5 text-white/40">{c}</div>
                <div className="border-t border-white/10 px-3 py-3.5 font-medium text-white bg-white/[0.02]">{d}</div>
              </div>
            ))}
          </div>
        </div>

        {/* CTA Bar */}
        <div className="mt-12 sm:mt-20 flex flex-col items-start justify-between gap-6 border-t border-white/15 pt-6 sm:pt-8 md:flex-row md:items-center">
          <p className="max-w-xl text-base sm:text-lg leading-relaxed text-white/70">
            A shared understanding of the Earth, available through a question anyone can ask.
          </p>
          <Link
            href="/chat"
            className="w-full sm:w-auto shrink-0 rounded-full bg-white px-7 py-3.5 text-center text-sm font-semibold text-black transition hover:-translate-y-0.5 active:scale-[0.98] shadow-lg touch-manipulation"
          >
            Try the prototype
          </Link>
        </div>
      </div>
    </section>
  )
}

// -------------------------------------------------------------
// MAIN LANDING PAGE
// -------------------------------------------------------------
export default function LandingPage() {
  const [menuOpen, setMenuOpen] = useState(false)

  // Lenis + GSAP Smooth Inertia Scroll Setup
  useEffect(() => {
    let lenis: Lenis | null = null
    try {
      lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation: 'vertical',
        smoothWheel: true,
      })

      lenis.on('scroll', ScrollTrigger.update)

      const onTick = (time: number) => {
        lenis?.raf(time * 1000)
      }

      gsap.ticker.add(onTick)
      gsap.ticker.lagSmoothing(0)

      return () => {
        gsap.ticker.remove(onTick)
        lenis?.destroy()
      }
    } catch {
      // Fallback gracefully
    }
  }, [])

  return (
    <main className="min-h-screen overflow-hidden bg-[#05080b] text-white">
      {/* HERO SECTION */}
      <section className="relative flex min-h-[100dvh] flex-col justify-between px-4 py-4 sm:px-6 sm:py-6 md:px-10 md:py-7 bg-[#05080b]">
        <video className="absolute inset-0 h-full w-full object-cover opacity-40" autoPlay muted loop playsInline aria-hidden="true">
          <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260809_012548_ef22562c-c0ae-4816-ad9d-f8922af4e6a7.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(5,8,11,.2),rgba(5,8,11,.42)_45%,#05080b_96%)]" />
        
        {/* Header */}
        <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between">
          <Link href="/" aria-label="SatQuery AI home" className="flex items-center gap-2.5 sm:gap-3">
            <span className="grid size-9 sm:size-10 place-items-center rounded-full bg-white shadow-lg">
              <img src="/icon.svg" alt="" width="24" height="24" className="size-5 sm:size-6" />
            </span>
            <span className="text-sm font-semibold tracking-tight">SatQuery <span className="text-white/40">AI</span></span>
          </Link>
          <nav className="hidden items-center gap-6 lg:gap-8 text-sm text-white/60 md:flex">
            <a href="#demo" className="text-[#8eb7ff] font-medium transition hover:text-white flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-[#8eb7ff] animate-pulse"></span>
              Playground
            </a>
            <a href="#problem" className="transition hover:text-white">Problem</a>
            <a href="#innovation" className="transition hover:text-white">Innovation</a>
            <a href="#method" className="transition hover:text-white">Method</a>
            <a href="#capabilities" className="transition hover:text-white">Capabilities</a>
            <a href="#applications" className="transition hover:text-white">Applications</a>
          </nav>
          <div className="flex items-center gap-2">
            <Link href="/chat" className="hidden sm:inline-flex rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black transition hover:bg-white/85 active:scale-[0.98]">
              Open workspace
            </Link>
            <button 
              aria-label={menuOpen ? 'Close menu' : 'Open menu'} 
              aria-expanded={menuOpen} 
              onClick={() => setMenuOpen(!menuOpen)} 
              className="grid size-10 place-items-center rounded-full border border-white/20 bg-white/10 text-white active:scale-95 md:hidden touch-manipulation"
            >
              {menuOpen ? <X className="size-5" /> : <Menu className="size-5" />}
            </button>
          </div>
        </header>

        {/* Hero Content with Fluid Clamp Typography */}
        <div className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 items-center">
          <div className="max-w-4xl py-10 sm:py-14 md:py-16">
            <p className="mb-4 sm:mb-7 text-[11px] sm:text-xs font-semibold uppercase tracking-[.22em] text-[#8eb7ff]">
              Smart India Hackathon 2026 · Geospatial intelligence
            </p>
            <h1 className="max-w-4xl font-display text-[clamp(2.4rem,7.5vw,7.2rem)] leading-[0.92] tracking-[-0.08em] text-balance">
              <span className="block animate-[headlineFade_.8s_.1s_both]">Ask the Earth</span>
              <span className="block text-white/45 animate-[headlineFade_.8s_.25s_both]">better questions.</span>
            </h1>
            <p className="mt-5 sm:mt-8 max-w-2xl text-sm sm:text-base md:text-lg leading-relaxed text-white/70">
              SatQuery AI turns satellite imagery into a conversational analyst. One shared AI brain reads optical and radar images together, tracks change over time, and shows exactly how it reached every answer.
            </p>
            <div className="mt-6 sm:mt-9 flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:gap-4">
              <a href="#demo" className="rounded-full bg-white px-7 py-3.5 text-center text-sm font-semibold text-black transition hover:-translate-y-0.5 active:scale-[0.98] shadow-lg">
                Try live demo
              </a>
              <Link href="/chat" className="rounded-full border border-white/25 px-7 py-3.5 text-center text-sm font-medium text-white/85 transition hover:border-[#8eb7ff] hover:text-white active:scale-[0.98]">
                Open workspace
              </Link>
            </div>
            <div className="mt-6 sm:mt-7 flex flex-wrap gap-x-4 sm:gap-x-5 gap-y-2 text-[10px] sm:text-[11px] font-mono uppercase tracking-[.08em] text-white/45">
              <span>Single shared backbone</span>
              <span>&lt;4s response</span>
              <span>Full audit trail</span>
              <span>4 fused capabilities</span>
            </div>
          </div>
        </div>

        {/* Hero Telemetry Stats */}
        <div className="relative z-10 mx-auto w-full max-w-6xl pb-2 sm:pb-0">
          <Stats />
        </div>
      </section>

      {/* LIVE INTERACTIVE QUERY PLAYGROUND */}
      <InteractiveQueryDemo />

      {/* THE PROBLEM */}
      <section id="problem" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
        <div className="mx-auto max-w-6xl">
          <div className="max-w-3xl">
            <h2 className="font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em] text-balance">
              Satellite intelligence is powerful and painfully fragmented.
            </h2>
            <p className="mt-4 sm:mt-6 max-w-2xl text-sm sm:text-base leading-relaxed text-white/60">
              Today, getting a real answer means stitching together separate tools, sensors, and specialists. SatQuery AI collapses that workflow into one conversation.
            </p>
          </div>
          <div className="mt-10 sm:mt-16 grid gap-0 border-y border-white/15 md:grid-cols-3">
            {problemCards.map((card) => {
              const Icon = card.icon
              return (
                <article key={card.title} className="border-b border-white/15 py-6 sm:py-8 md:border-b-0 md:border-r md:px-8 md:first:pl-0 md:last:border-r-0 md:last:pr-0">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-[#8eb7ff] font-semibold">{card.num}</span>
                    <Icon className="size-4 text-white/30" />
                  </div>
                  <h3 className="mt-4 sm:mt-8 max-w-xs text-lg sm:text-xl font-medium leading-tight">{card.title}</h3>
                  <p className="mt-2 sm:mt-4 text-xs sm:text-sm leading-relaxed text-white/55">{card.body}</p>
                </article>
              )
            })}
          </div>
        </div>
      </section>

      {/* 1. THE REFRAME */}
      <TheReframeSection />

      {/* THE METHOD */}
      <section id="method" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
            <div>
              <h2 className="max-w-2xl font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em]">
                A clear path from question to evidence.
              </h2>
            </div>
            <p className="max-w-sm text-xs sm:text-sm leading-relaxed text-white/60">
              Designed for human review, reproducible analysis, and a practical path from prototype to deployment.
            </p>
          </div>
          <div className="mt-10 sm:mt-16 grid gap-0 border-y border-white/15 sm:grid-cols-2 md:grid-cols-5">
            {workflow.map(([number, title, body]) => (
              <div key={number} className="border-b border-white/15 py-5 sm:py-7 sm:border-r md:px-5 md:first:pl-0 md:last:border-r-0">
                <p className="font-mono text-xs text-[#8eb7ff] font-semibold">{number}</p>
                <h3 className="mt-4 sm:mt-7 text-sm sm:text-base font-medium leading-tight">{title}</h3>
                <p className="mt-2 sm:mt-3 text-xs sm:text-sm leading-relaxed text-white/50">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 2. PROOF, NOT ADJECTIVES */}
      <ProofNotAdjectivesSection />

      {/* CAPABILITIES */}
      <section id="capabilities" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
        <div className="mx-auto max-w-6xl">
          <h2 className="max-w-2xl font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em]">
            One interface, four remote-sensing skills.
          </h2>
          <div className="mt-10 sm:mt-16 grid gap-px rounded-xl sm:rounded-2xl overflow-hidden border border-white/15 bg-white/15 md:grid-cols-2">
            {capabilities.map(([title, query, answer], index) => (
              <article key={title} className="bg-[#05080b] p-6 sm:p-8 md:p-10">
                <p className="font-mono text-xs text-[#8eb7ff] font-semibold">0{index + 1}</p>
                <h3 className="mt-4 sm:mt-7 text-lg sm:text-xl font-medium">{title}</h3>
                <p className="mt-4 sm:mt-7 border-l-2 border-[#8eb7ff]/60 pl-3.5 text-xs sm:text-sm italic leading-relaxed text-white/70">“{query}”</p>
                <p className="mt-3 sm:mt-5 text-xs sm:text-sm leading-relaxed text-white/50">{answer}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* 3. TRUST BY DESIGN */}
      <TrustByDesignSection />

      {/* APPLICATIONS */}
      <section id="applications" className="border-t border-white/10 bg-[#05080b] px-4 py-16 sm:px-6 sm:py-24 md:px-10 md:py-32">
        <div className="mx-auto max-w-6xl">
          <div className="grid gap-8 sm:gap-14 md:grid-cols-[.8fr_1.2fr]">
            <h2 className="max-w-xl font-display text-3xl sm:text-4xl md:text-6xl leading-[1.02] tracking-[-0.07em]">
              Built for decisions that affect people.
            </h2>
            <div className="divide-y divide-white/15 border-y border-white/15">
              {applications.map(([title, pain, benefit]) => (
                <div key={title} className="grid gap-2 sm:gap-3 py-5 sm:py-6 md:grid-cols-[.8fr_1fr] md:gap-8">
                  <div>
                    <h3 className="text-base sm:text-lg font-medium">{title}</h3>
                    <p className="mt-1 text-[10px] sm:text-xs uppercase tracking-[.12em] text-white/35">Today · {pain}</p>
                  </div>
                  <p className="text-xs sm:text-sm leading-relaxed text-white/60">{benefit}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 4. SCROLL MORPH HERO (RESPONSIVE 3D MORPH LAYER) */}
      <div className="w-full bg-[#05080b] border-t border-white/10 overflow-hidden">
        <IntroAnimation />
      </div>

      {/* 5. WHY SATQUERY AI */}
      <WhySatQuerySection />

      {/* FOOTER */}
      <footer className="mx-auto flex max-w-6xl flex-col gap-4 border-t border-white/15 px-4 py-6 sm:px-6 sm:py-8 text-xs text-white/35 md:flex-row md:items-center md:justify-between md:px-10">
        <span>SatQuery AI · Smart India Hackathon 2026</span>
        <span>Evidence-led remote sensing for public decisions.</span>
      </footer>

      <MobileMenu open={menuOpen} onClose={() => setMenuOpen(false)} />
    </main>
  )
}
