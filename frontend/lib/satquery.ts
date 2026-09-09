/**
 * SatQuery AI API Client and Type Definitions
 * Adheres strictly to docs/UI-CONTRACT.md and satquery/schemas/trace.schema.json
 */

export interface SatQueryHealth {
  ok: boolean
  backend: "stub" | "real" | string
  model_available: boolean
  dtype?: string
  adapters?: {
    vqa?: string
    ground?: string
    change?: string
    fusion?: string
  }
}

export interface TraceInputImage {
  path: string
  format: string
  modality: "optical" | "sar" | "msi" | string
  bands?: number
  size_px?: [number, number]
  gsd_m?: number | null
  crs?: string | null
  acquired_at?: string | null
  sha256?: string
  bbox?: number[]
}

export interface TraceInputCheck {
  verdict: "accepted" | "rejected"
  code: string | null
  message: string | null
  warnings: string[]
  n_images: number
  modality: string[]
  gsd_m: (number | null)[]
  acquisitions: (string | null)[]
  crs: (string | null)[]
  footprint_check?: string
  temporal_order?: string | null
  images: TraceInputImage[]
}

export interface TraceRouting {
  rule_id: "R0" | "R1" | "R2" | "R3" | "R4" | "R5" | "R6" | "S1" | string
  by: "rule" | "planner" | "none" | string
  planner_used?: boolean
  fanout?: any
  pair_declaration?: string | null
}

export interface TraceStep {
  tool: string
  stub: boolean
  duration_ms: number
  images: number[]
  params?: Record<string, any>
  cache?: string | null
  confidence?: number | null
}

export interface TraceQuantities {
  norm_box?: [number, number, number, number]
  box_frame?: string
  adapter?: string
  dtype?: string
  adapter_rule?: string
  ref_target?: string
  resized_wh?: [number, number]
  export?: {
    path?: string
  }
  [key: string]: any
}

export interface TraceOutput {
  text: string
  confidence: number | null
  quantities?: TraceQuantities
  geojson?: string | null
  raster?: string | null
}

export interface TraceReplay {
  seed: number
  input_sha256: string[]
  registry_lock: string
}

export interface RenderedImage {
  name: string
  url: string
  type: string
}

export interface Trace {
  trace_id: string
  query: string
  classified_task: "vqa" | "grounding" | "captioning" | "change_vqa" | "fusion" | null
  abstained: boolean
  input_check: TraceInputCheck
  routing: TraceRouting
  steps: TraceStep[]
  output: TraceOutput
  replay: TraceReplay
  rendered_images?: RenderedImage[]
  geotarget?: Record<string, any>
  bhoonidhi_scenes?: any[]
}

export interface FilePayload {
  name: string
  b64: string
  previewUrl?: string
}

export const CANONICAL_DEMO_QUERIES = [
  {
    id: 1,
    title: "1. Road Detection (VQA)",
    query: "Is there a road?",
    rule: "R6",
    task: "vqa",
    files: ["lr_232.tif"],
    sidecars: [],
    expected: "yes"
  },
  {
    id: 2,
    title: "2. Building Count (VQA)",
    query: "How many buildings are there?",
    rule: "R6",
    task: "vqa",
    files: ["lr_232.tif"],
    sidecars: [],
    expected: "194"
  },
  {
    id: 3,
    title: "3. Settlement Type (VQA)",
    query: "Is it a rural or an urban area?",
    rule: "R6",
    task: "vqa",
    files: ["lr_232.tif"],
    sidecars: [],
    expected: "rural"
  },
  {
    id: 4,
    title: "4. Water Presence (VQA)",
    query: "Is there any water in the image?",
    rule: "R6",
    task: "vqa",
    files: ["opt.tif"],
    sidecars: ["opt.tif.meta.json"],
    expected: "yes"
  },
  {
    id: 5,
    title: "5. Pasture Localization (Grounding)",
    query: "Where is the largest connected region of pastures?",
    rule: "R4",
    task: "grounding",
    files: ["opt.tif"],
    sidecars: ["opt.tif.meta.json"],
    expected: "[0.53 0.0, 1.0 0.7]"
  },
  {
    id: 6,
    title: "6. Building Area Growth (Change VQA)",
    query: "Did the areas of buildings increase?",
    rule: "R3",
    task: "change_vqa",
    files: ["pre.png", "post.png"],
    sidecars: ["pre.png.meta.json", "post.png.meta.json"],
    expected: "no"
  },
  {
    id: 7,
    title: "7. Largest Change Class (Change VQA)",
    query: "What is the largest change?",
    rule: "R3",
    task: "change_vqa",
    files: ["pre.png", "post.png"],
    sidecars: ["pre.png.meta.json", "post.png.meta.json"],
    expected: "low_vegetation"
  },
  {
    id: 8,
    title: "8. Optical + SAR Water Detection (Fusion)",
    query: "Is there any water in the image?",
    rule: "R1",
    task: "fusion",
    files: ["opt.tif", "sar.tif"],
    sidecars: ["opt.tif.meta.json", "sar.tif.meta.json"],
    expected: "yes"
  },
  {
    id: 9,
    title: "9. Land Cover Classification (Fusion)",
    query: "Which class covers most of the image? a) Arable land, b) Pastures, c) Broad-leaved forest, d) Water",
    rule: "R1",
    task: "fusion",
    files: ["opt.tif", "sar.tif"],
    sidecars: ["opt.tif.meta.json", "sar.tif.meta.json"],
    expected: "d"
  },
  {
    id: 10,
    title: "10. Optical + SAR Built-Up & Change (Compound)",
    query: "Use the optical and SAR images together to identify built-up and water-covered regions, and tell me if built-up area increased since last year",
    rule: "COMPOUND",
    task: "rs_fusion_change",
    files: ["opt.tif", "sar.tif", "pre.png"],
    sidecars: ["opt.tif.meta.json", "sar.tif.meta.json", "pre.png.meta.json"],
    expected: "Built-up & water identified; change quantified."
  }
]

export function getSatQueryApiUrl(): string {
  if (typeof window !== "undefined") {
    const custom = localStorage.getItem("satquery_api_url")
    if (custom && !custom.includes("ts.net")) return custom
    // If old remote gateway was saved in localStorage, remove it so user talks to local controller
    if (custom && custom.includes("ts.net")) {
      localStorage.removeItem("satquery_api_url")
      localStorage.removeItem("satquery_token")
    }
    return process.env.NEXT_PUBLIC_SATQUERY_API_URL || "http://localhost:8765"
  }
  return process.env.SATQUERY_API_URL || process.env.NEXT_PUBLIC_SATQUERY_API_URL || "http://localhost:8765"
}

export function getSatQueryToken(): string {
  if (typeof window !== "undefined") {
    const custom = localStorage.getItem("satquery_token")
    if (custom !== null && custom !== "") return custom
  }
  return process.env.NEXT_PUBLIC_SATQUERY_TOKEN || ""
}

export function setSatQueryEndpoint(url: string, token: string = ""): void {
  if (typeof window !== "undefined") {
    if (url) localStorage.setItem("satquery_api_url", url)
    else localStorage.removeItem("satquery_api_url")
    if (token) localStorage.setItem("satquery_token", token)
    else localStorage.removeItem("satquery_token")
  }
}

/**
 * Checks backend health status
 */
export async function checkBackendHealth(): Promise<SatQueryHealth> {
  const base = getSatQueryApiUrl().replace(/\/$/, "")
  const res = await fetch(`${base}/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  })
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`)
  }
  return await res.json()
}

export interface BhoonidhiArchive {
  satellite: string
  access: string
  min_res_m: string
  max_res_m: string
  start_date: string
  end_date: string
  sensors: {
    name: string
    display: string
    resolution_m: string
    product: string
    start_date: string
    end_date: string
  }[]
}

export interface BhoonidhiScene {
  id: string
  satellite: string
  sensor: string
  product_type: string
  dop: string
  orbit: string
  tile_id: string
  priced: string
  coverage_pct: string
  corners: {
    nw: [number, number]
    ne: [number, number]
    se: [number, number]
    sw: [number, number]
  }
  download_ready: boolean
}

export interface BhoonidhiSearchResult {
  satellite: string
  sensor?: string
  bbox: [number, number, number, number]
  start_date: string
  end_date: string
  total_found: number
  returned: number
  scenes: BhoonidhiScene[]
  error?: string
}

/**
 * Fetch available satellite missions from ISRO Bhoonidhi
 */
export async function fetchBhoonidhiArchives(filter?: string): Promise<BhoonidhiArchive[]> {
  const base = getSatQueryApiUrl().replace(/\/$/, "")
  const url = filter ? `${base}/bhoonidhi/archives?q=${encodeURIComponent(filter)}` : `${base}/bhoonidhi/archives`
  const res = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  })
  if (!res.ok) {
    throw new Error(`Failed to fetch Bhoonidhi archives: HTTP ${res.status}`)
  }
  const data = await res.json()
  return data.archives || []
}

/**
 * Search ISRO Bhoonidhi catalog for real satellite scenes by bounding box and dates
 */
export async function searchBhoonidhi(params: {
  start_date: string
  end_date: string
  satellite: string
  sensor?: string
  bbox?: [number, number, number, number]
  limit?: number
}): Promise<BhoonidhiSearchResult> {
  const base = getSatQueryApiUrl().replace(/\/$/, "")
  const res = await fetch(`${base}/bhoonidhi/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || `Bhoonidhi search failed with HTTP ${res.status}`)
  }
  return await res.json()
}

/**
 * Convert a browser File object to Base64
 */
export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const res = reader.result as string
      // remove data:*/*;base64, prefix
      const commaIdx = res.indexOf(",")
      resolve(commaIdx >= 0 ? res.slice(commaIdx + 1) : res)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * Loads a demo fixture file from /fixtures/ in public directory and returns base64
 */
export async function loadFixtureAsBase64(filename: string): Promise<string> {
  const res = await fetch(`/fixtures/${filename}`)
  if (!res.ok) {
    throw new Error(`Could not load fixture ${filename}: ${res.statusText}`)
  }
  const blob = await res.blob()
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const str = reader.result as string
      const commaIdx = str.indexOf(",")
      resolve(commaIdx >= 0 ? str.slice(commaIdx + 1) : str)
    }
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

/**
 * Sends an analysis question and satellite imagery to SatQuery API
 */
export async function sendSatQuery(params: {
  query: string
  files?: { name: string; b64: string }[]
  seed?: number
  agentic?: boolean
  history?: any[]
}): Promise<Trace> {
  const base = getSatQueryApiUrl().replace(/\/$/, "")
  const token = getSatQueryToken()

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  }
  if (token) {
    headers["X-SatQuery-Token"] = token
  }

  const endpoint = params.agentic === false ? `${base}/answer` : `${base}/agent`

  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({
      query: params.query,
      files: params.files || [],
      seed: params.seed ?? 1337,
      agentic: true,
      history: params.history || [],
    }),
  })

  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.error || `Request failed with status ${res.status}`)
  }

  return data as Trace
}
