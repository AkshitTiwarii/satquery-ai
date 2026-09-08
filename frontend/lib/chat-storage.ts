/**
 * SatQuery AI - Chat Storage & Titling Service
 * 
 * Provides:
 * 1. Automatic semantic chat titling (like ChatGPT / Claude) from natural language prompts & multi-agent traces.
 * 2. Client cache persistence (localStorage) for non-logged-in guest users.
 * 3. Future-proof interface for permanent logged-in user database persistence.
 */

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  files?: Array<{ name: string; previewUrl?: string }>
  apiFiles?: Array<{ name: string; b64?: string; previewUrl?: string }>
  trace?: any
  previewUrls?: Record<string, string>
  createdAt?: string
  editedAt?: string
  isLive?: boolean
}

export interface Conversation {
  id: string
  title: string
  updatedAt: string
  messageCount: number
  preview: string
  pinned: boolean
  folder: string | null
  messages: ChatMessage[]
}

const GUEST_STORAGE_KEY = "satquery_guest_conversations_v1"
const ACTIVE_ID_KEY = "satquery_active_conv_id_v1"

/**
 * Intelligent chat title generator matching modern AI systems (ChatGPT, Claude).
 * Extracts core geographical regions, remote sensing tasks, temporal comparison years,
 * and eliminates conversational boilerplate.
 */
export function generateChatTitle(query: string = "", trace: any = null): string {
  if (!query && !trace) return "New Analysis"

  const qTrim = query.trim()
  const qLower = qTrim.toLowerCase()

  // 1. Check for explicit geolocation in query or trace
  const knownLocations = [
    "Delhi", "Mumbai", "Bengaluru", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Sundarbans", "Kerala", "Assam", "Gujarat", "Himalayas", "Ladakh", "Uttarakhand",
    "Thar Desert", "Western Ghats", "Brahmaputra", "Yamuna", "Ganga", "Pune", "Jaipur",
    "Ahmedabad", "Bhopal", "Varanasi", "Srinagar", "Goa", "Andaman", "Lakshadweep"
  ]
  const matchedLoc =
    knownLocations.find((loc) => qLower.includes(loc.toLowerCase())) ||
    trace?.geotarget?.city ||
    trace?.location ||
    ""

  // 2. Check for multi-year temporal comparisons (e.g. 1996 vs 2026)
  const yearMatches = qTrim.match(/\b(19\d\d|20\d\d)\b/g)
  const uniqueYears = yearMatches ? Array.from(new Set(yearMatches)) : []

  // 3. Determine primary remote sensing domain / topic
  let topic = ""
  if (qLower.includes("vegetation") || qLower.includes("ndvi") || qLower.includes("greenery") || qLower.includes("forest") || qLower.includes("tree")) {
    topic = "Vegetation Analysis"
  } else if (qLower.includes("flood") || qLower.includes("water") || qLower.includes("inundation") || qLower.includes("lake") || qLower.includes("river")) {
    topic = "Flood Inundation"
  } else if (qLower.includes("pasture") || qLower.includes("meadow") || qLower.includes("grassland") || qLower.includes("agriculture") || qLower.includes("crop")) {
    topic = "Pasture Localization"
  } else if (qLower.includes("urban") || qLower.includes("construction") || qLower.includes("building") || qLower.includes("sprawl") || qLower.includes("road")) {
    topic = "Urban Expansion"
  } else if (qLower.includes("sar") && (qLower.includes("optical") || qLower.includes("fusion") || qLower.includes("crossmodal"))) {
    topic = "SAR-Optical Fusion"
  } else if (trace?.classified_task === "multitemporal_change_detection") {
    topic = "Temporal Change"
  } else if (trace?.classified_task === "grounding_detection") {
    topic = "Feature Grounding"
  } else if (trace?.classified_task === "crossmodal_fusion") {
    topic = "Cross-Modal SAR"
  }

  // 4. Synthesize high-fidelity semantic title
  if (matchedLoc && topic && uniqueYears.length >= 2) {
    return `${matchedLoc} ${topic} (${uniqueYears.join(" vs ")})`
  }
  if (matchedLoc && topic) {
    return `${matchedLoc}: ${topic}`
  }
  if (matchedLoc && uniqueYears.length >= 2) {
    return `${matchedLoc} (${uniqueYears.join(" vs ")})`
  }
  if (matchedLoc) {
    return `${matchedLoc} Satellite Query`
  }
  if (topic) {
    return topic
  }

  // 5. Fallback: Clean conversational prompt boilerplate
  let cleaned = qTrim
    .replace(
      /^(can you please tell me|can you please|could you please|please tell me|what is the difference in|what is the difference between|what is the|tell me about|analyze the|show me|find the|check the|is there any|where is the|localize the|can you)\s+/i,
      "",
    )
    .replace(/[?.!]+$/, "")
    .trim()

  if (!cleaned) return "Satellite Query"

  // Capitalize first character and truncate gracefully
  cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
  if (cleaned.length > 36) {
    cleaned = cleaned.slice(0, 34).trim() + "…"
  }
  return cleaned
}

/**
 * Save chat conversations to cache / client storage.
 * Strips raw heavy base64 strings to stay within browser localStorage limits (typically 5MB),
 * while preserving preview URLs and complete message & trace structures.
 */
export function saveChatConversations(conversations: Conversation[], user: any = null): boolean {
  if (typeof window === "undefined") return false

  // For logged-in users, sync asynchronously to Neon PostgreSQL
  if (user) {
    syncConversationsToNeon(conversations).catch((err) => {
      console.warn("[SatQuery Storage] Error syncing to Neon DB:", err)
    })
  }

  // Guest / local cache persistence
  try {
    const sanitized = conversations.map((conv) => ({
      ...conv,
      messages: (conv.messages || []).map((msg) => ({
        ...msg,
        isLive: false, // Explicitly mark completed so reloads never re-stream
        // Keep previewUrls, traces, and text; strip large raw base64 arrays
        apiFiles: (msg.apiFiles || []).map((file) => ({
          name: file.name,
          previewUrl: file.previewUrl || "",
          // Only preserve inline b64 if it's very small (< 100KB)
          b64: file.b64 && file.b64.length < 100000 ? file.b64 : undefined,
        })),
      })),
    }))

    localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(sanitized))
    return true
  } catch (err) {
    console.warn("[SatQuery Storage] Failed to persist conversations to localStorage:", err)
    return false
  }
}

/**
 * Load chat conversations from client cache or fall back to default conversations.
 */
export function loadChatConversations(fallback: Conversation[] = []): Conversation[] {
  if (typeof window === "undefined") return fallback

  try {
    const raw = localStorage.getItem(GUEST_STORAGE_KEY)
    if (!raw) return fallback

    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed) && parsed.length > 0) {
      // Ensure all persisted messages have isLive=false so they never re-animate on page reload
      return parsed.map((c: any) => ({
        ...c,
        pinned: c.id === "demo-welcome" && c.pinned ? false : c.pinned,
        messages: (c.messages || []).map((m: any) => ({
          ...m,
          isLive: false,
        })),
      }))
    }
  } catch (err) {
    console.warn("[SatQuery Storage] Failed to parse cached conversations:", err)
  }
  return fallback
}

/**
 * Save the user's active conversation ID
 */
export function saveActiveChatId(id: string): void {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(ACTIVE_ID_KEY, id)
  } catch {}
}

/**
 * Load the user's active conversation ID
 */
export function loadActiveChatId(): string | null {
  if (typeof window === "undefined") return null
  try {
    return localStorage.getItem(ACTIVE_ID_KEY)
  } catch {
    return null
  }
}

/**
 * Clear all cached chat conversations
 */
export function clearCachedConversations(): void {
  if (typeof window === "undefined") return
  try {
    localStorage.removeItem(GUEST_STORAGE_KEY)
    localStorage.removeItem(ACTIVE_ID_KEY)
  } catch {}
}

/**
 * Persist user conversations directly to Neon PostgreSQL via API
 */
export async function syncConversationsToNeon(conversations: Conversation[]): Promise<boolean> {
  try {
    const sanitized = conversations.map((conv) => ({
      ...conv,
      messages: (conv.messages || []).map((msg) => ({
        ...msg,
        isLive: false,
      })),
    }))
    const res = await fetch("/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversations: sanitized }),
    })
    return res.ok
  } catch (err) {
    console.error("[SatQuery Sync Error]:", err)
    return false
  }
}

/**
 * Fetch permanent user conversations from Neon PostgreSQL
 */
export async function fetchConversationsFromNeon(): Promise<Conversation[] | null> {
  try {
    const res = await fetch("/api/chats")
    if (!res.ok) return null
    const data = await res.json()
    if (data.success && Array.isArray(data.conversations)) {
      return data.conversations.map((c: any) => ({
        ...c,
        messages: (c.messages || []).map((m: any) => ({
          ...m,
          isLive: false,
        })),
      }))
    }
  } catch (err) {
    console.error("[SatQuery Fetch Error]:", err)
  }
  return null
}

