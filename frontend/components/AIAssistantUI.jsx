"use client"

import React, { useEffect, useMemo, useRef, useState, useCallback } from "react"
import { Calendar, LayoutGrid, MoreHorizontal, PanelLeftOpen } from "lucide-react"
import Sidebar from "./Sidebar"
import Header from "./Header"
import ChatPane from "./ChatPane"
import GhostIconButton from "./GhostIconButton"
import ThemeToggle from "./ThemeToggle"
import { INITIAL_TEMPLATES, INITIAL_FOLDERS } from "./mockData"
import { checkBackendHealth, sendSatQuery } from "@/lib/satquery"
import {
  generateChatTitle,
  saveChatConversations,
  loadChatConversations,
  saveActiveChatId,
  loadActiveChatId,
  clearCachedConversations,
  fetchConversationsFromNeon,
} from "@/lib/chat-storage"
import AuthModal from "./AuthModal"
import AvatarPickerModal from "./AvatarPickerModal"

const INITIAL_SATQUERY_CONVERSATIONS = [
  {
    id: "demo-welcome",
    title: "SatQuery Multimodal Remote Sensing",
    updatedAt: "2026-09-08T15:28:49.000Z",
    messageCount: 1,
    preview: "SatQuery AI vision-language assistant initialized...",
    pinned: false,
    folder: null,
    messages: [
      {
        id: "msg-welcome",
        role: "assistant",
        content:
          "🛰️ **SatQuery AI Multi-Agent System (ISRO SIH26167)** is ready.\n\nYou don't need to configure coordinates, dates, or satellites manually. Simply type any prompt in plain text — for example:\n> *\"What is the difference in vegetation in Delhi in 2026 vs 1996?\"*\n\nThe multi-agent system will autonomously:\n1. Geocode and parse target coordinates and temporal periods\n2. Query ISRO's Bhoonidhi STAC catalog for actual satellite passes\n3. Ingest and render satellite imagery directly in chat\n4. Execute deterministic VLM bi-temporal change detection & spatial grounding\n5. Deliver authoritative geospatial analysis powered by domain-trained vision-language models.",
        createdAt: "2026-09-08T15:28:49.000Z",
        isLive: false,
        trace: {
          trace_id: "conv-welcome-init",
          classified_task: "conversation",
          abstained: false,
          input_check: {
            verdict: "accepted",
            message: "Autonomous Earth Observation Multi-Agent System Initialized",
            modality: ["Optical MSI (10m)", "C-Band SAR (10m)"],
            gsd_m: [10.0],
            crs: ["EPSG:4326 (WGS84)"],
          },
          routing: {
            by: "domain_planner",
            rule_id: "RS_AUTONOMOUS_ORCHESTRATOR",
            planner_used: true,
          },
          steps: [],
          thought_process:
            "**Initializing Autonomous Multi-Agent Remote Sensing Pipeline**\nUser initiated session with SatQuery AI assistant. The system is designed to provide authoritative, evidence-led remote sensing intelligence across India and global territories without requiring manual coordinate inputs.\n\n1. Autonomous Intent & Geocoding: The system dynamically parses natural language place names, landmarks, and dates, resolving them to geodetic WGS84 bounding boxes via Nominatim & Bhoonidhi.\n2. Live Data Ingestion: Active STAC queries connect directly to ISRO's Bhoonidhi data access portal, retrieving real satellite acquisitions with zero synthetic placeholders.\n3. Domain VLM Grounding: Ingested GeoTIFF rasters undergo deterministic spectral decomposition, cross-sensor SAR fusion, and bi-temporal change detection.\n\nSystem status: All cognitive agents, spatial grounding modules, and Bhoonidhi connectors are online and ready for incoming queries.",
        },
      },
    ],
  },
]

export default function AIAssistantUI() {
  const [mounted, setMounted] = useState(false)
  // hydrated: true only after initial localStorage load is complete so we never
  // accidentally overwrite real stored conversations with the placeholder initial state.
  const hydrated = useRef(false)
  // Deterministic server/client default to eliminate SSR hydration mismatch
  const [theme, setTheme] = useState("dark")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [collapsed, setCollapsed] = useState({ pinned: true, recent: false, folders: true, templates: true })

  const [conversations, setConversations] = useState(INITIAL_SATQUERY_CONVERSATIONS)
  const [selectedId, setSelectedId] = useState("demo-welcome")
  const [templates, setTemplates] = useState(INITIAL_TEMPLATES)
  const [folders, setFolders] = useState(INITIAL_FOLDERS)

  const [query, setQuery] = useState("")
  const searchRef = useRef(null)

  const [isThinking, setIsThinking] = useState(false)
  const [thinkingConvId, setThinkingConvId] = useState(null)
  const [backendHealth, setBackendHealth] = useState(null)
  const [currentUser, setCurrentUser] = useState(null)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [showAvatarPickerModal, setShowAvatarPickerModal] = useState(false)
  const composerRef = useRef(null)

  // 1. Client hydration: safely restore theme, layout preferences, and cached chat memory
  useEffect(() => {
    setMounted(true)
    try {
      // Hydrate Theme
      const savedTheme = localStorage.getItem("theme")
      if (savedTheme === "light" || savedTheme === "dark") {
        setTheme(savedTheme)
      } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
        setTheme("light")
      }

      // Hydrate Sidebar Collapsed State
      const savedSidebarCollapsed = localStorage.getItem("sidebar-collapsed-state")
      if (savedSidebarCollapsed) {
        setSidebarCollapsed(JSON.parse(savedSidebarCollapsed))
      }

      // Hydrate Collapsed Sections
      const savedSections = localStorage.getItem("sidebar-collapsed")
      if (savedSections) {
        setCollapsed(JSON.parse(savedSections))
      }

      // Hydrate Chat Memory Cache (Non-logged-in guest persistence)
      const cached = loadChatConversations(INITIAL_SATQUERY_CONVERSATIONS)
      if (cached && cached.length > 0) {
        setConversations(cached)
        const activeId = loadActiveChatId()
        if (activeId && cached.some((c) => c.id === activeId)) {
          setSelectedId(activeId)
        } else {
          setSelectedId(cached[0].id)
        }
      }
    } catch (err) {
      console.warn("Error hydrating state from client storage:", err)
    }

    // Check Neon DB Authentication Session
    fetch("/api/auth/me")
      .then((res) => res.json())
      .then((data) => {
        if (data.authenticated && data.user) {
          setCurrentUser(data.user)
          // Load permanent user conversations from Neon PostgreSQL
          fetchConversationsFromNeon().then((neonChats) => {
            if (neonChats && neonChats.length > 0) {
              setConversations(neonChats)
              const activeId = loadActiveChatId()
              if (activeId && neonChats.some((c) => c.id === activeId)) {
                setSelectedId(activeId)
              } else {
                setSelectedId(neonChats[0].id)
              }
            }
          })
        }
      })
      .catch((err) => {
        console.debug("Guest session active:", err)
      })
  }, [])

  // 2. Sync Theme to Document Attributes
  useEffect(() => {
    if (!mounted) return
    try {
      if (theme === "dark") {
        document.documentElement.classList.add("dark")
      } else {
        document.documentElement.classList.remove("dark")
      }
      document.documentElement.setAttribute("data-theme", theme)
      document.documentElement.style.colorScheme = theme
      localStorage.setItem("theme", theme)
    } catch {}
  }, [theme, mounted])

  // 3. Persist Sidebar State
  useEffect(() => {
    if (!mounted) return
    try {
      localStorage.setItem("sidebar-collapsed", JSON.stringify(collapsed))
    } catch {}
  }, [collapsed, mounted])

  useEffect(() => {
    if (!mounted) return
    try {
      localStorage.setItem("sidebar-collapsed-state", JSON.stringify(sidebarCollapsed))
    } catch {}
  }, [sidebarCollapsed, mounted])

  // 4. Persist Chat Memory (Guest Cache & Neon DB for Logged-In Users)
  // Guard: only persist after hydration is complete so we never overwrite
  // real stored conversations with the placeholder initial state on page load.
  useEffect(() => {
    if (!mounted) return
    if (!hydrated.current) {
      // Mark hydrated after first mount cycle — next change will be genuine user data
      hydrated.current = true
      return
    }
    saveChatConversations(conversations, currentUser)
  }, [conversations, mounted, currentUser])

  const handleAuthSuccess = async (user) => {
    setCurrentUser(user)
    try {
      const neonChats = await fetchConversationsFromNeon()
      if (neonChats && neonChats.length > 0) {
        setConversations(neonChats)
        setSelectedId(neonChats[0].id)
      } else if (conversations.length > 0) {
        // Sync existing conversations to user's new Neon account
        saveChatConversations(conversations, user)
      }
    } catch (err) {
      console.warn("Error syncing user data on auth success:", err)
    }
  }

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" })
    } catch {}
    setCurrentUser(null)
    const cached = loadChatConversations(INITIAL_SATQUERY_CONVERSATIONS)
    setConversations(cached)
    if (cached.length > 0) setSelectedId(cached[0].id)
  }

  useEffect(() => {
    if (!mounted || !selectedId) return
    saveActiveChatId(selectedId)
  }, [selectedId, mounted])

  // 5. Backend Health Polling
  const refreshHealth = useCallback(async () => {
    try {
      const health = await checkBackendHealth()
      setBackendHealth(health)
    } catch (err) {
      console.warn("SatQuery backend health check failed:", err)
      setBackendHealth({ ok: false, backend: "offline", model_available: false })
    }
  }, [])

  useEffect(() => {
    refreshHealth()
    const interval = setInterval(refreshHealth, 15000)
    return () => clearInterval(interval)
  }, [refreshHealth])

  // 6. Keyboard Shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
        e.preventDefault()
        createNewChat()
      }
      if (!e.metaKey && !e.ctrlKey && e.key === "/") {
        const tag = document.activeElement?.tagName?.toLowerCase()
        if (tag !== "input" && tag !== "textarea") {
          e.preventDefault()
          searchRef.current?.focus()
        }
      }
      if (e.key === "Escape" && sidebarOpen) setSidebarOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [sidebarOpen, conversations])

  // 7. Filtering & Grouping
  const filtered = useMemo(() => {
    if (!query.trim()) return conversations
    const q = query.toLowerCase()
    return conversations.filter((c) => c.title.toLowerCase().includes(q) || c.preview.toLowerCase().includes(q))
  }, [conversations, query])

  const pinned = filtered.filter((c) => c.pinned).sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))

  const recent = filtered
    .filter((c) => !c.pinned)
    .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))
    .slice(0, 20)

  const folderCounts = useMemo(() => {
    const map = Object.fromEntries(folders.map((f) => [f.name, 0]))
    for (const c of conversations) if (map[c.folder] != null) map[c.folder] += 1
    return map
  }, [conversations, folders])

  function togglePin(id) {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)))
  }

  function createNewChat() {
    const id = "conv-" + Math.random().toString(36).slice(2, 9)
    const item = {
      id,
      title: "New Analysis",
      updatedAt: new Date().toISOString(),
      messageCount: 0,
      preview: "Attach satellite imagery or type any geographic query...",
      pinned: false,
      folder: "ISRO SIH26167",
      messages: [],
    }
    setConversations((prev) => [item, ...prev])
    setSelectedId(id)
    setSidebarOpen(false)
  }

  function deleteConversation(id) {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id)
      if (selectedId === id) {
        if (next.length > 0) {
          setSelectedId(next[0].id)
        } else {
          // If no conversations left, create fresh
          const newId = "conv-" + Math.random().toString(36).slice(2, 9)
          const fresh = {
            id: newId,
            title: "New Analysis",
            updatedAt: new Date().toISOString(),
            messageCount: 0,
            preview: "Attach satellite imagery to begin...",
            pinned: false,
            folder: "ISRO SIH26167",
            messages: [],
          }
          setSelectedId(newId)
          return [fresh]
        }
      }
      return next
    })
  }

  function renameConversation(id, newTitle) {
    if (!newTitle?.trim()) return
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle.trim() } : c))
    )
  }

  function clearAllConversations() {
    if (confirm("Are you sure you want to clear all chat history? This will reset your local cache.")) {
      clearCachedConversations()
      setConversations(INITIAL_SATQUERY_CONVERSATIONS)
      setSelectedId("demo-welcome")
    }
  }

  function createFolder(folderName) {
    const name = folderName || prompt("Folder name")
    if (!name) return
    if (folders.some((f) => f.name.toLowerCase() === name.toLowerCase())) {
      return alert("Folder already exists.")
    }
    setFolders((prev) => [...prev, { id: Math.random().toString(36).slice(2), name }])
  }

  async function sendMessage(convId, payload) {
    const text = typeof payload === "string" ? payload : payload.query
    const attachedFiles = typeof payload === "string" ? [] : payload.files || []

    if (!text?.trim()) return
    const now = new Date().toISOString()

    const previewUrls = {}
    attachedFiles.forEach((f) => {
      if (f.previewUrl) previewUrls[f.name] = f.previewUrl
    })

    let apiFiles = attachedFiles.map((f) => ({
      name: f.name,
      b64: f.b64,
      previewUrl: f.previewUrl,
    }))

    // Only inherit files from immediate previous turn if the user explicitly refers to "in this image/scene"
    // and does NOT ask for a new geographic region or catalog search
    const isExplicitImageReferral =
      /\b(in this image|in that image|in the image|in this scene|in the scene|in this picture|in it)\b/i.test(text) &&
      !/\b(imagery of|images of|scenes of|satellite data|search|show me imagery|passes for|over|near|around)\b/i.test(text)

    if (apiFiles.length === 0 && isExplicitImageReferral) {
      const conv = conversations.find((c) => c.id === convId)
      const prevMsgWithFiles = (conv?.messages || [])
        .slice()
        .reverse()
        .find((m) => m.apiFiles && m.apiFiles.length > 0)

      if (prevMsgWithFiles && prevMsgWithFiles.apiFiles?.length > 0) {
        apiFiles = prevMsgWithFiles.apiFiles
        apiFiles.forEach((f) => {
          if (f.previewUrl) previewUrls[f.name] = f.previewUrl
        })
      }
    }

    const userMsg = {
      id: Math.random().toString(36).slice(2),
      role: "user",
      content: text,
      files: apiFiles.map((f) => ({ name: f.name, previewUrl: f.previewUrl || previewUrls[f.name] })),
      apiFiles,
      createdAt: now,
    }

    // Auto-name chat: like ChatGPT / Claude, automatically title the chat when user submits prompt
    const convToUpdate = conversations.find((c) => c.id === convId)
    const isInitialTitle =
      !convToUpdate ||
      convToUpdate.title === "New Analysis" ||
      convToUpdate.title.startsWith("New Analysis") ||
      convToUpdate.title === "New Chat"

    const autoTitle = isInitialTitle ? generateChatTitle(text) : convToUpdate.title

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c
        const msgs = [...(c.messages || []), userMsg]
        return {
          ...c,
          title: autoTitle,
          messages: msgs,
          updatedAt: now,
          messageCount: msgs.length,
          preview: text.slice(0, 80),
        }
      }),
    )

    setIsThinking(true)
    setThinkingConvId(convId)

    try {
      const trace = await sendSatQuery({
        query: text,
        files: apiFiles.map((f) => ({ name: f.name, b64: f.b64 })),
      })

      const asstMsg = {
        id: Math.random().toString(36).slice(2),
        role: "assistant",
        content: trace.output?.text || (trace.abstained ? trace.input_check?.message : "No text returned"),
        trace,
        previewUrls,
        isLive: true,
        createdAt: new Date().toISOString(),
      }

      // Refine chat title using multi-agent trace metadata if available
      const refinedTitle = isInitialTitle ? generateChatTitle(text, trace) : autoTitle

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c
          const msgs = [...(c.messages || []), asstMsg]
          return {
            ...c,
            title: refinedTitle,
            messages: msgs,
            updatedAt: new Date().toISOString(),
            messageCount: msgs.length,
            preview: asstMsg.content.slice(0, 80),
          }
        }),
      )
    } catch (err) {
      console.error("SatQuery execution failed:", err)
      const errorMsg = {
        id: Math.random().toString(36).slice(2),
        role: "assistant",
        content: `⚠️ **SatQuery API Error**: ${err.message}\n\nPlease verify that the backend server is running on port 8765 (\`python backend/run_server.py\`) or check your network connection.`,
        createdAt: new Date().toISOString(),
      }
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c
          const msgs = [...(c.messages || []), errorMsg]
          return {
            ...c,
            messages: msgs,
            updatedAt: new Date().toISOString(),
            messageCount: msgs.length,
            preview: errorMsg.content.slice(0, 80),
          }
        }),
      )
    } finally {
      setIsThinking(false)
      setThinkingConvId(null)
    }
  }

  function editMessage(convId, messageId, newContent) {
    const now = new Date().toISOString()
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c
        const msgs = (c.messages || []).map((m) =>
          m.id === messageId ? { ...m, content: newContent, editedAt: now } : m,
        )
        return {
          ...c,
          messages: msgs,
          preview: msgs[msgs.length - 1]?.content?.slice(0, 80) || c.preview,
        }
      }),
    )
  }

  function resendMessage(convId, messageId) {
    const conv = conversations.find((c) => c.id === convId)
    const msg = conv?.messages?.find((m) => m.id === messageId)
    if (!msg) return
    sendMessage(convId, msg.content)
  }

  function pauseThinking() {
    setIsThinking(false)
    setThinkingConvId(null)
  }

  function handleUseTemplate(template) {
    if (composerRef.current) {
      composerRef.current.insertTemplate(template.content)
    }
  }

  const handleUpdateAvatar = async (newAvatarId) => {
    if (currentUser) {
      const updatedUser = { ...currentUser, avatarId: newAvatarId }
      setCurrentUser(updatedUser)
      try {
        await fetch("/api/auth/avatar", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ avatarId: newAvatarId }),
        })
      } catch (err) {
        console.error("Failed to update avatar on backend:", err)
      }
    }
  }

  // Lock document and body overflow to eliminate any external window scrolling
  useEffect(() => {
    const origHtmlOverflow = document.documentElement.style.overflow
    const origBodyOverflow = document.body.style.overflow
    document.documentElement.style.overflow = "hidden"
    document.body.style.overflow = "hidden"
    return () => {
      document.documentElement.style.overflow = origHtmlOverflow
      document.body.style.overflow = origBodyOverflow
    }
  }, [])

  const selected = conversations.find((c) => c.id === selectedId) || null

  return (
    <div className="fixed inset-0 flex flex-col h-full w-full overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Mobile Top Header with Sidebar Trigger */}
      <div className="md:hidden shrink-0 sticky top-0 z-40 flex items-center gap-2 border-b border-zinc-200/60 bg-white/80 px-3 py-2 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/70">
        <button
          type="button"
          onClick={() => setSidebarOpen(true)}
          className="rounded-lg p-1.5 text-zinc-600 hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:text-zinc-300 dark:hover:bg-zinc-800"
          aria-label="Open sidebar"
          title="Open sidebar"
        >
          <PanelLeftOpen className="h-5 w-5" />
        </button>
        <div className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="inline-flex h-4 w-4 items-center justify-center">🛰️</span> SatQuery AI
        </div>
        <div className="ml-auto flex items-center gap-2">
          <GhostIconButton label="Schedule">
            <Calendar className="h-4 w-4" />
          </GhostIconButton>
          <GhostIconButton label="Apps">
            <LayoutGrid className="h-4 w-4" />
          </GhostIconButton>
          <GhostIconButton label="More">
            <MoreHorizontal className="h-4 w-4" />
          </GhostIconButton>
          <ThemeToggle theme={theme} setTheme={setTheme} />
        </div>
      </div>

      <div className="mx-auto flex w-full max-w-[1400px] flex-1 min-h-0 overflow-hidden">
        <Sidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          theme={theme}
          setTheme={setTheme}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          sidebarCollapsed={sidebarCollapsed}
          setSidebarCollapsed={setSidebarCollapsed}
          conversations={conversations}
          pinned={pinned}
          recent={recent}
          folders={folders}
          folderCounts={folderCounts}
          selectedId={selectedId}
          onSelect={(id) => setSelectedId(id)}
          togglePin={togglePin}
          onDeleteConversation={deleteConversation}
          onRenameConversation={renameConversation}
          onClearAllConversations={clearAllConversations}
          query={query}
          setQuery={setQuery}
          searchRef={searchRef}
          createFolder={createFolder}
          createNewChat={createNewChat}
          templates={templates}
          setTemplates={setTemplates}
          onUseTemplate={handleUseTemplate}
          currentUser={currentUser}
          onOpenAuth={() => setShowAuthModal(true)}
          onOpenAvatarPicker={() => setShowAvatarPickerModal(true)}
          onLogout={handleLogout}
        />

        <main className="relative flex min-w-0 flex-1 flex-col h-full overflow-hidden">
          <Header
            createNewChat={createNewChat}
            sidebarCollapsed={sidebarCollapsed}
            setSidebarOpen={setSidebarOpen}
            backendHealth={backendHealth}
            onRefreshHealth={refreshHealth}
            currentUser={currentUser}
            onOpenAuth={() => setShowAuthModal(true)}
            onOpenAvatarPicker={() => setShowAvatarPickerModal(true)}
            onLogout={handleLogout}
          />
          <ChatPane
            ref={composerRef}
            conversation={selected}
            onSend={(payload) => selected && sendMessage(selected.id, payload)}
            onEditMessage={(messageId, newContent) => selected && editMessage(selected.id, messageId, newContent)}
            onResendMessage={(messageId) => selected && resendMessage(selected.id, messageId)}
            isThinking={isThinking && thinkingConvId === selected?.id}
            onPauseThinking={pauseThinking}
          />
        </main>
      </div>

      {/* Neon DB User Authentication Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={handleAuthSuccess}
      />

      {/* KokonutUI Avatar Picker Modal */}
      <AvatarPickerModal
        isOpen={showAvatarPickerModal}
        onClose={() => setShowAvatarPickerModal(false)}
        currentAvatarId={currentUser?.avatarId || 1}
        onSelectAvatar={handleUpdateAvatar}
      />
    </div>
  )
}
