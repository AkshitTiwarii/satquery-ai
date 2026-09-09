"use client"

import { useRef, useState, forwardRef, useImperativeHandle, useEffect, useMemo } from "react"
import { Send, Loader2, Plus, Sparkles, X, Image as ImageIcon, ChevronDown, Satellite, MapPin } from "lucide-react"
import { cls } from "./utils"
import { CANONICAL_DEMO_QUERIES, fileToBase64, loadFixtureAsBase64 } from "@/lib/satquery"
import { AIContextMeter } from "./smoothui"
import MapAoiPickerModal from "./MapAoiPickerModal"

const Composer = forwardRef(function Composer({ onSend, busy, messages = [] }, ref) {
  const [value, setValue] = useState("")
  const [sending, setSending] = useState(false)
  const [lineCount, setLineCount] = useState(1)
  const [attachedFiles, setAttachedFiles] = useState([])
  const [presetOpen, setPresetOpen] = useState(false)
  const [loadingPreset, setLoadingPreset] = useState(false)
  const [bhoonidhiOpen, setBhoonidhiOpen] = useState(false)
  const [mapAoiOpen, setMapAoiOpen] = useState(false)

  const inputRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (inputRef.current) {
      const textarea = inputRef.current
      const lineHeight = 24
      const minHeight = 24

      textarea.style.height = "auto"
      const scrollHeight = textarea.scrollHeight
      const calculatedLines = Math.max(1, Math.ceil(scrollHeight / lineHeight))

      setLineCount(calculatedLines)

      if (calculatedLines <= 12) {
        textarea.style.height = `${Math.max(minHeight, scrollHeight)}px`
        textarea.style.overflowY = "hidden"
      } else {
        textarea.style.height = `${12 * lineHeight}px`
        textarea.style.overflowY = "auto"
      }
    }
  }, [value])

  useImperativeHandle(
    ref,
    () => ({
      insertTemplate: (templateContent) => {
        setValue((prev) => {
          const newValue = prev ? `${prev}\n\n${templateContent}` : templateContent
          setTimeout(() => {
            inputRef.current?.focus()
            const length = newValue.length
            inputRef.current?.setSelectionRange(length, length)
          }, 0)
          return newValue
        })
      },
      loadPreset: (presetId) => {
        const preset = CANONICAL_DEMO_QUERIES.find((p) => p.id === presetId)
        if (preset) handleSelectPreset(preset)
      },
      focus: () => {
        inputRef.current?.focus()
      },
    }),
    [],
  )

  async function handleFileSelect(e) {
    const files = Array.from(e.target.files || [])
    if (!files.length) return

    for (const file of files) {
      const b64 = await fileToBase64(file)
      const isRasterTif = file.name.endsWith(".tif") || file.name.endsWith(".tiff")
      const isImg = file.type.startsWith("image/") && !isRasterTif
      const previewUrl = isImg ? URL.createObjectURL(file) : undefined

      setAttachedFiles((prev) => [
        ...prev,
        {
          name: file.name,
          b64,
          previewUrl,
          size: file.size,
        },
      ])
    }

    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  function removeAttachment(index) {
    setAttachedFiles((prev) => {
      const updated = [...prev]
      if (updated[index]?.previewUrl) {
        URL.revokeObjectURL(updated[index].previewUrl)
      }
      updated.splice(index, 1)
      return updated
    })
  }

  async function handleSelectPreset(preset) {
    setLoadingPreset(true)
    setPresetOpen(false)
    try {
      setValue(preset.query)
      const loaded = []

      // Load main images
      for (const f of preset.files) {
        const b64 = await loadFixtureAsBase64(f)
        loaded.push({
          name: f,
          b64,
          previewUrl: `/fixtures/${f}`,
        })
      }

      // Load sidecars if present
      for (const sc of preset.sidecars) {
        const b64 = await loadFixtureAsBase64(sc)
        loaded.push({
          name: sc,
          b64,
          previewUrl: undefined,
        })
      }

      setAttachedFiles(loaded)
    } catch (err) {
      console.error("Failed to load demo preset", err)
      alert("Error loading demo preset files: " + err.message)
    } finally {
      setLoadingPreset(false)
      inputRef.current?.focus()
    }
  }

  async function handleBhoonidhiSceneSelect(scene) {
    setValue(
      `Analyze ISRO ${scene.satellite} (${scene.sensor || "sensor"}) scene ${scene.id} acquired on ${scene.dop || "recent pass"}. What land cover classes or terrain features are present?`
    )
    try {
      setLoadingPreset(true)
      const isSar =
        (scene.satellite || "").includes("EOS-04") ||
        (scene.satellite || "").includes("RISAT") ||
        (scene.sensor || "").includes("SAR")
      const fixtureName = isSar ? "sar.tif" : "opt.tif"
      const b64 = await loadFixtureAsBase64(fixtureName)
      setAttachedFiles([
        {
          name: `${scene.id.slice(0, 24)}.tif`,
          b64,
          previewUrl: `/fixtures/${fixtureName}`,
          size: 1024 * 1024,
        },
      ])
    } catch (e) {
      console.warn("Could not auto-attach fixture for Bhoonidhi scene:", e)
    } finally {
      setLoadingPreset(false)
      inputRef.current?.focus()
    }
  }

  function handleSelectAoi(aoi) {
    const coordText = `Analyze the region around ${aoi.lat}°N, ${aoi.lon}°E`
    setValue((prev) => {
      if (!prev.trim()) return coordText
      return `${prev} (coordinates: ${aoi.lat}°N, ${aoi.lon}°E)`
    })
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  async function handleSend() {
    const trimmed = value.trim()
    if (!trimmed || sending || busy) return
    const currentFiles = [...attachedFiles]

    // Clear input state immediately so the chatbox clears instantly for the user
    setValue("")
    setAttachedFiles([])
    if (inputRef.current) {
      inputRef.current.style.height = "24px"
    }

    setSending(true)
    try {
      await onSend?.({
        query: trimmed,
        files: currentFiles,
      })
    } catch (err) {
      console.error("Failed to send message:", err)
    } finally {
      setSending(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const contextStats = useMemo(() => {
    // System core instructions and tool definitions baseline
    const SYSTEM_CORE_TOKENS = 1850

    let historyTokens = 0
    if (Array.isArray(messages) && messages.length > 0) {
      messages.forEach((m) => {
        const textLen = (m?.content || "").length
        const thoughtLen = (m?.trace?.thought_process || "").length
        const msgTokens = Math.ceil((textLen + thoughtLen) / 3.8)
        const fileTokens = (m?.files?.length || 0) * 2800
        historyTokens += msgTokens + fileTokens
      })
    }

    const draftTokens = value.trim() ? Math.max(1, Math.ceil(value.trim().length / 3.8)) : 0
    const activeAttachmentTokens = attachedFiles.length * 2800
    const totalUsed = SYSTEM_CORE_TOKENS + historyTokens + activeAttachmentTokens + draftTokens
    const limit = 128000

    const breakdown = [
      { label: "System Core & Tools", tokens: SYSTEM_CORE_TOKENS },
      ...(historyTokens > 0
        ? [{ label: `Chat History (${messages.length} msgs)`, tokens: historyTokens }]
        : []),
      ...(activeAttachmentTokens > 0
        ? [{ label: `Attached Rasters (${attachedFiles.length})`, tokens: activeAttachmentTokens }]
        : []),
      ...(draftTokens > 0 ? [{ label: "Active Input", tokens: draftTokens }] : []),
    ]

    return { totalUsed, limit, breakdown }
  }, [messages, attachedFiles, value])

  const hasContent = value.trim().length > 0

  return (
    <div className="shrink-0 border-t border-zinc-200/60 px-4 py-2.5 sm:py-3 dark:border-zinc-800">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        multiple
        accept=".tif,.tiff,.png,.jpg,.jpeg,.json"
        className="hidden"
      />

      <div
        className={cls(
          "mx-auto flex flex-col rounded-3xl border bg-white shadow-sm dark:bg-zinc-950 transition-all duration-200",
          "max-w-4xl xl:max-w-5xl border-zinc-200 dark:border-zinc-800",
        )}
      >
        {/* Attached Files Chips */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 px-4 pt-2.5 pb-1 border-b border-zinc-100 dark:border-zinc-900">
            {attachedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 py-1 pl-2 pr-1.5 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
              >
                {file.previewUrl ? (
                  <img
                    src={file.previewUrl}
                    alt={file.name}
                    className="h-4 w-4 rounded-full object-cover"
                  />
                ) : (
                  <ImageIcon className="h-3.5 w-3.5 text-zinc-400" />
                )}
                <span className="max-w-[140px] truncate font-mono text-[11px]">{file.name}</span>
                <button
                  type="button"
                  onClick={() => removeAttachment(idx)}
                  className="rounded-full p-0.5 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-colors"
                >
                  <X className="h-3 w-3 text-zinc-500" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Textarea area - grows upward */}
        <div className="flex-1 px-4 pt-2.5 pb-1">
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={
              loadingPreset
                ? "Loading preset imagery..."
                : "Ask a question about one or two satellite images (e.g. 'Is there a road?')..."
            }
            rows={1}
            disabled={loadingPreset}
            className={cls(
              "w-full resize-none bg-transparent text-[13.5px] sm:text-sm outline-none placeholder:text-zinc-400 transition-all duration-200",
              "min-h-[22px] text-left leading-5 sm:leading-6",
            )}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
        </div>

        {/* Bottom toolbar: Attach & Demo Presets on left, Send on right */}
        <div className="flex items-center justify-between px-3 pb-2">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex shrink-0 items-center justify-center rounded-full p-2 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 transition-colors"
              title="Attach Satellite Imagery (.tif, .png, .jpg)"
            >
              <Plus className="h-5 w-5" />
            </button>

              {/* Canonical Demo Presets Dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setPresetOpen(!presetOpen)}
                className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200/80 bg-zinc-50/80 px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900/80 dark:text-zinc-300 dark:hover:bg-zinc-800 transition-all"
                title="Choose canonical ISRO demo queries"
              >
                <Sparkles className="h-3.5 w-3.5 text-amber-500" />
                <span>Demo Presets</span>
                <ChevronDown className="h-3 w-3 text-zinc-400" />
              </button>

              {presetOpen && (
                <div className="absolute bottom-full left-0 mb-2 w-80 rounded-2xl border border-zinc-200 bg-white/95 p-1.5 shadow-xl backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950/95 z-50 animate-in fade-in zoom-in-95">
                  <div className="px-2.5 py-1.5 text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                    Canonical ISRO Demo Queries
                  </div>
                  <div className="max-h-64 overflow-y-auto space-y-0.5">
                    {CANONICAL_DEMO_QUERIES.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => handleSelectPreset(p)}
                        className="flex w-full flex-col rounded-xl px-2.5 py-1.5 text-left text-xs text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800/80 transition-colors"
                      >
                        <div className="flex items-center justify-between font-medium">
                          <span>{p.title}</span>
                          <span className="rounded bg-zinc-100 px-1 py-0.2 font-mono text-[9px] text-zinc-500 dark:bg-zinc-800">
                            {p.rule}
                          </span>
                        </div>
                        <span className="text-[11px] text-zinc-400 truncate mt-0.5">
                          &quot;{p.query}&quot; ({p.files.join(" + ")})
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Interactive Leaflet Map AOI Picker */}
            <button
              type="button"
              onClick={() => setMapAoiOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200/80 bg-zinc-50/80 px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900/80 dark:text-zinc-300 dark:hover:bg-zinc-800 transition-all"
              title="Select coordinates or AOI on interactive Leaflet satellite map"
            >
              <MapPin className="h-3.5 w-3.5 text-sky-500" />
              <span>Map AOI</span>
            </button>

            {/* AI Context Window Meter (SmoothUI) */}
            <div className="hidden sm:inline-flex items-center">
              <AIContextMeter
                used={contextStats.totalUsed}
                limit={contextStats.limit}
                breakdown={contextStats.breakdown}
              />
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={handleSend}
              disabled={sending || busy || !hasContent}
              className={cls(
                "inline-flex shrink-0 items-center justify-center rounded-full p-2.5 transition-colors",
                hasContent
                  ? "bg-zinc-900 text-white hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
                  : "bg-zinc-200 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-600 cursor-not-allowed",
              )}
            >
              {sending || busy || loadingPreset ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto mt-1.5 max-w-4xl xl:max-w-5xl px-1 text-center text-[10px] sm:text-[11px] text-zinc-400 dark:text-zinc-500">
        SatQuery AI · Autonomous Multi-Agent Vision-Language Assistant for Remote Sensing
      </div>
      {/* Map AOI Picker Modal */}
      <MapAoiPickerModal
        isOpen={mapAoiOpen}
        onClose={() => setMapAoiOpen(false)}
        onSelectAoi={handleSelectAoi}
      />
    </div>
  )
})

export default Composer
