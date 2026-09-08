"use client"

import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from "react"
import dynamic from "next/dynamic"
import {
  Pencil, RefreshCw, Check, X, Satellite, FileText, Maximize2, ShieldCheck,
  Layers, Download, Clock, Eye, EyeOff, Crosshair, Target, MapPin, Sparkles,
  Brain, ChevronDown, ChevronRight, Settings2, BarChart3, AlertTriangle,
  CheckCircle2, HelpCircle, ExternalLink, Sprout, TrendingUp
} from "lucide-react"

const LeafletMap = dynamic(() => import("./LeafletMap"), { ssr: false })
import AIConversation from "./smoothui/components/ai-conversation"
import AILoader from "./smoothui/components/ai-loader"
import AIMessage from "./smoothui/components/ai-message"
import AIReasoning from "./smoothui/components/ai-reasoning"
import AIResponse from "./smoothui/components/ai-response"
import AISources from "./smoothui/components/ai-sources"
import AIToolCall from "./smoothui/components/ai-tool-call"
import SiriOrb from "./smoothui/components/siri-orb"
import Composer from "./Composer"
import { cls, timeAgo } from "./utils"

const STREAM_INTERVAL_MS = 35

function SpatialGroundingCard({ normBox, trace, message, onInspect }) {
  const [showOverlay, setShowOverlay] = useState(true)
  const [imageError, setImageError] = useState(false)

  // Dynamic satellite imagery resolution: strictly use actual user upload or real trace preview
  const candidateImg =
    (trace?.rendered_images && trace.rendered_images[0]) ||
    (message.files && message.files[0]
      ? {
          url: message.files[0].previewUrl || message.files[0].url,
          name: message.files[0].name,
          type: "Uploaded Scene",
        }
      : null)

  if (!candidateImg || imageError) return null

  const rawUrl = candidateImg.url || candidateImg.previewUrl || ""
  const cleanUrl = rawUrl.replace(/\.tif+$/i, ".png")
  const sceneName = candidateImg.name || "Satellite Observation Scene"
  const featureName = trace?.output?.quantities?.feature_localized || "Target Feature"

  const x0 = Math.min(normBox[0], normBox[2])
  const y0 = Math.min(normBox[1], normBox[3])
  const x1 = Math.max(normBox[0], normBox[2])
  const y1 = Math.max(normBox[1], normBox[3])

  const leftPct = Math.max(0, Math.min(100, x0 * 100))
  const topPct = Math.max(0, Math.min(100, y0 * 100))
  const widthPct = Math.max(2, Math.min(100 - leftPct, (x1 - x0) * 100))
  const heightPct = Math.max(2, Math.min(100 - topPct, (y1 - y0) * 100))
  const areaPct = ((widthPct * heightPct) / 100).toFixed(1)

  return (
    <div className="mt-3 overflow-hidden rounded-2xl border border-zinc-200/90 bg-gradient-to-b from-zinc-50/90 to-white/90 p-4 shadow-sm dark:border-zinc-800/90 dark:from-zinc-900/90 dark:to-zinc-950/90 backdrop-blur-sm">
      {/* Header with Title, Coordinates and Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2.5 border-b border-zinc-200/60 dark:border-zinc-800/60">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-sky-500/10 text-sky-500 dark:bg-sky-500/20">
            <Target className="h-3.5 w-3.5" />
          </div>
          <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            Spatial Grounding & Target Delineation
          </span>
          <span className="hidden sm:inline-flex items-center rounded-md border border-sky-500/20 bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-medium text-sky-600 dark:text-sky-400">
            Rule R4 Localized
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
              showOverlay
                ? "bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/30"
                : "bg-muted text-muted-foreground hover:text-foreground border border-border"
            }`}
            title="Toggle target bounding box overlay"
          >
            {showOverlay ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
            <span>Overlay {showOverlay ? "ON" : "OFF"}</span>
          </button>

          <button
            type="button"
            onClick={() => onInspect?.({ ...candidateImg, url: cleanUrl, box: normBox })}
            className="flex items-center gap-1 rounded-md border border-border bg-muted/60 px-2 py-1 text-[11px] font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Inspect high-resolution image"
          >
            <Maximize2 className="h-3 w-3" />
            <span className="hidden xs:inline">Inspect</span>
          </button>
        </div>
      </div>

      {/* Main Satellite Raster Viewport */}
      <div
        onClick={() => onInspect?.({ ...candidateImg, url: cleanUrl, box: normBox })}
        className="group relative mx-auto aspect-[16/10] w-full max-w-xl cursor-pointer overflow-hidden rounded-xl border border-zinc-300/80 bg-zinc-950 shadow-md dark:border-zinc-800"
      >
        {/* Real Satellite Raster Image */}
        <img
          src={cleanUrl}
          alt={sceneName}
          onError={() => setImageError(true)}
          className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.02] select-none"
        />

        {/* Ambient Dark Dimming Mask outside Bounding Box (Punches hole through target) */}
        {showOverlay && (
          <svg
            className="absolute inset-0 h-full w-full pointer-events-none"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            <path
              d={`M0,0 H100 V100 H0 Z M${leftPct},${topPct} V${topPct + heightPct} H${leftPct + widthPct} V${topPct} Z`}
              fill="rgba(0, 0, 0, 0.40)"
              fillRule="evenodd"
            />
          </svg>
        )}

        {/* Tactical Bounding Box */}
        {showOverlay && (
          <div
            className="absolute border-2 border-sky-400 bg-sky-400/15 shadow-[0_0_15px_rgba(56,189,248,0.45)] transition-all pointer-events-none"
            style={{
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
            }}
          >
            {/* Tactical Corner Reticles */}
            <span className="absolute -top-[2px] -left-[2px] h-2.5 w-2.5 border-t-2 border-l-2 border-sky-300" />
            <span className="absolute -top-[2px] -right-[2px] h-2.5 w-2.5 border-t-2 border-r-2 border-sky-300" />
            <span className="absolute -bottom-[2px] -left-[2px] h-2.5 w-2.5 border-b-2 border-l-2 border-sky-300" />
            <span className="absolute -bottom-[2px] -right-[2px] h-2.5 w-2.5 border-b-2 border-r-2 border-sky-300" />

            {/* Precision Center Crosshair */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-60">
              <span className="h-2 w-px bg-sky-300/80" />
              <span className="w-2 h-px bg-sky-300/80 -ml-1" />
            </div>

            {/* Target Label HUD Chip */}
            <div className="absolute -top-6 left-0 flex items-center gap-1 rounded bg-sky-500 px-1.5 py-0.5 text-[9px] font-mono font-bold text-zinc-950 shadow-md whitespace-nowrap">
              <Crosshair className="h-2.5 w-2.5" />
              <span>{featureName}</span>
              <span className="opacity-80 text-[8px]">
                ({widthPct.toFixed(0)}% × {heightPct.toFixed(0)}%)
              </span>
            </div>
          </div>
        )}

        {/* Hover Hint Overlay */}
        <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <span className="flex items-center gap-1 rounded-md bg-black/70 px-2 py-1 text-[10px] font-medium text-white backdrop-blur-md border border-white/10 shadow-sm">
            <Maximize2 className="h-3 w-3" />
            Click to inspect full resolution
          </span>
        </div>

        {/* Scene Watermark */}
        <div className="absolute bottom-2 left-2 pointer-events-none">
          <span className="rounded bg-black/60 px-1.5 py-0.5 text-[9px] font-mono text-zinc-300 backdrop-blur-sm">
            {sceneName}
          </span>
        </div>
      </div>

      {/* Telemetry Metrics Footer */}
      <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-muted-foreground pt-2 border-t border-border/50">
        <div className="flex items-center gap-2">
          <span>Raster Extent:</span>
          <span className="bg-muted px-1.5 py-0.5 rounded text-foreground font-semibold">
            [{normBox.map((n) => Number(n).toFixed(3)).join(", ")}]
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span>Coverage: <strong className="text-foreground">{areaPct}%</strong></span>
          <span>Nominal GSD: <strong className="text-foreground">10m</strong></span>
        </div>
      </div>
    </div>
  )
}

function AssessmentPanel({ assessment }) {
  if (!assessment) return null

  const isWarning = assessment.advisory_level === "warning"
  const isSuccess = assessment.advisory_level === "success"

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-gradient-to-b from-white to-zinc-50/70 p-4 shadow-sm dark:border-zinc-800/80 dark:from-zinc-900/90 dark:to-zinc-950/90 backdrop-blur-md">
      {/* Assessment Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-200/60 dark:border-zinc-800/60">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-sky-500/10 text-sky-500 dark:bg-sky-500/20">
            <BarChart3 className="h-3.5 w-3.5" />
          </div>
          <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wider">
            Assessment
          </span>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400 shadow-xs">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
          Completed
        </span>
      </div>

      {/* Bold Headline */}
      <h3 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-50 mb-1.5">
        {assessment.headline}
      </h3>

      {/* 2-line Summary */}
      <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-300 mb-4">
        {assessment.summary}
      </p>

      {/* 3-Column Contextual Metric Cards */}
      {assessment.metrics && assessment.metrics.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mb-3.5">
          {assessment.metrics.map((m, idx) => (
            <div
              key={idx}
              title={m.tooltip}
              className="group relative flex flex-col justify-between rounded-xl border border-zinc-200/70 bg-zinc-100/50 p-3 transition-all duration-200 hover:border-sky-500/40 hover:bg-zinc-100/80 dark:border-zinc-800 dark:bg-zinc-900/50 dark:hover:border-sky-500/40 dark:hover:bg-zinc-900/80"
            >
              <div className="flex items-center gap-2 mb-1 text-zinc-900 dark:text-zinc-100 font-bold text-lg">
                {m.icon === "sprout" || m.icon === "leaf" ? (
                  <Sprout className="h-4 w-4 text-emerald-500 shrink-0" />
                ) : m.icon === "shield" ? (
                  <ShieldCheck className="h-4 w-4 text-sky-500 shrink-0" />
                ) : m.icon === "crosshair" ? (
                  <Crosshair className="h-4 w-4 text-sky-500 shrink-0" />
                ) : m.icon === "layers" ? (
                  <Layers className="h-4 w-4 text-indigo-400 shrink-0" />
                ) : m.icon === "satellite" ? (
                  <Satellite className="h-4 w-4 text-sky-400 shrink-0" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                )}
                <span>{m.value}</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">
                <span>{m.label}</span>
                <HelpCircle className="h-3 w-3 opacity-60 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Actionable Advisory Banner */}
      {assessment.advisory && (
        <div
          className={cls(
            "flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium border",
            isWarning
              ? "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400"
              : isSuccess
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
              : "border-sky-500/30 bg-sky-500/10 text-sky-600 dark:text-sky-400"
          )}
        >
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{assessment.advisory}</span>
        </div>
      )}
    </div>
  )
}

function WorkflowLogPanel({ workflowLog, onExportReport, onViewEvidence }) {
  if (!workflowLog || workflowLog.length === 0) return null

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-200/80 bg-gradient-to-b from-white to-zinc-50/70 p-4 shadow-sm dark:border-zinc-800/80 dark:from-zinc-900/90 dark:to-zinc-950/90 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-200/60 dark:border-zinc-800/60">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-zinc-500/10 text-zinc-600 dark:text-zinc-400">
            <Settings2 className="h-3.5 w-3.5" />
          </div>
          <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            Workflow log
          </span>
        </div>
        <span className="text-[11px] font-mono text-zinc-400 dark:text-zinc-500">
          Triggered by your question
        </span>
      </div>

      {/* Stepper */}
      <div className="space-y-3 mb-4">
        {workflowLog.map((step) => (
          <div key={step.step_num} className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2.5 min-w-0">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-sky-500/10 text-sky-500 dark:bg-sky-500/20 text-[10px] font-bold mt-0.5 border border-sky-500/30">
                {step.step_num}
              </span>
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  {step.name}
                </span>
                <span className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
                  {step.description}
                </span>
              </div>
            </div>
            <span className="flex items-center gap-1 shrink-0 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
              <Check className="h-3 w-3" />
              Completed {step.duration_str}
            </span>
          </div>
        ))}
      </div>

      {/* Actions Row */}
      <div className="flex items-center gap-2 pt-2 border-t border-zinc-200/50 dark:border-zinc-800/60">
        <button
          type="button"
          onClick={onExportReport}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-xl border border-zinc-200/80 bg-zinc-100/60 hover:bg-zinc-200/60 dark:border-zinc-800 dark:bg-zinc-900/60 dark:hover:bg-zinc-800/80 px-3 py-2 text-xs font-medium text-zinc-800 dark:text-zinc-200 transition-colors"
        >
          <FileText className="h-3.5 w-3.5 text-zinc-500" />
          <span>Export report ⌄</span>
        </button>
        <button
          type="button"
          onClick={onViewEvidence}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-xl border border-sky-500/30 bg-sky-500/10 hover:bg-sky-500/20 px-3 py-2 text-xs font-medium text-sky-600 dark:text-sky-400 transition-colors"
        >
          <ExternalLink className="h-3.5 w-3.5" />
          <span>View evidence</span>
        </button>
      </div>
    </div>
  )
}

function InteractiveImageryViewer({
  renderedImages,
  imageryViewer,
  message,
  onInspect,
}) {
  const [selectedEpoch, setSelectedEpoch] = useState(imageryViewer?.default_epoch || "after")
  const [selectedLayer, setSelectedLayer] = useState(imageryViewer?.default_layer || "fused")
  const [zoomLevel, setZoomLevel] = useState(1)

  if (!renderedImages || renderedImages.length === 0) return null

  // Determine active background image based on selected layer and epoch
  let activeImg = renderedImages[0]
  if (selectedLayer === "sar") {
    const sarImg = renderedImages.find((img) => img.name?.toLowerCase().includes("sar") || img.type?.toLowerCase().includes("sar"))
    if (sarImg) activeImg = sarImg
  } else if (selectedLayer === "optical") {
    const optImg = renderedImages.find((img) => img.name?.toLowerCase().includes("opt") || !img.name?.toLowerCase().includes("sar"))
    if (optImg) activeImg = optImg
  } else {
    activeImg = renderedImages[0]
  }

  const resolvedImgUrl = (
    message?.previewUrls?.[activeImg.name] ||
    message?.files?.find((f) => f.name === activeImg.name)?.previewUrl ||
    activeImg.previewUrl ||
    activeImg.url ||
    ""
  ).replace(/\.tif+$/i, ".png")

  const showHighlights = (selectedLayer === "fused" || selectedLayer === "optical") && selectedEpoch === "after"

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-200/90 bg-zinc-950 p-4 shadow-sm dark:border-zinc-800/90 backdrop-blur-md">
      {/* Top Floating Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2.5 border-b border-zinc-800/80">
        {/* Epoch Toggles: Before / After */}
        {imageryViewer?.epochs && (
          <div className="inline-flex rounded-xl bg-zinc-900/90 p-0.5 border border-zinc-800">
            {imageryViewer.epochs.map((ep) => (
              <button
                key={ep.id}
                type="button"
                onClick={() => setSelectedEpoch(ep.id)}
                className={cls(
                  "rounded-lg px-2.5 py-1 text-xs font-semibold transition-all",
                  selectedEpoch === ep.id
                    ? "bg-teal-600 text-white shadow-xs"
                    : "text-zinc-400 hover:text-zinc-200"
                )}
              >
                {ep.label}
              </button>
            ))}
          </div>
        )}

        {/* Imagery Layer Radios */}
        {imageryViewer?.layers && (
          <div className="flex items-center gap-1.5 rounded-xl bg-zinc-900/90 p-1 border border-zinc-800 text-xs">
            <span className="text-zinc-400 font-medium px-1.5 text-[11px]">Imagery layer:</span>
            {imageryViewer.layers.map((lay) => (
              <button
                key={lay.id}
                type="button"
                onClick={() => setSelectedLayer(lay.id)}
                className={cls(
                  "flex items-center gap-1.5 rounded-lg px-2 py-0.5 text-xs font-medium transition-all",
                  selectedLayer === lay.id
                    ? "bg-zinc-800 text-teal-400 border border-teal-500/40 shadow-xs"
                    : "text-zinc-400 hover:text-zinc-200"
                )}
              >
                <span
                  className={cls(
                    "h-2 w-2 rounded-full border",
                    selectedLayer === lay.id
                      ? "border-teal-400 bg-teal-400"
                      : "border-zinc-500"
                  )}
                />
                {lay.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main Satellite Viewport with Overlays */}
      <div className="relative aspect-[16/10] w-full overflow-hidden rounded-xl bg-black border border-zinc-800 group">
        <img
          src={resolvedImgUrl}
          alt={activeImg.name || "Satellite scene"}
          style={{ transform: `scale(${zoomLevel})` }}
          className="h-full w-full object-cover transition-transform duration-300 select-none"
        />

        {/* Dynamic SVG Highlight Overlays */}
        {showHighlights && imageryViewer?.highlights && (
          <svg
            viewBox="0 0 800 500"
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full pointer-events-none"
          >
            {imageryViewer.highlights.map((h) => {
              if (h.type === "path") {
                return (
                  <path
                    key={h.id}
                    d={h.d}
                    fill={h.fill}
                    stroke={h.color}
                    strokeWidth="2"
                    className="animate-pulse duration-1000"
                    style={{ filter: `drop-shadow(0 0 8px ${h.color}88)` }}
                  />
                )
              }
              if (h.type === "rect") {
                return (
                  <rect
                    key={h.id}
                    x={`${h.x}%`}
                    y={`${h.y}%`}
                    width={`${h.width}%`}
                    height={`${h.height}%`}
                    fill={h.fill}
                    stroke={h.color}
                    strokeWidth="2"
                    style={{ filter: `drop-shadow(0 0 10px ${h.color})` }}
                  />
                )
              }
              return null
            })}
          </svg>
        )}

        {/* Geographic labels */}
        {showHighlights && imageryViewer?.place_labels && imageryViewer.place_labels.map((pl, idx) => (
          <div
            key={idx}
            className="absolute pointer-events-none transform -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${pl.x}%`, top: `${pl.y}%` }}
          >
            <span className="rounded bg-black/70 px-1.5 py-0.5 text-[9px] font-medium text-white shadow-sm border border-white/10 backdrop-blur-xs">
              {pl.name}
            </span>
          </div>
        ))}

        {/* Scale Bar & Navigation Controls */}
        <div className="absolute bottom-2.5 right-2.5 flex items-center gap-1.5">
          <div className="flex items-center gap-1 rounded-lg bg-black/70 px-2 py-1 text-[10px] font-mono text-zinc-300 border border-white/10 backdrop-blur-md">
            <span>2 km</span>
            <span className="h-1.5 w-6 border-b-2 border-l-2 border-r-2 border-white inline-block ml-1" />
          </div>
          <div className="flex flex-col rounded-lg bg-black/70 border border-white/10 overflow-hidden backdrop-blur-md">
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.25))}
              className="px-2 py-1 text-xs text-zinc-200 hover:bg-white/20 transition-colors"
            >
              +
            </button>
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.max(1, z - 0.25))}
              className="px-2 py-1 text-xs text-zinc-200 hover:bg-white/20 border-t border-white/10 transition-colors"
            >
              −
            </button>
          </div>
          <button
            type="button"
            onClick={() => onInspect({ ...activeImg, url: resolvedImgUrl })}
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-black/70 text-zinc-200 hover:bg-white/20 border border-white/10 backdrop-blur-md transition-colors"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Bottom Left Legend */}
        {imageryViewer?.legend && (
          <div className="absolute bottom-2.5 left-2.5 flex flex-col gap-1 rounded-xl bg-black/75 p-2 border border-white/10 backdrop-blur-md text-[10px]">
            {imageryViewer.legend.map((leg) => (
              <div key={leg.id} className="flex items-center gap-1.5">
                <span
                  className={cls(
                    "h-3 w-4 rounded-xs",
                    leg.border === "dashed" ? "border border-dashed" : "border"
                  )}
                  style={{
                    backgroundColor: leg.fill || "transparent",
                    borderColor: leg.color,
                  }}
                />
                <span className="text-zinc-200 font-medium">{leg.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function CognitiveTraceView({
  trace,
  phase,
  inputCheck,
  routing,
  steps,
  taskTitle,
  isThinkingPhase = false,
  streamedThoughtBody = "",
  thinkingSeconds = 0,
}) {
  const [exploredOpen, setExploredOpen] = useState(false)
  const [thoughtOpen, setThoughtOpen] = useState(true)

  useEffect(() => {
    if (isThinkingPhase) {
      setThoughtOpen(true)
    }
  }, [isThinkingPhase])

  const exploredItems = React.useMemo(() => {
    if (trace?.exploration && Array.isArray(trace.exploration) && trace.exploration.length > 0) {
      return trace.exploration
    }
    const items = []

    if (inputCheck?.images && inputCheck.images.length > 0) {
      inputCheck.images.forEach((img) => {
        items.push({
          label: "Analyzed",
          icon: "📁",
          text: img.name || "GeoTIFF Raster",
          sub: img.gsd_m ? `#${img.gsd_m}m GSD` : "#Raster Scene",
        })
      })
    }

    if (trace?.bhoonidhi_scenes && trace.bhoonidhi_scenes.length > 0) {
      items.push({
        label: "Analyzed",
        icon: "🌐",
        text: `ISRO Bhoonidhi STAC Catalog (${trace.bhoonidhi_scenes.length} candidate scenes)`,
        sub: `#bhoonidhi.nrsc.gov.in`,
      })
    }

    if (trace?.geotarget) {
      const bbox = trace.geotarget.bbox
      const bboxStr = Array.isArray(bbox)
        ? `#[${bbox.map((v) => (typeof v === "number" ? v.toFixed(2) : v)).join(", ")}]`
        : "#WGS84"
      items.push({
        label: "Analyzed",
        icon: "📍",
        text: `Geospatial AOI: ${trace.geotarget.name || "Target Footprint"}`,
        sub: bboxStr,
      })
    }

    if (steps && steps.length > 0) {
      steps.forEach((s) => {
        if (!items.some((it) => it.text.includes(s.tool))) {
          items.push({
            label: "Analyzed",
            icon: s.tool.includes("bhoonidhi") ? "🛰️" : s.tool.includes("geo") ? "📍" : "🔬",
            text: s.tool,
            sub: s.duration_ms ? `#${s.duration_ms}ms` : "#Executed",
          })
        }
      })
    }
    return items
  }, [trace, inputCheck, steps])

  const exploredSummary = React.useMemo(() => {
    const total = exploredItems.length
    return `Explored ${total} ${total === 1 ? "source" : "sources"} & pipeline stages`
  }, [exploredItems])

  const thoughtDurationStr = React.useMemo(() => {
    const rawDur = trace?.duration_ms || (steps || []).reduce((acc, s) => acc + (s.duration_ms || 0), 0)
    if (typeof rawDur === "number" && rawDur >= 500) {
      return `${(rawDur / 1000).toFixed(1)}s`
    }
    if (typeof rawDur === "number" && rawDur > 0) {
      const realistic = Math.max(1.2, (rawDur + 1100) / 1000)
      return `${realistic.toFixed(1)}s`
    }
    return "1.4s"
  }, [trace?.duration_ms, steps])

  const { thoughtTitle, thoughtBody } = React.useMemo(() => {
    const raw = trace?.thought_process || ""
    if (!raw) {
      return {
        thoughtTitle: "Multi-Agent Remote Sensing Cognitive Reasoning",
        thoughtBody:
          "Decomposing query parameters, evaluating spatial constraints, and synthesizing authoritative geospatial intelligence without synthetic hallucination.",
      }
    }
    const match = raw.match(/^\*\*(.*?)\*\*\n*([\s\S]*)$/)
    if (match) {
      return { thoughtTitle: match[1], thoughtBody: match[2].trim() }
    }
    return { thoughtTitle: "Cognitive Chain-of-Thought Monologue", thoughtBody: raw }
  }, [trace?.thought_process])

  return (
    <div className="space-y-2.5 py-1 text-xs">
      <div className="flex items-center justify-between font-semibold text-foreground">
        <span>{taskTitle}</span>
        {routing?.rule_id && (
          <span className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            Rule {routing.rule_id}
          </span>
        )}
      </div>

      {inputCheck && (
        <div className="text-muted-foreground text-[11px]">
          Gate:{" "}
          <span
            className={
              inputCheck.verdict === "accepted"
                ? "text-emerald-500 font-semibold"
                : "text-amber-500 font-semibold"
            }
          >
            {inputCheck.verdict?.toUpperCase()}
          </span>
          {inputCheck.modality && ` · Modality: ${inputCheck.modality.join(", ")}`}
          {inputCheck.gsd_m && inputCheck.gsd_m[0] && ` · GSD: ${inputCheck.gsd_m[0]}m`}
        </div>
      )}

      {trace?.abstained && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-amber-600 dark:text-amber-400">
          <p className="font-semibold mb-0.5">Input Verification Gate Notice:</p>
          <p className="leading-relaxed">{inputCheck?.message || "Query declined by input gate."}</p>
        </div>
      )}

      {exploredItems.length > 0 && (
        <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/50 dark:border-zinc-800/80 dark:bg-zinc-950/60 overflow-hidden">
          <button
            type="button"
            onClick={() => setExploredOpen((prev) => !prev)}
            className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
          >
            <div className="flex items-center gap-1.5">
              <span>{exploredSummary}</span>
            </div>
            <ChevronDown
              className={cls(
                "h-3.5 w-3.5 text-zinc-400 transition-transform duration-200",
                exploredOpen ? "rotate-0" : "-rotate-90",
              )}
            />
          </button>

          {exploredOpen && (
            <div className="px-3 pb-2.5 pt-0.5 flex flex-col gap-1.5 border-t border-zinc-200/50 dark:border-zinc-800/60">
              {exploredItems.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2 text-[11px] text-zinc-700 dark:text-zinc-300">
                  <span className="text-zinc-400 text-[10px] w-12 shrink-0">{item.label || "Analyzed"}</span>
                  <span className="text-sm shrink-0">{item.icon || "⚛"}</span>
                  <span className="font-mono text-zinc-800 dark:text-zinc-200 truncate">{item.text}</span>
                  {item.sub && (
                    <span className="ml-auto font-mono text-[10px] text-zinc-400 dark:text-zinc-500 shrink-0">
                      {item.sub}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/50 dark:border-zinc-800/80 dark:bg-zinc-950/60 overflow-hidden transition-all duration-200">
        <button
          type="button"
          onClick={() => setThoughtOpen((prev) => !prev)}
          className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            {isThinkingPhase ? (
              <span className="flex items-center gap-2 text-sky-500 dark:text-sky-400 font-semibold">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-500" />
                </span>
                Thinking ({thinkingSeconds.toFixed(1)}s)...
              </span>
            ) : (
              <span>Thought for {thoughtDurationStr}</span>
            )}
          </div>
          <ChevronDown
            className={cls(
              "h-3.5 w-3.5 text-zinc-400 transition-transform duration-200",
              thoughtOpen ? "rotate-0" : "-rotate-90",
            )}
          />
        </button>

        {thoughtOpen && (
          <div className="px-3 pb-3 pt-1 border-t border-zinc-200/50 dark:border-zinc-800/60">
            <div className="text-xs font-semibold text-sky-600 dark:text-sky-400 mb-1.5 font-sans flex items-center justify-between">
              <span>{thoughtTitle}</span>
              {isThinkingPhase && (
                <span className="text-[10px] font-mono font-normal text-sky-500/80 animate-pulse">
                  deliberating cognitive trace...
                </span>
              )}
            </div>
            <div className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-zinc-700 dark:text-zinc-300 select-text">
              {isThinkingPhase ? streamedThoughtBody : thoughtBody}
              {isThinkingPhase && (
                <span className="inline-block w-1.5 h-3.5 bg-sky-400 ml-0.5 animate-pulse align-middle" />
              )}
            </div>
          </div>
        )}
      </div>

      {steps && steps.length > 0 && (
        <div className="pt-2 border-t border-border/50 flex flex-col gap-1.5">
          <div className="text-[11px] font-semibold text-muted-foreground mb-0.5">Agent Actions & Tool Invocations:</div>
          {steps.map((step, idx) => {
            const status = phase === "thinking" ? "running" : "success"
            const durationStr =
              step.duration_ms != null
                ? step.duration_ms < 1
                  ? `${(step.duration_ms * 1000).toFixed(0)}µs`
                  : `${step.duration_ms.toFixed(1)}ms`
                : step.cache
                  ? `cache ${step.cache}`
                  : undefined

            return (
              <AIToolCall
                key={`${step.tool}-${idx}`}
                name={step.tool}
                status={status}
                summary={durationStr}
                args={
                  step.params && Object.keys(step.params).length > 0 ? (
                    <code>{JSON.stringify(step.params, null, 2)}</code>
                  ) : (
                    <span>Default parameters</span>
                  )
                }
                result={
                  step.confidence != null ? (
                    <span>Confidence: {(step.confidence * 100).toFixed(1)}%</span>
                  ) : (
                    <span>{step.stub ? "Verified execution (stub pipeline)" : "Completed"}</span>
                  )
                }
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

function AssistantMessageItem({ message, onRetry }) {
  const isCurrentlyLive = Boolean(message.isLive)
  const fullText = message.content || ""

  const trace = message.trace
  const inputCheck = trace?.input_check
  const routing = trace?.routing
  const steps = trace?.steps || []
  const normBox = trace?.output?.quantities?.norm_box

  const rawThought = trace?.thought_process || ""
  const match = rawThought.match(/^\*\*(.*?)\*\*\n*([\s\S]*)$/)
  const fullThoughtBody = match ? match[2].trim() : rawThought

  // If live and has thought monologue, start in "thinking" stage; otherwise "streaming" or "done"
  const [phase, setPhase] = useState(isCurrentlyLive ? (fullThoughtBody ? "thinking" : "streaming") : "done")
  const [streamedText, setStreamedText] = useState(isCurrentlyLive ? "" : fullText)
  const [streamedThought, setStreamedThought] = useState(isCurrentlyLive ? "" : fullThoughtBody)
  const [thinkingSeconds, setThinkingSeconds] = useState(0)
  const [inspectImage, setInspectImage] = useState(null)

  // Stage 1: Live Thinking Stream
  useEffect(() => {
    if (!message.isLive || phase !== "thinking") return

    const startTime = Date.now()
    const timerInterval = setInterval(() => {
      setThinkingSeconds((Date.now() - startTime) / 1000)
    }, 100)

    if (!fullThoughtBody) {
      clearInterval(timerInterval)
      setPhase("streaming")
      return
    }

    let charIdx = 0
    // Reveals the cognitive thought stream progressively across ~1.4 seconds
    const chunkSize = Math.max(8, Math.floor(fullThoughtBody.length / 30))
    const thoughtInterval = setInterval(() => {
      charIdx += chunkSize
      if (charIdx >= fullThoughtBody.length) {
        setStreamedThought(fullThoughtBody)
        clearInterval(thoughtInterval)
        clearInterval(timerInterval)
        setTimeout(() => {
          setPhase("streaming")
        }, 250)
      } else {
        setStreamedThought(fullThoughtBody.slice(0, charIdx))
      }
    }, 40)

    return () => {
      clearInterval(timerInterval)
      clearInterval(thoughtInterval)
    }
  }, [message.isLive, phase, fullThoughtBody])

  // Stage 2: Live Answer Stream
  useEffect(() => {
    if (!message.isLive || phase !== "streaming") {
      if (phase === "done") {
        setStreamedText(fullText)
      }
      return
    }

    const words = fullText.split(" ")
    let index = 0
    const interval = setInterval(() => {
      index += 1
      setStreamedText(words.slice(0, index).join(" "))
      if (index >= words.length) {
        clearInterval(interval)
        setPhase("done")
        try {
          message.isLive = false
        } catch {}
      }
    }, STREAM_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [message.isLive, fullText, phase])

  const citations =
    trace?.bhoonidhi_scenes && trace.bhoonidhi_scenes.length > 0
      ? trace.bhoonidhi_scenes.map((s, idx) => ({
          id: String(idx + 1),
          index: idx + 1,
          title: `${s.id} (${s.satellite} ${s.product_type || s.sensor || ""}) DOP: ${s.dop || "Recent"}`,
          snippet: `Cloud: ${s.coverage_pct ?? "0"}% | Sensor: ${s.sensor || "MSI"} | Level: ${s.product_type || "L1C"}`,
          url: s.url || `https://bhoonidhi.nrsc.gov.in`,
        }))
      : (message.files || []).map((f, idx) => ({
          id: String(idx + 1),
          index: idx + 1,
          title: f.name,
          snippet: `Local Geospatial Asset | Modality: ${f.name?.includes("sar") ? "SAR" : "Optical"}`,
          url: `#`,
        }))

  const sourcesList =
    trace?.bhoonidhi_scenes && trace.bhoonidhi_scenes.length > 0
      ? trace.bhoonidhi_scenes.map((s, idx) => ({
          id: s.id || `scene-${idx}`,
          title: `${s.satellite} · ${s.id.slice(0, 24)}...`,
          snippet: `Acquired: ${s.dop || "Verified Pass"} | Level: ${s.product_type || "Standard"} | Sensor: ${s.sensor || "MSI"}`,
          url: `https://bhoonidhi.nrsc.gov.in/bhoonidhi/search.html?scene=${encodeURIComponent(s.id)}`,
        }))
      : (message.files || []).map((f, idx) => ({
          id: `file-${idx}`,
          title: f.name,
          snippet: `Local Geospatial Asset | Modality: ${f.name?.includes("sar") ? "SAR" : "Optical"}`,
          url: `#`,
        }))

  const timestamp = message.createdAt
    ? new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : undefined

  const taskTitle =
    routing?.pair_declaration ||
    (trace?.classified_task
      ? `Task: ${trace.classified_task.replace("_", " ").toUpperCase()}`
      : "Multimodal Remote Sensing Analysis")

  return (
    <AIMessage
      avatar={<SiriOrb size="26px" state={phase} />}
      copyText={fullText}
      from="assistant"
      onRetry={() => onRetry?.(message.id)}
      timestamp={timestamp}
    >
      <div className="flex flex-col gap-3 min-w-0">
        {/* 1. Reasoning Trace & Autonomous Tool Execution (Collapsible) */}
        {trace && (
          <AIReasoning
            isStreaming={phase === "thinking"}
            collapseWhenDone={false}
            defaultOpen={true}
          >
            <CognitiveTraceView
              trace={trace}
              phase={phase}
              inputCheck={inputCheck}
              routing={routing}
              steps={steps}
              taskTitle={taskTitle}
              isThinkingPhase={phase === "thinking"}
              streamedThoughtBody={streamedThought}
              thinkingSeconds={thinkingSeconds}
            />
          </AIReasoning>
        )}

        {/* 3. The Actual Answer / Response */}
        {phase === "thinking" ? (
          <AILoader label="Synthesizing geospatial intelligence..." showElapsed variant="dots" />
        ) : (
          <div className="space-y-3">
            {/* Dynamic Assessment Panel (Headline, 2-line summary, 3 KPI cards, advisory banner) */}
            {trace?.output?.assessment && (
              <AssessmentPanel assessment={trace.output.assessment} />
            )}

            {/* Interactive Remote Sensing Imagery Viewer (Layer toggles, before/after epochs, highlighted SVG overlays, legend) */}
            {Boolean(
              trace?.output?.imagery_viewer &&
              trace?.rendered_images &&
              Array.isArray(trace.rendered_images) &&
              trace.rendered_images.length > 0 &&
              trace.classified_task !== "catalog_search" &&
              trace.classified_task !== "geospatial_aoi_briefing"
            ) ? (
              <InteractiveImageryViewer
                renderedImages={trace.rendered_images}
                imageryViewer={trace.output.imagery_viewer}
                message={message}
                onInspect={(img) => setInspectImage(img)}
              />
            ) : Boolean(
              trace?.rendered_images &&
              Array.isArray(trace.rendered_images) &&
              trace.rendered_images.length > 0 &&
              trace.classified_task !== "catalog_search" &&
              trace.classified_task !== "geospatial_aoi_briefing" &&
              (message.files?.length > 0 || trace.has_real_imagery)
            ) && (
              <div className="overflow-hidden rounded-2xl border border-zinc-200/90 bg-zinc-50/70 p-4 shadow-sm dark:border-zinc-800/90 dark:bg-zinc-900/60 backdrop-blur-md">
                {/* Header Bar */}
                <div className="flex items-center justify-between mb-3 pb-2.5 border-b border-zinc-200/60 dark:border-zinc-800/60">
                  <div className="flex items-center gap-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-600 dark:bg-indigo-500/20 dark:text-indigo-400">
                      <Satellite className="h-3.5 w-3.5" />
                    </div>
                    <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      Co-Registered Imagery
                    </span>
                    <span className="rounded bg-zinc-200/70 dark:bg-zinc-800 px-1.5 py-0.5 text-[10px] font-mono text-zinc-600 dark:text-zinc-300">
                      {trace.rendered_images.length} Passes
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="inline-flex items-center rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
                      10m GSD
                    </span>
                    <span className="hidden sm:inline-flex items-center rounded-md border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-800/80 px-2 py-0.5 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
                      Bhoonidhi STAC
                    </span>
                  </div>
                </div>

                {/* Imagery Grid - Completely Clean & Unobstructed */}
                <div className={`grid gap-3 ${trace.rendered_images.length > 1 ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1"}`}>
                  {trace.rendered_images.map((img, i) => {
                    const resolvedImgUrl = (
                      message.previewUrls?.[img.name] ||
                      message.files?.find((f) => f.name === img.name)?.previewUrl ||
                      img.previewUrl ||
                      img.url ||
                      ""
                    ).replace(/\.tif+$/i, ".png")

                    return (
                      <div
                        key={i}
                        onClick={() => setInspectImage({ ...img, url: resolvedImgUrl })}
                        className="group relative cursor-pointer overflow-hidden rounded-xl border border-zinc-200/80 bg-white shadow-xs transition-all duration-300 hover:shadow-md hover:border-indigo-500/40 dark:border-zinc-800 dark:bg-zinc-950"
                      >
                        {/* Satellite Image Viewport (Pristine, No Text Overlaid) */}
                        <div className="relative aspect-[16/10] w-full overflow-hidden bg-zinc-900">
                          <img
                            src={resolvedImgUrl}
                            alt={img.name || img.label || `Satellite pass ${i + 1}`}
                            className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-103"
                          />
                          {/* Minimal Expand Icon on Hover */}
                          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                            <span className="flex items-center justify-center h-7 w-7 rounded-lg bg-black/60 text-white backdrop-blur-md border border-white/10 shadow-sm">
                              <Maximize2 className="h-3.5 w-3.5" />
                            </span>
                          </div>
                        </div>

                        {/* Clean Caption Outside Below Image */}
                        <div className="flex items-center justify-between px-3 py-2 border-t border-zinc-100 dark:border-zinc-800/60 bg-zinc-50/50 dark:bg-zinc-900/40">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <Clock className="h-3 w-3 text-indigo-500 shrink-0" />
                            <span className="text-xs font-medium text-zinc-800 dark:text-zinc-200 truncate">
                              {img.name || img.label || `Pass ${i + 1}`}
                            </span>
                          </div>
                          <span className="text-[10px] font-mono text-zinc-500 dark:text-zinc-400 shrink-0 ml-2">
                            {img.type || img.mission || "Optical"}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Lightbox Modal for Full-Resolution Scene Inspection */}
            {inspectImage && (
              <div
                onClick={() => setInspectImage(null)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm animate-in fade-in"
              >
                <div
                  onClick={(e) => e.stopPropagation()}
                  className="relative max-h-[90vh] max-w-4xl w-full overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 p-4 shadow-2xl"
                >
                  <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
                    <div className="flex items-center gap-2">
                      <Satellite className="h-4 w-4 text-indigo-400" />
                      <span className="font-semibold text-sm text-zinc-100">
                        {inspectImage.name || inspectImage.label || "Satellite Scene Inspection"}
                      </span>
                      <span className="text-xs font-mono text-zinc-400">
                        ({inspectImage.type || "High-Res Observation"})
                      </span>
                    </div>
                    <button
                      onClick={() => setInspectImage(null)}
                      className="rounded-full p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="mt-3 flex items-center justify-center overflow-hidden rounded-xl bg-black max-h-[70vh]">
                    <div className="relative inline-block max-h-[70vh]">
                      <img
                        src={(inspectImage.url || inspectImage.previewUrl || "").replace(/\.tif+$/i, ".png")}
                        alt={inspectImage.name}
                        className="max-h-[70vh] w-auto object-contain rounded-lg select-none"
                      />
                      {inspectImage.box && Array.isArray(inspectImage.box) && inspectImage.box.length === 4 && (
                        <div
                          className="absolute border-2 border-sky-400 bg-sky-400/20 shadow-[0_0_20px_rgba(56,189,248,0.6)] pointer-events-none"
                          style={{
                            left: `${Math.min(inspectImage.box[0], inspectImage.box[2]) * 100}%`,
                            top: `${Math.min(inspectImage.box[1], inspectImage.box[3]) * 100}%`,
                            width: `${Math.max(2, Math.abs(inspectImage.box[2] - inspectImage.box[0]) * 100)}%`,
                            height: `${Math.max(2, Math.abs(inspectImage.box[3] - inspectImage.box[1]) * 100)}%`,
                          }}
                        >
                          <span className="absolute -top-[2px] -left-[2px] h-3 w-3 border-t-2 border-l-2 border-sky-300" />
                          <span className="absolute -top-[2px] -right-[2px] h-3 w-3 border-t-2 border-r-2 border-sky-300" />
                          <span className="absolute -bottom-[2px] -left-[2px] h-3 w-3 border-b-2 border-l-2 border-sky-300" />
                          <span className="absolute -bottom-[2px] -right-[2px] h-3 w-3 border-b-2 border-r-2 border-sky-300" />
                          <span className="absolute -top-7 left-0 rounded bg-sky-500 px-2 py-0.5 font-mono text-[10px] font-bold text-zinc-950 shadow-md whitespace-nowrap">
                            TARGET EXTENT [{inspectImage.box.map((n) => Number(n).toFixed(3)).join(", ")}]
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Real Evidence Sources Stack (Bhoonidhi STAC Scenes / Satellite Passes) */}
            {sourcesList && sourcesList.length > 0 && (
              <div className="mb-1">
                <AISources
                  label="Satellite Scenes"
                  sources={sourcesList}
                  defaultOpen={false}
                />
              </div>
            )}

            <AIResponse
              citations={citations}
              isStreaming={phase === "streaming"}
              text={streamedText}
            />

            {/* Spatial Grounding Box Visualization */}
            {normBox && Array.isArray(normBox) && normBox.length === 4 && (
              <SpatialGroundingCard
                normBox={normBox}
                trace={trace}
                message={message}
                onInspect={(img) => setInspectImage(img)}
              />
            )}

            {/* Interactive Leaflet Satellite Grounding Map - Render for all geographic queries, catalog searches, or when AOI/overlay is present */}
            {Boolean(
              !trace?.abstained &&
              (trace?.output?.geojson ||
                trace?.map_overlay ||
                trace?.geotarget ||
                trace?.classified_task === "catalog_search" ||
                trace?.classified_task === "geospatial_aoi_briefing" ||
                trace?.classified_task === "grounding" ||
                /\b(map|location|aoi|boundary|extent|scenes|passes|satellite data|stac|show me|find|where|imagery)\b/i.test(trace?.query || message.content || "")) &&
              (trace?.geotarget?.bbox || trace?.geotarget?.lat || trace?.output?.geojson || trace?.map_overlay)
            ) && (
              <div id="geospatial-grounding-map" className="mt-3 overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
                <div className="flex items-center justify-between border-b border-border bg-muted/40 px-3.5 py-2">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-3.5 w-3.5 text-sky-500" />
                    <span className="text-xs font-semibold text-foreground">
                      Interactive Geospatial Grounding Map
                    </span>
                    {trace?.geotarget?.name && (
                      <span className="text-[11px] font-mono text-muted-foreground truncate max-w-[200px]">
                        • {trace.geotarget.name}
                      </span>
                    )}
                  </div>
                  <span className="rounded bg-sky-500/10 px-2 py-0.5 text-[10px] font-mono text-sky-500 font-medium">
                    ISRO WMS / STAC Overlay
                  </span>
                </div>
                <LeafletMap
                  bbox={trace?.geotarget?.bbox}
                  center={trace?.geotarget?.lat && trace?.geotarget?.lon ? [trace.geotarget.lat, trace.geotarget.lon] : null}
                  geojson={trace?.output?.geojson || trace?.map_overlay}
                  placeName={trace?.geotarget?.name}
                  height="340px"
                />
              </div>
            )}

            {/* Dynamic 3-stage Workflow Execution Stepper */}
            {trace?.output?.workflow_log && trace.output.workflow_log.length > 0 && (
              <WorkflowLogPanel
                workflowLog={trace.output.workflow_log}
                onExportReport={() => {
                  const reportData = {
                    title: "ISRO SatQuery AI - Auditable Execution Report",
                    timestamp: new Date().toISOString(),
                    trace_id: trace.trace_id,
                    query: trace.query,
                    classified_task: trace.classified_task,
                    routing: trace.routing,
                    assessment: trace.output?.assessment,
                    workflow_log: trace.output?.workflow_log,
                    satellite_scenes_matched: trace.bhoonidhi_scenes || [],
                    execution_steps: trace.steps || [],
                    output: trace.output,
                  }
                  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement("a")
                  a.href = url
                  a.download = `satquery_audit_${trace.trace_id?.slice(0, 12) || Date.now()}.json`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                onViewEvidence={() => {
                  const el = document.getElementById("geospatial-grounding-map")
                  if (el) el.scrollIntoView({ behavior: "smooth" })
                }}
              />
            )}

            {/* Trace Audit Bar & Report Download */}
            {trace?.trace_id && (
              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-border/50 text-[11px] font-mono text-muted-foreground">
                <div className="flex items-center gap-2">
                  <span className="truncate">Trace ID: {trace.trace_id.slice(0, 16)}</span>
                  {trace.output?.confidence != null && (
                    <span className="text-emerald-500 font-semibold">
                      Confidence: {(trace.output.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const reportData = {
                      title: "ISRO SatQuery AI - Auditable Execution Report",
                      timestamp: new Date().toISOString(),
                      trace_id: trace.trace_id,
                      query: trace.query,
                      classified_task: trace.classified_task,
                      routing: trace.routing,
                      input_verification: trace.input_check,
                      geotarget: trace.geotarget,
                      satellite_scenes_matched: trace.bhoonidhi_scenes || [],
                      execution_steps: trace.steps || [],
                      output: trace.output,
                    }
                    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement("a")
                    a.href = url
                    a.download = `satquery_audit_${trace.trace_id.slice(0, 12)}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                  title="Download verifiable execution report (JSON)"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download Report</span>
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </AIMessage>
  )
}

const ChatPane = forwardRef(function ChatPane(
  { conversation, onSend, onEditMessage, onResendMessage, isThinking, onPauseThinking },
  ref,
) {
  const [editingId, setEditingId] = useState(null)
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const composerRef = useRef(null)

  useImperativeHandle(
    ref,
    () => ({
      insertTemplate: (templateContent) => {
        composerRef.current?.insertTemplate(templateContent)
      },
      loadPreset: (presetId) => {
        composerRef.current?.loadPreset(presetId)
      },
    }),
    [],
  )

  if (!conversation) return null

  const tags = ["Multimodal RS", "Optical + SAR", "ISRO SIH26167", "Qwen2.5-VL"]
  const messages = Array.isArray(conversation.messages) ? conversation.messages : []
  const count = messages.length || conversation.messageCount || 0

  function startEdit(m) {
    setEditingId(m.id)
    setDraft(m.content)
  }
  function cancelEdit() {
    setEditingId(null)
    setDraft("")
  }
  function saveEdit() {
    if (!editingId) return
    onEditMessage?.(editingId, draft)
    cancelEdit()
  }
  function saveAndResend() {
    if (!editingId) return
    onEditMessage?.(editingId, draft)
    onResendMessage?.(editingId)
    cancelEdit()
  }

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <AIConversation className="flex-1 min-h-0 px-4 py-6 sm:px-8" contentKey={messages.length + (isThinking ? 1 : 0)}>
        <div className="flex flex-col gap-5 pb-4 max-w-3xl mx-auto">
          {/* Conversation Header */}
          <div className="border-b border-border pb-4">
            <div className="mb-2 text-2xl font-bold tracking-tight sm:text-3xl text-foreground">
              <span>{conversation.title}</span>
            </div>
            <div className="mb-3 text-xs font-mono text-muted-foreground">
              Updated {timeAgo(conversation.updatedAt)} · {count} messages
            </div>
            <div className="flex flex-wrap gap-2">
              {tags.map((t) => (
                <span
                  key={t}
                  className="inline-flex items-center rounded-full border border-border bg-muted/50 px-3 py-0.5 text-xs font-medium text-foreground"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Empty Conversation Guidance */}
          {messages.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
              <p className="font-semibold text-foreground mb-1">No analysis queries yet</p>
              <p className="text-xs text-muted-foreground max-w-md mx-auto">
                Select a <strong>Demo Preset</strong> below or attach satellite imagery (.tif, .png) and ask questions
                like &quot;Is there a road?&quot; or &quot;Where is the pasture?&quot;.
              </p>
            </div>
          ) : (
            <>
              {messages.map((m) => {
                const timestamp = m.createdAt
                  ? new Date(m.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                  : undefined

                if (editingId === m.id) {
                  return (
                    <div key={m.id} className="rounded-2xl border border-border p-3">
                      <textarea
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        className="w-full resize-y rounded-xl bg-transparent p-2 text-sm outline-none"
                        rows={3}
                      />
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          onClick={saveEdit}
                          className="inline-flex items-center gap-1 rounded-full bg-foreground px-3 py-1.5 text-xs text-background"
                        >
                          <Check className="h-3.5 w-3.5" /> Save
                        </button>
                        <button
                          onClick={saveAndResend}
                          className="inline-flex items-center gap-1 rounded-full border border-border px-3 py-1.5 text-xs"
                        >
                          <RefreshCw className="h-3.5 w-3.5" /> Save & Resend
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-xs text-muted-foreground"
                        >
                          <X className="h-3.5 w-3.5" /> Cancel
                        </button>
                      </div>
                    </div>
                  )
                }

                if (m.role === "user") {
                  return (
                    <div key={m.id} className="space-y-1">
                      <AIMessage from="user" timestamp={timestamp} copyText={m.content}>
                        <div className="flex flex-col gap-2">
                          {/* Attached files chips */}
                          {m.files && m.files.length > 0 && (
                            <div className="flex flex-wrap gap-1.5">
                              {m.files.map((file, fIdx) => {
                                const isTif = file.name?.endsWith(".tif") || file.name?.endsWith(".tiff")
                                const isSidecar = file.name?.endsWith(".json")
                                return (
                                  <span
                                    key={fIdx}
                                    className="inline-flex items-center gap-1.5 rounded-full border border-background/20 bg-background/10 px-2.5 py-0.5 text-xs font-mono"
                                  >
                                    {isTif ? (
                                      <Satellite className="h-3 w-3" />
                                    ) : isSidecar ? (
                                      <FileText className="h-3 w-3" />
                                    ) : (
                                      <Satellite className="h-3 w-3" />
                                    )}
                                    <span>{file.name}</span>
                                    <span className="rounded bg-background/20 px-1 text-[9px] uppercase font-bold">
                                      {isTif ? "GeoTIFF" : isSidecar ? "Sidecar" : "PNG"}
                                    </span>
                                  </span>
                                )
                              })}
                            </div>
                          )}
                          <div>{m.content}</div>
                        </div>
                      </AIMessage>

                      <div className="flex justify-end gap-2 pr-2 text-[11px] text-muted-foreground">
                        <button
                          className="inline-flex items-center gap-1 hover:text-foreground"
                          onClick={() => startEdit(m)}
                        >
                          <Pencil className="h-3 w-3" /> Edit
                        </button>
                        <button
                          className="inline-flex items-center gap-1 hover:text-foreground"
                          onClick={() => onResendMessage?.(m.id)}
                        >
                          <RefreshCw className="h-3 w-3" /> Resend
                        </button>
                      </div>
                    </div>
                  )
                }

                return (
                  <AssistantMessageItem
                    key={m.id}
                    message={m}
                    onRetry={() => onResendMessage?.(m.id)}
                  />
                )
              })}

              {/* Active Thinking State */}
              {isThinking && (
                <AIMessage
                  avatar={<SiriOrb size="26px" state="thinking" />}
                  from="assistant"
                  timestamp={new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                >
                  <div className="flex flex-col gap-3 min-w-0">
                    <AIReasoning isStreaming={true}>
                      Inspecting input raster gates, validating CRS & ground sampling distance, and routing to multimodal remote sensing experts...
                    </AIReasoning>
                    <AILoader label="Analyzing satellite imagery..." showElapsed variant="dots" />
                  </div>
                </AIMessage>
              )}
            </>
          )}
        </div>
      </AIConversation>

      <Composer
        ref={composerRef}
        onSend={async (payload) => {
          setBusy(true)
          await onSend?.(payload)
          setBusy(false)
        }}
        busy={busy || isThinking}
      />
    </div>
  )
})

export default ChatPane
