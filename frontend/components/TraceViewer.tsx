"use client"

import React, { useState } from "react"
import {
  CheckCircle2,
  AlertTriangle,
  Layers,
  Clock,
  Maximize2,
  MapPin,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Cpu,
  ShieldCheck,
  Compass,
  FileCheck2
} from "lucide-react"
import { Trace } from "@/lib/satquery"

interface TraceViewerProps {
  trace: Trace
  previewUrls?: Record<string, string>
}

export default function TraceViewer({ trace, previewUrls }: TraceViewerProps) {
  const [expanded, setExpanded] = useState(false)
  const [copiedAudit, setCopiedAudit] = useState(false)

  const isRefusal = trace.abstained
  const normBox = trace.output?.quantities?.norm_box

  const taskTitles: Record<string, string> = {
    vqa: "Visual Question Answering",
    grounding: "Spatial Feature Grounding",
    change_vqa: "Bi-Temporal Change Analysis",
    fusion: "Optical + SAR Microwave Fusion",
    captioning: "Remote Sensing Scene Captioning",
  }

  const ruleDescriptions: Record<string, string> = {
    R0: "Refusal Gate",
    R1: "Two images (Optical + SAR) -> Fusion",
    R2: "SAR bi-temporal pair -> SAR Log-Ratio Change",
    R3: "Optical bi-temporal pair -> Change Detection",
    R4: "Single image + spatial phrase -> Grounding",
    R5: "Single image + summary phrase -> Captioning",
    R6: "Single image general reasoning -> VQA",
    S1: "Sub-metre grounding -> Base Model Weights",
  }

  function handleCopyAudit() {
    const text = `Trace ID: ${trace.trace_id}\nSeed: ${trace.replay?.seed}\nRule: ${trace.routing?.rule_id}\nLock: ${trace.replay?.registry_lock}`
    navigator.clipboard.writeText(text)
    setCopiedAudit(true)
    setTimeout(() => setCopiedAudit(false), 2000)
  }

  const totalDurationMs = trace.steps?.reduce((acc, s) => acc + (s.duration_ms || 0), 0) || 0

  return (
    <div className="w-full space-y-3">
      {/* 1. Primary Answer Banner */}
      <div className="rounded-2xl border border-zinc-200/90 bg-gradient-to-b from-white/90 to-zinc-50/90 p-4 shadow-sm backdrop-blur-md dark:border-zinc-800/90 dark:from-zinc-900/90 dark:to-zinc-950/90">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-100 pb-3 dark:border-zinc-800">
          <div className="flex items-center gap-2">
            {isRefusal ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-600 dark:bg-amber-500/20 dark:text-amber-400">
                <AlertTriangle className="h-3.5 w-3.5" />
                Abstained · Gate {trace.input_check?.code || "Refusal"}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-400">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {taskTitles[trace.classified_task || ""] || trace.classified_task?.toUpperCase() || "Analysis Complete"}
              </span>
            )}

            <span className="rounded-md border border-zinc-200 bg-zinc-50 px-2 py-0.5 font-mono text-[11px] text-zinc-600 dark:border-zinc-800 dark:bg-zinc-800/60 dark:text-zinc-400">
              {trace.routing?.rule_id || "R6"} ({ruleDescriptions[trace.routing?.rule_id || ""] || "Rule"})
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono text-zinc-500 dark:text-zinc-400">
            {/* Confidence: Show ONLY when present in trace */}
            {trace.output?.confidence !== null && trace.output?.confidence !== undefined && (
              <div className="flex items-center gap-1.5">
                <span className="text-zinc-400">Confidence:</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                  {(trace.output.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}

            <div className="flex items-center gap-1 text-zinc-400">
              <Clock className="h-3.5 w-3.5" />
              <span>
                {totalDurationMs < 1000 ? `${totalDurationMs.toFixed(1)}ms` : `${(totalDurationMs / 1000).toFixed(2)}s`}
              </span>
            </div>
          </div>
        </div>

        {/* Answer Content */}
        <div className="pt-3">
          {isRefusal ? (
            <div className="rounded-xl border border-amber-200/80 bg-amber-50/60 p-3 text-xs text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
              <p className="font-semibold mb-1">Gate Verification Reason:</p>
              <p className="leading-relaxed">{trace.output?.text || trace.input_check?.message}</p>
            </div>
          ) : (
            <div className="text-base font-semibold tracking-tight text-zinc-900 dark:text-white leading-relaxed">
              {trace.output?.text}
            </div>
          )}

          {/* Spatial Grounding Box Visualization */}
          {normBox && Array.isArray(normBox) && normBox.length === 4 && (
            <div className="mt-3 rounded-xl border border-indigo-200/80 bg-indigo-50/50 p-3.5 dark:border-indigo-900/50 dark:bg-indigo-950/20">
              <div className="flex items-center justify-between text-xs font-medium text-indigo-700 dark:text-indigo-300 mb-2">
                <span className="inline-flex items-center gap-1.5 font-semibold">
                  <Maximize2 className="h-3.5 w-3.5" />
                  Grounding Bounding Box
                </span>
                <span className="font-mono text-[11px] bg-white/80 dark:bg-zinc-900/80 px-2 py-0.5 rounded border border-indigo-200 dark:border-indigo-800">
                  [xmin, ymin, xmax, ymax]: [{normBox.map((n) => n.toFixed(3)).join(", ")}]
                </span>
              </div>

              {/* Graphical Box Overlay */}
              <div className="relative mx-auto mt-2 h-44 w-full max-w-md overflow-hidden rounded-xl border border-indigo-200 bg-zinc-950 dark:border-indigo-900 shadow-inner">
                {/* Visual Satellite Grid */}
                <div className="absolute inset-0 bg-[radial-gradient(#312e81_1px,transparent_1px)] [background-size:16px_16px] opacity-30"></div>

                {(() => {
                  const rawUrl =
                    (previewUrls && Object.values(previewUrls)[0]) ||
                    ((trace as any)?.rendered_images?.[0]?.url) ||
                    "/fixtures/lr_232.png"
                  const cleanUrl = rawUrl.replace(/\.tif+$/i, ".png")
                  return (
                    <img
                      src={cleanUrl}
                      alt="Satellite observation raster"
                      className="h-full w-full object-cover opacity-85 select-none"
                    />
                  )
                })()}

                {/* The Extracted Bounding Box */}
                <div
                  className="absolute border-2 border-emerald-400 bg-emerald-400/20 shadow-lg shadow-emerald-500/20 transition-all backdrop-blur-[1px]"
                  style={{
                    left: `${Math.min(normBox[0], normBox[2]) * 100}%`,
                    top: `${Math.min(normBox[1], normBox[3]) * 100}%`,
                    width: `${Math.max(0.08, Math.abs(normBox[2] - normBox[0])) * 100}%`,
                    height: `${Math.max(0.08, Math.abs(normBox[3] - normBox[1])) * 100}%`,
                  }}
                >
                  <span className="absolute -top-5 left-0 rounded bg-emerald-500 px-1.5 py-0.2 font-mono text-[9px] font-bold text-zinc-950">
                    Target
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* GeoJSON Feature Data if present */}
          {trace.output?.geojson && (
            <div className="mt-3 rounded-xl border border-zinc-200 bg-zinc-50 p-2.5 dark:border-zinc-800 dark:bg-zinc-900/50">
              <div className="flex items-center gap-1.5 font-medium text-zinc-700 dark:text-zinc-300 mb-1 text-xs">
                <MapPin className="h-3.5 w-3.5 text-blue-500" />
                <span>GeoJSON Vector Feature</span>
              </div>
              <pre className="max-h-20 overflow-auto rounded bg-zinc-100 p-2 font-mono text-[10px] text-zinc-700 dark:bg-zinc-950 dark:text-zinc-400">
                {trace.output.geojson}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* 2. Collapsible Execution & Verification Trace */}
      <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/50 dark:border-zinc-800/80 dark:bg-zinc-900/40 overflow-hidden">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex w-full items-center justify-between px-3.5 py-2 text-xs font-medium text-zinc-600 hover:text-zinc-950 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
        >
          <span className="flex items-center gap-2">
            <Layers className="h-3.5 w-3.5 text-indigo-500" />
            <span>Inspect Verification & Tool Pipeline ({trace.steps?.length || 0} tools executed)</span>
          </span>
          {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        </button>

        {expanded && (
          <div className="space-y-3.5 border-t border-zinc-200/70 p-3.5 dark:border-zinc-800 text-xs">
            {/* Input Gate Summary */}
            <div>
              <div className="flex items-center gap-1.5 font-semibold text-zinc-700 dark:text-zinc-300 mb-1.5">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
                <span>Input Verification Gate:</span>
              </div>
              <div className="space-y-1">
                {trace.input_check?.images?.map((img, i) => (
                  <div
                    key={i}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-white p-2 border border-zinc-200/80 dark:bg-zinc-950/60 dark:border-zinc-800/80"
                  >
                    <div className="flex items-center gap-1.5 font-mono text-[11px]">
                      <span className="rounded bg-indigo-500/10 px-1.5 py-0.2 text-indigo-600 dark:text-indigo-400 font-semibold">
                        {img.format || "TIFF"}
                      </span>
                      <span className="font-semibold text-zinc-900 dark:text-zinc-200">{img.path}</span>
                    </div>
                    <div className="flex items-center gap-2.5 font-mono text-[10px] text-zinc-500">
                      <span>Modality: <strong className="text-zinc-700 dark:text-zinc-300">{img.modality}</strong></span>
                      {img.gsd_m && <span>GSD: {img.gsd_m.toFixed(1)}m</span>}
                      {img.size_px && <span>{img.size_px[0]}×{img.size_px[1]}px</span>}
                      {img.acquired_at && <span>{new Date(img.acquired_at).toLocaleDateString()}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Step-by-step Tool Execution Pipeline */}
            <div>
              <div className="flex items-center gap-1.5 font-semibold text-zinc-700 dark:text-zinc-300 mb-1.5">
                <Cpu className="h-3.5 w-3.5 text-blue-500" />
                <span>Execution Timeline:</span>
              </div>
              <div className="space-y-1">
                {trace.steps?.map((step, sIdx) => (
                  <div
                    key={sIdx}
                    className="flex items-center justify-between rounded-lg bg-white px-3 py-1.5 border border-zinc-200/80 dark:bg-zinc-950/60 dark:border-zinc-800/80 text-[11px]"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-400 font-mono text-[10px]">#{sIdx + 1}</span>
                      <span className="font-mono font-medium text-zinc-800 dark:text-zinc-200">
                        {step.tool}
                      </span>
                      {step.stub && (
                        <span className="rounded-full bg-amber-500/10 px-1.5 py-0.2 text-[9px] font-medium text-amber-600 dark:text-amber-400">
                          wired, not live
                        </span>
                      )}
                    </div>
                    <div className="font-mono text-zinc-500">
                      {step.duration_ms < 1
                        ? `${(step.duration_ms * 1000).toFixed(0)}µs`
                        : `${step.duration_ms.toFixed(2)}ms`}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Audit Line */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-zinc-200/60 pt-2 font-mono text-[10px] text-zinc-400 dark:border-zinc-800">
              <div>
                <span>Trace ID: {trace.trace_id}</span> · <span>Seed: {trace.replay?.seed}</span>
              </div>
              <button
                type="button"
                onClick={handleCopyAudit}
                className="inline-flex items-center gap-1 rounded px-2 py-0.5 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-colors text-zinc-500 dark:text-zinc-400"
              >
                {copiedAudit ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
                {copiedAudit ? "Copied" : "Copy Audit"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
