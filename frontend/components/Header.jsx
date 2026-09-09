"use client"
import { useState } from "react"
import {
  MoreHorizontal,
  Menu,
  ChevronDown,
  Satellite
} from "lucide-react"
import GhostIconButton from "./GhostIconButton"
import UserAvatar from "./UserAvatar"
export default function Header({
  sidebarCollapsed,
  setSidebarOpen,
  backendHealth,
  onRefreshHealth,
  currentUser,
  onOpenAuth,
  onOpenAvatarPicker,
  onLogout,
}) {
  const [selectedBot, setSelectedBot] = useState("SatQuery-VL")
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)

  const models = [
    { name: "SatQuery-VL", desc: "Joint Optical, SAR & MSI Vision-Language Engine", icon: "🛰️" },
    { name: "DOFA Multispectral", desc: "12-Band Cross-Modal Remote Sensing", icon: "🌐" },
  ]

  const isOnline = backendHealth?.ok
  const isGpuActive = backendHealth?.mode === "remote_gpu_active" || backendHealth?.backend === "real" || Boolean(backendHealth?.model_available)

  const userInitials = currentUser?.name
    ? currentUser.name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
    : "G"

  return (
    <div className="shrink-0 sticky top-0 z-30 flex items-center gap-2 border-b border-zinc-200/80 bg-white/80 px-4 py-2.5 backdrop-blur dark:border-zinc-800/80 dark:bg-zinc-950/80">
      <button
        onClick={() => setSidebarOpen?.(true)}
        className="md:hidden inline-flex items-center justify-center rounded-lg p-2 hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:hover:bg-zinc-800 text-zinc-600 dark:text-zinc-300"
        aria-label="Open sidebar"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Model Selection Dropdown */}
      <div className="hidden md:flex relative">
        <button
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          className="inline-flex items-center gap-2 rounded-full border border-zinc-200/90 bg-white px-3 py-1.5 text-xs font-semibold tracking-tight hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800 transition-all shadow-sm"
        >
          <Satellite className="h-4 w-4 text-indigo-500" />
          <span>{selectedBot}</span>
          <ChevronDown className="h-3.5 w-3.5 text-zinc-400" />
        </button>

        {isDropdownOpen && (
          <div className="absolute top-full left-0 mt-1.5 w-64 rounded-xl border border-zinc-200 bg-white p-1.5 shadow-xl backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950 z-50 animate-in fade-in">
            {models.map((m) => (
              <button
                key={m.name}
                onClick={() => {
                  setSelectedBot(m.name)
                  setIsDropdownOpen(false)
                }}
                className="w-full flex items-start gap-2.5 px-3 py-2 text-xs text-left hover:bg-zinc-100 dark:hover:bg-zinc-800/80 rounded-lg transition-colors"
              >
                <span className="text-base">{m.icon}</span>
                <div>
                  <div className="font-semibold text-zinc-900 dark:text-zinc-100">{m.name}</div>
                  <div className="text-[10px] text-zinc-400">{m.desc}</div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Clean Engine Status Indicator */}
      <div className="flex items-center gap-2 rounded-full border border-zinc-200/90 bg-zinc-50/80 px-3 py-1.5 text-xs font-medium text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/80 dark:text-zinc-300 shadow-sm">
        <span
          className={`h-2 w-2 rounded-full ${
            isOnline ? "bg-emerald-500" : "bg-rose-500"
          }`}
        />
        <span className="font-semibold tracking-tight">
          {isOnline ? "SatQuery Engine Active" : "Engine Offline"}
        </span>
        {isOnline && (
          <span className="text-[10px] text-zinc-400 font-mono hidden sm:inline">
            {isGpuActive ? "• GPU-Accelerated" : "• Standard Fallback"}
          </span>
        )}
      </div>


      <div className="ml-auto flex items-center gap-2">
        {/* User Auth Section */}
        {currentUser ? (
          <div className="relative">
            <button
              onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              className="flex items-center gap-2 rounded-full border border-zinc-200/80 bg-zinc-50/90 pl-1 pr-3 py-1 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900/90 dark:hover:bg-zinc-800 transition-all shadow-sm"
            >
              <UserAvatar avatarId={currentUser.avatarId || 1} name={currentUser.name} size={24} />
              <span className="max-w-[110px] truncate text-zinc-800 dark:text-zinc-200 font-medium">
                {currentUser.name || currentUser.email}
              </span>
              <ChevronDown className="h-3 w-3 text-zinc-400" />
            </button>

            {isUserMenuOpen && (
              <div className="absolute right-0 top-full mt-1.5 w-60 rounded-xl border border-zinc-200 bg-white p-2 shadow-xl backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950 z-50 animate-in fade-in">
                <div className="flex items-center gap-2.5 p-2 border-b border-zinc-100 dark:border-zinc-800/80 mb-1">
                  <UserAvatar avatarId={currentUser.avatarId || 1} name={currentUser.name} size={36} />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 truncate">
                      {currentUser.name}
                    </div>
                    <div className="text-[10px] text-zinc-400 truncate">
                      {currentUser.email}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      <span>Neon DB Active</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => {
                    setIsUserMenuOpen(false)
                    onOpenAvatarPicker?.()
                  }}
                  className="w-full flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-indigo-600 hover:bg-indigo-50 dark:text-indigo-400 dark:hover:bg-indigo-950/30 transition-colors"
                >
                  Change Avatar
                </button>

                <button
                  onClick={() => {
                    setIsUserMenuOpen(false)
                    onLogout?.()
                  }}
                  className="w-full flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-rose-600 hover:bg-rose-50 dark:text-rose-400 dark:hover:bg-rose-950/30 transition-colors"
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={onOpenAuth}
            className="inline-flex items-center gap-1.5 rounded-full border border-indigo-500/30 bg-indigo-50/80 px-3.5 py-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100 dark:border-indigo-500/40 dark:bg-indigo-950/40 dark:text-indigo-300 dark:hover:bg-indigo-900/60 transition-all shadow-sm"
          >
            <span>Sign In</span>
          </button>
        )}

        <GhostIconButton label="More">
          <MoreHorizontal className="h-4 w-4" />
        </GhostIconButton>
      </div>
    </div>
  )
}
