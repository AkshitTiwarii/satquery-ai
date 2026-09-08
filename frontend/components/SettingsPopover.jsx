"use client"
import { useState } from "react"
import { Globe, HelpCircle, Crown, BookOpen, LogOut, ChevronRight, Settings, Trash2, Database, LogIn, Sparkles } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import UserAvatar from "./UserAvatar"

export default function SettingsPopover({
  children,
  currentUser,
  onLogout,
  onOpenAuth,
  onOpenAvatarPicker,
  onClearAllConversations,
}) {
  const [open, setOpen] = useState(false)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent className="w-72 p-0" align="start" side="top">
        <div className="p-3">
          {/* User Email Heading */}
          <div className="text-xs font-mono text-zinc-500 dark:text-zinc-400 mb-2 truncate">
            {currentUser?.email || "Guest Session (Local)"}
          </div>

          {/* User Profile Card */}
          <div className="flex items-center gap-3 p-2.5 rounded-xl bg-zinc-100/80 dark:bg-zinc-800/60 mb-2 border border-zinc-200/50 dark:border-zinc-700/50">
            <div
              onClick={() => {
                if (currentUser && onOpenAvatarPicker) {
                  setOpen(false)
                  onOpenAvatarPicker()
                }
              }}
              className="cursor-pointer hover:opacity-85 transition-opacity"
              title="Click to change avatar"
            >
              <UserAvatar avatarId={currentUser?.avatarId || 1} name={currentUser?.name} size={36} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 truncate">
                {currentUser?.name || "ISRO Guest Scientist"}
              </div>
              <div className="flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
                <Database className="h-3 w-3" />
                <span>{currentUser ? "Neon PostgreSQL Active" : "Local Browser Storage"}</span>
              </div>
            </div>
            {currentUser && (
              <div className="text-emerald-500">
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            )}
          </div>

          {currentUser && onOpenAvatarPicker && (
            <button
              type="button"
              onClick={() => {
                setOpen(false)
                onOpenAvatarPicker()
              }}
              className="mb-2.5 flex items-center justify-center gap-1.5 w-full py-1.5 px-2 rounded-lg border border-indigo-500/30 bg-indigo-50/50 hover:bg-indigo-100/70 dark:border-indigo-500/40 dark:bg-indigo-950/30 dark:hover:bg-indigo-900/50 text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 transition-colors"
            >
              <Sparkles className="h-3 w-3" />
              <span>Change Profile Avatar</span>
            </button>
          )}

          <div className="space-y-0.5">
            <button className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors text-zinc-700 dark:text-zinc-200">
              <Settings className="h-4 w-4 text-zinc-500" />
              <span>Settings</span>
            </button>

            <button className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors text-zinc-700 dark:text-zinc-200">
              <Globe className="h-4 w-4 text-zinc-500" />
              <span>Language</span>
              <ChevronRight className="h-4 w-4 ml-auto text-zinc-400" />
            </button>

            <button className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors text-zinc-700 dark:text-zinc-200">
              <HelpCircle className="h-4 w-4 text-zinc-500" />
              <span>Get help</span>
            </button>
          </div>

          <div className="my-2 border-t border-zinc-200/80 dark:border-zinc-800" />

          <div className="space-y-0.5">
            <button className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors text-zinc-700 dark:text-zinc-200">
              <Crown className="h-4 w-4 text-amber-500" />
              <span>SIH 2026 Enterprise</span>
            </button>

            <button className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors text-zinc-700 dark:text-zinc-200">
              <BookOpen className="h-4 w-4 text-zinc-500" />
              <span>ISRO Bhoonidhi Docs</span>
              <ChevronRight className="h-4 w-4 ml-auto text-zinc-400" />
            </button>
          </div>

          <div className="my-2 border-t border-zinc-200/80 dark:border-zinc-800" />

          {onClearAllConversations && (
            <button
              onClick={() => {
                onClearAllConversations()
                setOpen(false)
              }}
              className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-rose-50 dark:hover:bg-rose-950/30 rounded-lg transition-colors text-rose-600 dark:text-rose-400"
            >
              <Trash2 className="h-4 w-4" />
              <span>Clear chat cache</span>
            </button>
          )}

          {currentUser ? (
            <button
              onClick={() => {
                setOpen(false)
                onLogout?.()
              }}
              className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-rose-50 dark:hover:bg-rose-950/30 rounded-lg transition-colors text-rose-600 dark:text-rose-400"
            >
              <LogOut className="h-4 w-4" />
              <span>Log out</span>
            </button>
          ) : (
            <button
              onClick={() => {
                setOpen(false)
                onOpenAuth?.()
              }}
              className="flex items-center gap-3 w-full px-3 py-2 text-xs font-medium text-left hover:bg-indigo-50 dark:hover:bg-indigo-950/30 rounded-lg transition-colors text-indigo-600 dark:text-indigo-400 font-semibold"
            >
              <LogIn className="h-4 w-4" />
              <span>Sign In / Register</span>
            </button>
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
