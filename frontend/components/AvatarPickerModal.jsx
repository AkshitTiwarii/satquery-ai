"use client"

import React, { useState } from "react"
import { Check, X } from "lucide-react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import { cn } from "@/lib/utils"
import { AVATARS, AVATAR_RGB } from "./UserAvatar"

const containerVariants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.05 },
  },
}

const thumbnailVariants = {
  initial: { opacity: 0, y: 6 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.28, ease: "easeOut" },
  },
}

export default function AvatarPickerModal({
  isOpen,
  onClose,
  currentAvatarId = 1,
  onSelectAvatar,
}) {
  const [selectedId, setSelectedId] = useState(Number(currentAvatarId) || 1)
  const shouldReduceMotion = useReducedMotion()

  if (!isOpen) return null

  const selectedAvatar = AVATARS.find((a) => a.id === selectedId) || AVATARS[0]
  const rgb = AVATAR_RGB[selectedAvatar.id] || "255, 0, 91"

  const handleConfirm = () => {
    onSelectAvatar?.(selectedId)
    onClose?.()
  }

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm"
        />

        {/* Modal Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          transition={{ duration: 0.2 }}
          className="relative mx-auto w-full max-w-[380px] rounded-2xl border border-zinc-200/80 bg-white p-6 shadow-2xl backdrop-blur-xl dark:border-zinc-800 dark:bg-zinc-950 z-10"
        >
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute right-4 top-4 rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-900 dark:hover:text-zinc-200 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>

          {/* Header */}
          <div className="space-y-1 text-center mb-6">
            <h2 className="font-bold text-lg text-zinc-900 dark:text-zinc-100 tracking-tight">
              Pick Your Avatar
            </h2>
            <p className="text-zinc-500 dark:text-zinc-400 text-xs">
              Choose an avatar for your SatQuery profile
            </p>
          </div>

          {/* Avatar Stage */}
          <div className="flex flex-col items-center gap-4">
            <div className="relative h-32 w-32">
              {/* Animated per-avatar color ring */}
              <motion.div
                animate={{
                  boxShadow: `0 0 0 2px rgba(${rgb}, 0.55), 0 6px 24px rgba(${rgb}, 0.22)`,
                }}
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 rounded-full"
                transition={
                  shouldReduceMotion
                    ? { duration: 0 }
                    : { duration: 0.45, ease: "easeOut" }
                }
              />

              {/* Avatar circle */}
              <div className="relative h-full w-full overflow-hidden rounded-full border border-white/20">
                <AnimatePresence mode="wait">
                  <motion.div
                    animate={{ opacity: 1 }}
                    className="absolute inset-0 flex items-center justify-center"
                    exit={{ opacity: 0 }}
                    initial={{ opacity: 0 }}
                    key={selectedAvatar.id}
                    transition={
                      shouldReduceMotion
                        ? { duration: 0 }
                        : { duration: 0.2, ease: "easeOut" }
                    }
                  >
                    <div className="scale-[3.5] transform">
                      {selectedAvatar.renderSvg(36)}
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Avatar label */}
            <AnimatePresence mode="wait">
              <motion.span
                animate={{ opacity: 1 }}
                className="text-[10px] font-mono text-zinc-500 dark:text-zinc-400 uppercase tracking-widest"
                exit={{ opacity: 0 }}
                initial={{ opacity: 0 }}
                key={selectedAvatar.id}
                transition={
                  shouldReduceMotion
                    ? { duration: 0 }
                    : { duration: 0.16, ease: "easeOut" }
                }
              >
                {selectedAvatar.alt}
              </motion.span>
            </AnimatePresence>

            {/* Thumbnail strip */}
            <motion.div
              animate="animate"
              className="flex gap-2.5 my-2"
              initial="initial"
              variants={containerVariants}
            >
              {AVATARS.map((avatar) => {
                const isSelected = selectedAvatar.id === avatar.id
                return (
                  <motion.button
                    key={avatar.id}
                    aria-label={`Select ${avatar.alt}`}
                    aria-pressed={isSelected}
                    onClick={() => setSelectedId(avatar.id)}
                    className={cn(
                      "relative h-12 w-12 overflow-hidden rounded-xl border transition-all duration-200",
                      isSelected
                        ? "border-indigo-500 ring-2 ring-indigo-500 ring-offset-2 ring-offset-white dark:ring-offset-black opacity-100 scale-105"
                        : "border-zinc-200 dark:border-zinc-800 opacity-60 hover:opacity-100"
                    )}
                    type="button"
                    variants={thumbnailVariants}
                    whileHover={shouldReduceMotion ? {} : { scale: 1.06 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.94 }}
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="scale-[1.8] transform">{avatar.renderSvg(36)}</div>
                    </div>
                    {isSelected && (
                      <div className="absolute -right-0.5 -bottom-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-white shadow-xs">
                        <Check className="h-2.5 w-2.5" />
                      </div>
                    )}
                  </motion.button>
                )
              })}
            </motion.div>
          </div>

          {/* Action Button */}
          <button
            type="button"
            onClick={handleConfirm}
            className="mt-6 flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-indigo-600 to-sky-600 py-2.5 text-xs font-semibold text-white shadow-md shadow-indigo-600/25 hover:from-indigo-500 hover:to-sky-500 transition-all"
          >
            Save Avatar
          </button>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
