"use client"
import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, Loader2, CheckCircle2, Satellite } from "lucide-react"
import { Label } from "./ui/label"
import { Input } from "./ui/input"
import { cn } from "@/lib/utils"
import UserAvatar, { AVATARS } from "./UserAvatar"
import {
  IconBrandGithub,
  IconBrandGoogle,
} from "@tabler/icons-react"

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [mode, setMode] = useState("login") // 'login' | 'register'
  const [firstName, setFirstName] = useState("")
  const [lastName, setLastName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [avatarId, setAvatarId] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [successMsg, setSuccessMsg] = useState("")

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setSuccessMsg("")

    if (!email.trim() || !password) {
      setError("Please fill in all required fields.")
      return
    }

    if (mode === "register" && password.length < 6) {
      setError("Password must be at least 6 characters.")
      return
    }

    setLoading(true)

    try {
      const endpoint = mode === "register" ? "/api/auth/register" : "/api/auth/login"
      const fullName = [firstName.trim(), lastName.trim()].filter(Boolean).join(" ")
      const payload = mode === "register"
        ? { email: email.trim(), password, name: fullName, avatarId }
        : { email: email.trim(), password }

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      const data = await res.json()

      if (!res.ok || !data.success) {
        throw new Error(data.error || "Authentication failed. Please check your credentials.")
      }

      setSuccessMsg(mode === "register" ? "Account created successfully!" : "Signed in successfully!")

      setTimeout(() => {
        onAuthSuccess?.(data.user)
        onClose?.()
      }, 500)
    } catch (err) {
      setError(err.message || "An unexpected error occurred.")
    } finally {
      setLoading(false)
    }
  }

  const handleDemoSocialLogin = (provider) => {
    setError(`Direct OAuth with ${provider} will link via your Neon Auth / Google Cloud identity provider. For now, sign in with email/password.`)
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

        {/* Aceternity Container */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 12 }}
          transition={{ duration: 0.2 }}
          className="shadow-input relative mx-auto w-full max-w-md rounded-2xl border border-neutral-200 bg-white p-6 md:p-8 dark:border-neutral-800 dark:bg-black z-10"
        >
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute right-4 top-4 rounded-lg p-1.5 text-neutral-400 hover:bg-neutral-100 hover:text-neutral-700 dark:hover:bg-neutral-900 dark:hover:text-neutral-200 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>

          <div className="flex items-center gap-2 mb-1">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-indigo-600 text-white shadow-sm">
              <Satellite className="h-4 w-4" />
            </span>
            <span className="text-xs font-mono uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
              SatQuery AI
            </span>
          </div>

          <h2 className="text-xl font-bold text-neutral-800 dark:text-neutral-200">
            {mode === "login" ? "Welcome back" : "Create an account"}
          </h2>
          <p className="mt-1.5 text-xs text-neutral-600 dark:text-neutral-400">
            {mode === "login"
              ? "Sign in to access your synchronized satellite imagery and queries."
              : "Register to permanently save your remote sensing analyses to Neon PostgreSQL."}
          </p>

          {/* Mode Switcher Tabs */}
          <div className="mt-4 flex rounded-lg bg-neutral-100 p-1 dark:bg-neutral-900">
            <button
              type="button"
              onClick={() => {
                setMode("login")
                setError("")
                setSuccessMsg("")
              }}
              className={`flex-1 rounded-md py-1.5 text-xs font-semibold transition-all ${
                mode === "login"
                  ? "bg-white text-neutral-900 shadow-sm dark:bg-neutral-800 dark:text-neutral-100"
                  : "text-neutral-500 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("register")
                setError("")
                setSuccessMsg("")
              }}
              className={`flex-1 rounded-md py-1.5 text-xs font-semibold transition-all ${
                mode === "register"
                  ? "bg-white text-neutral-900 shadow-sm dark:bg-neutral-800 dark:text-neutral-100"
                  : "text-neutral-500 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200"
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* Alert feedback */}
          {error && (
            <div className="mt-3.5 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs font-medium text-rose-600 dark:text-rose-400">
              {error}
            </div>
          )}
          {successMsg && (
            <div className="mt-3.5 flex items-center gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-xs font-medium text-emerald-600 dark:text-emerald-400">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>{successMsg}</span>
            </div>
          )}

          {/* Form */}
          <form className="mt-5" onSubmit={handleSubmit}>
            {mode === "register" && (
              <>
                {/* KokonutUI Avatar Picker Strip */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1.5">
                    <Label className="text-xs font-medium">Choose Your Avatar</Label>
                    <span className="text-[10px] uppercase font-mono text-neutral-400">
                      Avatar {avatarId} Selected
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2.5 rounded-xl border border-neutral-200 bg-neutral-50/50 p-2 dark:border-neutral-800 dark:bg-neutral-900/50">
                    {AVATARS.map((av) => {
                      const isSelected = avatarId === av.id
                      return (
                        <button
                          key={av.id}
                          type="button"
                          onClick={() => setAvatarId(av.id)}
                          className={cn(
                            "relative flex h-12 w-12 items-center justify-center rounded-xl border transition-all duration-200",
                            isSelected
                              ? "border-indigo-600 ring-2 ring-indigo-500 ring-offset-2 ring-offset-white dark:ring-offset-black scale-105 shadow-md bg-white dark:bg-zinc-800"
                              : "border-neutral-200 dark:border-neutral-800 opacity-60 hover:opacity-100 hover:scale-102 bg-white/40 dark:bg-zinc-900/40"
                          )}
                          title={av.alt}
                        >
                          <UserAvatar avatarId={av.id} size={38} className="border-0 shadow-none" />
                          {isSelected && (
                            <span className="absolute -bottom-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-white text-[9px] shadow-sm">
                              ✓
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="mb-3 flex flex-col space-y-2 md:flex-row md:space-y-0 md:space-x-2">
                  <LabelInputContainer>
                    <Label htmlFor="firstname">First name</Label>
                    <Input
                      id="firstname"
                      placeholder="Akshit"
                      type="text"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                    />
                  </LabelInputContainer>
                  <LabelInputContainer>
                    <Label htmlFor="lastname">Last name</Label>
                    <Input
                      id="lastname"
                      placeholder="Tiwari"
                      type="text"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                    />
                  </LabelInputContainer>
                </div>
              </>
            )}

            <LabelInputContainer className="mb-3">
              <Label htmlFor="email">Email Address</Label>
              <Input
                id="email"
                placeholder="scientist@isro.gov.in"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </LabelInputContainer>

            <LabelInputContainer className="mb-5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                placeholder="••••••••"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </LabelInputContainer>

            <button
              className="group/btn relative flex h-10 w-full items-center justify-center rounded-md bg-gradient-to-br from-black to-neutral-600 font-medium text-white shadow-[0px_1px_0px_0px_#ffffff40_inset,0px_-1px_0px_0px_#ffffff40_inset] dark:bg-zinc-800 dark:from-zinc-900 dark:to-zinc-900 dark:shadow-[0px_1px_0px_0px_#27272a_inset,0px_-1px_0px_0px_#27272a_inset] disabled:opacity-60 transition duration-300"
              type="submit"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Connecting to Neon...</span>
                </span>
              ) : mode === "register" ? (
                <>
                  Sign up &rarr;
                  <BottomGradient />
                </>
              ) : (
                <>
                  Sign in &rarr;
                  <BottomGradient />
                </>
              )}
            </button>

            <div className="my-5 h-[1px] w-full bg-gradient-to-r from-transparent via-neutral-300 to-transparent dark:via-neutral-700" />

            <div className="flex flex-col space-y-2.5">
              <button
                type="button"
                onClick={() => handleDemoSocialLogin("GitHub")}
                className="group/btn shadow-input relative flex h-9 w-full items-center justify-start space-x-2.5 rounded-md bg-neutral-50 px-3.5 font-medium text-black dark:bg-zinc-900 dark:shadow-[0px_0px_1px_1px_#262626] transition hover:bg-neutral-100 dark:hover:bg-zinc-800"
              >
                <IconBrandGithub className="h-4 w-4 text-neutral-800 dark:text-neutral-300" />
                <span className="text-xs text-neutral-700 dark:text-neutral-300">
                  Continue with GitHub
                </span>
                <BottomGradient />
              </button>
              <button
                type="button"
                onClick={() => handleDemoSocialLogin("Google")}
                className="group/btn shadow-input relative flex h-9 w-full items-center justify-start space-x-2.5 rounded-md bg-neutral-50 px-3.5 font-medium text-black dark:bg-zinc-900 dark:shadow-[0px_0px_1px_1px_#262626] transition hover:bg-neutral-100 dark:hover:bg-zinc-800"
              >
                <IconBrandGoogle className="h-4 w-4 text-neutral-800 dark:text-neutral-300" />
                <span className="text-xs text-neutral-700 dark:text-neutral-300">
                  Continue with Google
                </span>
                <BottomGradient />
              </button>
            </div>
          </form>

          <p className="mt-4 text-center text-[11px] text-neutral-500 dark:text-neutral-400">
            Synced directly to Neon Serverless PostgreSQL
          </p>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}

const BottomGradient = () => {
  return (
    <>
      <span className="absolute inset-x-0 -bottom-px block h-px w-full bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-0 transition duration-500 group-hover/btn:opacity-100" />
      <span className="absolute inset-x-10 -bottom-px mx-auto block h-px w-1/2 bg-gradient-to-r from-transparent via-indigo-500 to-transparent opacity-0 blur-sm transition duration-500 group-hover/btn:opacity-100" />
    </>
  )
}

const LabelInputContainer = ({
  children,
  className,
}) => {
  return (
    <div className={cn("flex w-full flex-col space-y-1.5", className)}>
      {children}
    </div>
  )
}
