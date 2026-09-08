"use client"

import React from "react"
import { User } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import SiriOrb from "./smoothui/components/siri-orb"
import { cls } from "./utils"

export default function Message({ role, children, content, state = "idle", copyText }) {
  const isUser = role === "user"

  return (
    <div className={cls("flex gap-3.5 my-3", isUser ? "justify-end" : "justify-start")}>
      {/* Assistant Avatar: Clean dynamic SiriOrb without gradients or pulsating dots */}
      {!isUser && (
        <div className="mt-1 shrink-0">
          <SiriOrb size="26px" state={state} />
        </div>
      )}

      {/* Message Body */}
      <div
        className={cls(
          "max-w-[88%] rounded-2xl px-4 py-3 text-sm shadow-sm transition-all",
          isUser
            ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 font-medium"
            : "border border-zinc-200/80 bg-white/95 text-zinc-800 backdrop-blur-md dark:border-zinc-800/90 dark:bg-zinc-900/90 dark:text-zinc-200",
        )}
      >
        {content ? (
          <div className="prose prose-sm dark:prose-invert max-w-none break-words leading-relaxed space-y-2">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                strong: ({ children }) => (
                  <strong className="font-semibold text-zinc-950 dark:text-white">{children}</strong>
                ),
                code: ({ children }) => (
                  <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs text-sky-600 dark:bg-zinc-800 dark:text-sky-400">
                    {children}
                  </code>
                ),
                ul: ({ children }) => <ul className="list-disc pl-4 space-y-1 mb-2">{children}</ul>,
                li: ({ children }) => <li>{children}</li>,
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          children
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-zinc-800 text-white dark:bg-zinc-200 dark:text-zinc-900 shadow-sm">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  )
}
