"use client"
import React from "react"
import { cn } from "@/lib/utils"

export const AVATAR_RGB = {
  1: "255, 0, 91",
  2: "255, 125, 16",
  3: "255, 0, 91",
  4: "137, 252, 179",
}

export const AVATARS = [
  {
    id: 1,
    alt: "Avatar 1",
    rgb: "255, 0, 91",
    renderSvg: (size = 36) => (
      <svg
        aria-label="Avatar 1"
        fill="none"
        height={size}
        width={size}
        viewBox="0 0 36 36"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
      >
        <mask height="36" id="avatar-mask-1" maskUnits="userSpaceOnUse" width="36" x="0" y="0">
          <rect fill="#FFFFFF" height="36" rx="72" width="36" />
        </mask>
        <g mask="url(#avatar-mask-1)">
          <rect fill="#ff005b" height="36" width="36" />
          <rect
            fill="#ffb238"
            height="36"
            rx="6"
            transform="translate(9 -5) rotate(219 18 18) scale(1)"
            width="36"
            x="0"
            y="0"
          />
          <g transform="translate(4.5 -4) rotate(9 18 18)">
            <path d="M15 19c2 1 4 1 6 0" fill="none" stroke="#000000" strokeLinecap="round" />
            <rect fill="#000000" height="2" rx="1" stroke="none" width="1.5" x="10" y="14" />
            <rect fill="#000000" height="2" rx="1" stroke="none" width="1.5" x="24" y="14" />
          </g>
        </g>
      </svg>
    ),
  },
  {
    id: 2,
    alt: "Avatar 2",
    rgb: "255, 125, 16",
    renderSvg: (size = 36) => (
      <svg
        aria-label="Avatar 2"
        fill="none"
        height={size}
        width={size}
        viewBox="0 0 36 36"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
      >
        <mask height="36" id="avatar-mask-2" maskUnits="userSpaceOnUse" width="36" x="0" y="0">
          <rect fill="#FFFFFF" height="36" rx="72" width="36" />
        </mask>
        <g mask="url(#avatar-mask-2)">
          <rect fill="#ff7d10" height="36" width="36" />
          <rect
            fill="#0a0310"
            height="36"
            rx="6"
            transform="translate(5 -1) rotate(55 18 18) scale(1.1)"
            width="36"
            x="0"
            y="0"
          />
          <g transform="translate(7 -6) rotate(-5 18 18)">
            <path d="M15 20c2 1 4 1 6 0" fill="none" stroke="#FFFFFF" strokeLinecap="round" />
            <rect fill="#FFFFFF" height="2" rx="1" stroke="none" width="1.5" x="14" y="14" />
            <rect fill="#FFFFFF" height="2" rx="1" stroke="none" width="1.5" x="20" y="14" />
          </g>
        </g>
      </svg>
    ),
  },
  {
    id: 3,
    alt: "Avatar 3",
    rgb: "255, 0, 91",
    renderSvg: (size = 36) => (
      <svg
        aria-label="Avatar 3"
        fill="none"
        height={size}
        width={size}
        viewBox="0 0 36 36"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
      >
        <mask height="36" id="avatar-mask-3" maskUnits="userSpaceOnUse" width="36" x="0" y="0">
          <rect fill="#FFFFFF" height="36" rx="72" width="36" />
        </mask>
        <g mask="url(#avatar-mask-3)">
          <rect fill="#0a0310" height="36" width="36" />
          <rect
            fill="#ff005b"
            height="36"
            rx="36"
            transform="translate(-3 7) rotate(227 18 18) scale(1.2)"
            width="36"
            x="0"
            y="0"
          />
          <g transform="translate(-3 3.5) rotate(7 18 18)">
            <path d="M13,21 a1,0.75 0 0,0 10,0" fill="#FFFFFF" />
            <rect fill="#FFFFFF" height="2" rx="1" stroke="none" width="1.5" x="12" y="14" />
            <rect fill="#FFFFFF" height="2" rx="1" stroke="none" width="1.5" x="22" y="14" />
          </g>
        </g>
      </svg>
    ),
  },
  {
    id: 4,
    alt: "Avatar 4",
    rgb: "137, 252, 179",
    renderSvg: (size = 36) => (
      <svg
        aria-label="Avatar 4"
        fill="none"
        height={size}
        width={size}
        viewBox="0 0 36 36"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
      >
        <mask height="36" id="avatar-mask-4" maskUnits="userSpaceOnUse" width="36" x="0" y="0">
          <rect fill="#FFFFFF" height="36" rx="72" width="36" />
        </mask>
        <g mask="url(#avatar-mask-4)">
          <rect fill="#d8fcb3" height="36" width="36" />
          <rect
            fill="#89fcb3"
            height="36"
            rx="6"
            transform="translate(9 -5) rotate(219 18 18) scale(1)"
            width="36"
            x="0"
            y="0"
          />
          <g transform="translate(4.5 -4) rotate(9 18 18)">
            <path d="M15 19c2 1 4 1 6 0" fill="none" stroke="#000000" strokeLinecap="round" />
            <rect fill="#000000" height="2" rx="1" stroke="none" width="1.5" x="10" y="14" />
            <rect fill="#000000" height="2" rx="1" stroke="none" width="1.5" x="24" y="14" />
          </g>
        </g>
      </svg>
    ),
  },
]

export default function UserAvatar({
  avatarId = 1,
  name = "User",
  size = 32,
  className = "",
}) {
  const selected = AVATARS.find((a) => a.id === Number(avatarId)) || AVATARS[0]

  return (
    <div
      style={{ width: size, height: size }}
      className={cn(
        "relative shrink-0 overflow-hidden rounded-full shadow-xs border border-white/20 dark:border-zinc-700/60 bg-zinc-900 flex items-center justify-center select-none",
        className
      )}
    >
      <div className="flex items-center justify-center w-full h-full">
        {selected.renderSvg(size)}
      </div>
    </div>
  )
}
