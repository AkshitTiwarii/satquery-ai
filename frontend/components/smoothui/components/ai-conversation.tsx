"use client";

import { cn } from "@/lib/utils";
import { ArrowDown } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

const SPRING_DEFAULT = {
  bounce: 0.1,
  duration: 0.25,
  type: "spring" as const,
};

const BOTTOM_THRESHOLD_PX = 48;

export type AIConversationProps = {
  children: ReactNode;
  className?: string;
  contentKey?: string | number;
};

const AIConversation = ({
  children,
  className,
  contentKey,
}: AIConversationProps) => {
  const shouldReduceMotion = useReducedMotion();
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [isPinned, setIsPinned] = useState(true);

  const measure = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    const distance =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    setIsPinned(distance <= BOTTOM_THRESHOLD_PX);
  }, []);

  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    if (typeof viewport.scrollTo === "function") {
      viewport.scrollTo({ behavior, top: viewport.scrollHeight });
    } else {
      viewport.scrollTop = viewport.scrollHeight;
    }
    setIsPinned(true);
  }, []);

  const isFirstMountRef = useRef(true);
  const prevContentKeyRef = useRef(contentKey);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const prevKeyStr = String(prevContentKeyRef.current || "");
    const currKeyStr = String(contentKey || "");
    const prevConvId = prevKeyStr.split("-")[0];
    const currConvId = currKeyStr.split("-")[0];
    const isConvSwitch = prevConvId !== currConvId;

    if (isFirstMountRef.current || isConvSwitch) {
      // Instantly position at the bottom without any visible scrolling from the top
      viewport.scrollTop = viewport.scrollHeight;
      setIsPinned(true);
      isFirstMountRef.current = false;
      prevContentKeyRef.current = contentKey;

      // Ensure any subsequent font/image reflows keep it pinned at bottom without scrolling from top
      const raf = requestAnimationFrame(() => {
        if (viewport) viewport.scrollTop = viewport.scrollHeight;
      });
      const t = setTimeout(() => {
        if (viewport) viewport.scrollTop = viewport.scrollHeight;
      }, 50);

      return () => {
        cancelAnimationFrame(raf);
        clearTimeout(t);
      };
    }

    if (isPinned) {
      scrollToBottom(shouldReduceMotion ? "auto" : "smooth");
    }
    prevContentKeyRef.current = contentKey;
  }, [contentKey, isPinned, scrollToBottom, shouldReduceMotion]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    const observer = new ResizeObserver(measure);
    for (const child of Array.from(viewport.children)) {
      observer.observe(child);
    }
    return () => observer.disconnect();
  }, [measure]);

  return (
    <div className={cn("relative min-h-0 w-full", className)}>
      <div
        className="h-full overflow-y-auto overscroll-contain no-scrollbar [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
        onScroll={measure}
        ref={viewportRef}
      >
        {children}
      </div>

      <AnimatePresence initial={false}>
        {!isPinned && (
          <motion.button
            animate={{ opacity: 1, scale: 1, y: 0 }}
            aria-label="Jump to latest"
            className="absolute inset-x-0 bottom-3 mx-auto flex w-fit cursor-pointer items-center gap-1.5 rounded-full border border-border bg-background/90 py-1.5 pr-3 pl-2.5 text-foreground text-xs shadow-sm backdrop-blur"
            exit={
              shouldReduceMotion
                ? { opacity: 0, transition: { duration: 0 } }
                : { opacity: 0, scale: 0.96, y: 8 }
            }
            initial={
              shouldReduceMotion
                ? { opacity: 1, scale: 1, y: 0 }
                : { opacity: 0, scale: 0.96, y: 8 }
            }
            onClick={() =>
              scrollToBottom(shouldReduceMotion ? "auto" : "smooth")
            }
            transition={shouldReduceMotion ? { duration: 0 } : SPRING_DEFAULT}
            type="button"
          >
            <ArrowDown aria-hidden="true" size={13} />
            Jump to latest
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AIConversation;
