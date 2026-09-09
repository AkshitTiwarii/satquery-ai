"use client";

import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "motion/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import AICitation from "./ai-citation";

export type AIResponseCitation = {
  id: string;
  index: number;
  title: string;
  url?: string;
  snippet?: string;
};

export type AIResponseProps = {
  citations?: AIResponseCitation[];
  className?: string;
  isStreaming?: boolean;
  text: string;
};

const AIResponseCaret = ({
  shouldReduceMotion,
}: {
  shouldReduceMotion: boolean | null;
}) => (
  <motion.span
    animate={shouldReduceMotion ? { opacity: 1 } : { opacity: [1, 0.15, 1] }}
    aria-hidden="true"
    className="ml-0.5 inline-block h-[1em] w-[2px] translate-y-[0.15em] rounded-full bg-indigo-500 align-baseline"
    transition={
      shouldReduceMotion
        ? { duration: 0 }
        : { duration: 1, ease: "linear", repeat: Number.POSITIVE_INFINITY }
    }
  />
);

export default function AIResponse({
  citations = [],
  className,
  isStreaming = false,
  text = "",
}: AIResponseProps) {
  const shouldReduceMotion = useReducedMotion();

  // Pre-process citations if citations are provided: e.g. convert [1] to markdown link or citation badge
  const citationMap = new Map(citations.map((c) => [c.index, c]));

  return (
    <div
      className={cn(
        "ai-response-prose text-[15px] sm:text-[15.5px] leading-[1.7] text-zinc-800 dark:text-[#d1d1d6] selection:bg-indigo-500/20",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="font-serif text-xl sm:text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 mt-6 mb-3 first:mt-0">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="font-serif text-lg sm:text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100 mt-5 mb-2.5 first:mt-0">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="font-serif text-base sm:text-lg font-semibold text-zinc-900 dark:text-zinc-100 mt-4 mb-2 first:mt-0">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="mb-3.5 last:mb-0 leading-[1.7] text-zinc-800 dark:text-[#d1d1d6]">
              {children}
            </p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-zinc-950 dark:text-white">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-zinc-800 dark:text-zinc-200">
              {children}
            </em>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-3 border-l-2 border-zinc-400 dark:border-zinc-600 bg-zinc-100/60 dark:bg-zinc-900/60 px-4 py-2 rounded-r-lg text-zinc-700 dark:text-zinc-300 italic">
              {children}
            </blockquote>
          ),
          ul: ({ children }) => (
            <ul className="my-3 list-disc list-outside pl-5 space-y-2 text-zinc-800 dark:text-[#d1d1d6]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-3 list-decimal list-outside pl-5 space-y-2 text-zinc-800 dark:text-[#d1d1d6]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-[1.7] text-zinc-800 dark:text-[#d1d1d6]">
              {children}
            </li>
          ),
          code: ({ className: codeClassName, children, ...props }) => {
            const isBlock = /language-/.test(codeClassName || "");
            if (isBlock) {
              return (
                <div className="my-4 overflow-hidden rounded-xl border border-zinc-200/90 bg-[#161618] p-4 text-[13px] text-zinc-100 shadow-sm dark:border-zinc-850">
                  <code className="font-mono block whitespace-pre overflow-x-auto leading-relaxed">
                    {children}
                  </code>
                </div>
              );
            }
            return (
              <code className="rounded-md bg-zinc-200/80 px-1.5 py-0.5 font-mono text-[12.5px] font-medium text-rose-600 dark:bg-[#28272a] dark:text-[#f28b75] border border-zinc-300/40 dark:border-zinc-700/40">
                {children}
              </code>
            );
          },
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
              <table className="min-w-full divide-y divide-zinc-200 text-xs dark:divide-zinc-800">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-zinc-100 dark:bg-zinc-900/80 font-semibold text-zinc-900 dark:text-zinc-100">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-zinc-200 bg-transparent dark:divide-zinc-800">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-zinc-50 dark:hover:bg-zinc-900/40 transition-colors">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold">{children}</th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-zinc-700 dark:text-zinc-300">{children}</td>
          ),
          a: ({ href, children }) => {
            const textStr = String(children || "");
            const citeMatch = textStr.match(/^\[?(\d+)\]?$/);
            if (citeMatch) {
              const idx = parseInt(citeMatch[1], 10);
              const cite = citationMap.get(idx);
              if (cite) {
                return (
                  <AICitation
                    className="mx-0.5 align-baseline"
                    label={cite.index}
                    title={cite.title}
                    description={cite.snippet || "ISRO Bhoonidhi STAC Observation Scene"}
                    url={cite.url || href || "#"}
                  />
                );
              }
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sky-500 underline underline-offset-2 hover:text-sky-400 font-medium transition-colors"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
      {isStreaming ? (
        <AIResponseCaret shouldReduceMotion={shouldReduceMotion} />
      ) : null}
    </div>
  );
}
