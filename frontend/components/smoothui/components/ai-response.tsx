"use client";

import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "motion/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type AIResponseCitation = {
  id: string;
  index: number;
  title: string;
  url?: string;
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
        "ai-response-prose text-sm leading-relaxed text-zinc-800 dark:text-zinc-200 selection:bg-indigo-500/20",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 mt-4 mb-2 first:mt-0">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 mt-3.5 mb-1.5 first:mt-0">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 mt-3 mb-1 first:mt-0">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="mb-2.5 last:mb-0 leading-relaxed text-zinc-800 dark:text-zinc-200">
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
            <blockquote className="my-2.5 border-l-2 border-indigo-500/60 bg-indigo-50/50 dark:bg-indigo-950/20 px-3.5 py-1.5 rounded-r-lg text-zinc-700 dark:text-zinc-300 italic">
              {children}
            </blockquote>
          ),
          ul: ({ children }) => (
            <ul className="my-2 list-disc list-outside pl-5 space-y-1 text-zinc-800 dark:text-zinc-200">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2 list-decimal list-outside pl-5 space-y-1 text-zinc-800 dark:text-zinc-200">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed text-zinc-800 dark:text-zinc-200">
              {children}
            </li>
          ),
          code: ({ className: codeClassName, children, ...props }) => {
            const isBlock = /language-/.test(codeClassName || "");
            if (isBlock) {
              return (
                <div className="my-3 overflow-hidden rounded-xl border border-zinc-200 bg-zinc-950 p-3 text-xs text-zinc-100 shadow-sm dark:border-zinc-800">
                  <code className="font-mono block whitespace-pre overflow-x-auto">
                    {children}
                  </code>
                </div>
              );
            }
            return (
              <code className="rounded bg-zinc-200/70 px-1.5 py-0.5 font-mono text-xs font-medium text-indigo-700 dark:bg-zinc-800 dark:text-indigo-300">
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
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 underline underline-offset-2 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
            >
              {children}
            </a>
          ),
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
