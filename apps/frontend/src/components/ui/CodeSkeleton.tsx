'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

/**
 * CodeSkeleton - Loading skeleton for code blocks
 *
 * Features:
 * - Mimics code block layout with line numbers
 * - Animated shimmer effect
 * - Variable line lengths for realistic appearance
 * - Responsive sizing
 */

interface CodeSkeletonProps {
  lines?: number;
  showLineNumbers?: boolean;
  showHeader?: boolean;
  className?: string;
}

// Predefined line widths for realistic code appearance
const LINE_WIDTHS = [
  '75%', '40%', '85%', '60%', '90%', '45%', '70%', '55%',
  '80%', '35%', '65%', '50%', '95%', '30%', '75%', '60%',
];

export function CodeSkeleton({
  lines = 10,
  showLineNumbers = true,
  showHeader = true,
  className,
}: CodeSkeletonProps) {
  // Generate consistent line widths based on line number
  const getLineWidth = (index: number) => {
    return LINE_WIDTHS[index % LINE_WIDTHS.length];
  };

  return (
    <div
      className={cn(
        'rounded-lg overflow-hidden',
        'bg-gray-900',
        'animate-pulse',
        className
      )}
    >
      {/* Header skeleton */}
      {showHeader && (
        <div className="flex items-center justify-between px-4 py-3 bg-gray-800/80 border-b border-gray-700/50">
          <div className="flex items-center gap-3">
            {/* Window controls skeleton */}
            <div className="hidden sm:flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-gray-600" />
              <div className="w-3 h-3 rounded-full bg-gray-600" />
              <div className="w-3 h-3 rounded-full bg-gray-600" />
            </div>

            {/* Language badge skeleton */}
            <div className="flex items-center gap-2 pl-3 sm:border-l border-gray-700">
              <div className="w-5 h-5 rounded bg-gray-600" />
              <div className="hidden sm:block w-16 h-4 rounded bg-gray-600" />
            </div>

            {/* File stats skeleton */}
            <div className="hidden md:block w-24 h-3 rounded bg-gray-700" />
          </div>

          {/* Controls skeleton */}
          <div className="flex items-center gap-1">
            <div className="w-8 h-8 rounded bg-gray-700" />
            <div className="w-8 h-8 rounded bg-gray-700" />
            <div className="w-16 h-8 rounded bg-gray-700" />
          </div>
        </div>
      )}

      {/* Code area skeleton */}
      <div className="flex min-h-[200px] max-h-[500px]">
        {/* Line numbers skeleton */}
        {showLineNumbers && (
          <div className="flex-shrink-0 py-4 px-3 bg-gray-900/50 border-r border-gray-700/50">
            {Array.from({ length: lines }).map((_, index) => (
              <div
                key={index}
                className="flex justify-end items-center h-6"
              >
                <div
                  className="h-3 rounded bg-gray-700/50"
                  style={{
                    width: `${String(lines).length * 8 + 8}px`,
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Code content skeleton */}
        <div className="flex-1 p-4 space-y-1">
          {Array.from({ length: lines }).map((_, index) => (
            <div key={index} className="h-6 flex items-center">
              {/* Random indentation */}
              <div
                style={{
                  width: `${(index % 4) * 24}px`,
                }}
              />
              {/* Code line */}
              <div
                className="h-3 rounded bg-gray-700/60"
                style={{
                  width: getLineWidth(index),
                }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Footer skeleton (optional) */}
      <div className="hidden sm:flex items-center justify-between px-4 py-1.5 bg-gray-800/80 border-t border-gray-700/50">
        <div className="flex items-center gap-4">
          <div className="w-16 h-3 rounded bg-gray-700" />
          <div className="w-12 h-3 rounded bg-gray-700" />
          <div className="w-8 h-3 rounded bg-gray-700" />
        </div>
        <div className="flex items-center gap-4">
          <div className="w-16 h-3 rounded bg-gray-700" />
        </div>
      </div>

      {/* Shimmer overlay */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite]"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent)',
          }}
        />
      </div>
    </div>
  );
}

/**
 * Inline code skeleton for smaller code snippets
 */
export function InlineCodeSkeleton({
  width = '100px',
  className,
}: {
  width?: string | number;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'inline-block h-5 rounded bg-gray-700/60 animate-pulse',
        className
      )}
      style={{ width }}
    />
  );
}

export default CodeSkeleton;
