'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';
import { hapticFeedback } from '@/lib/haptics';

/**
 * CodeKeyboardBar - Mobile-optimized keyboard accessory for code input
 *
 * Features:
 * - Quick access to common code symbols
 * - Indentation controls
 * - Undo/redo shortcuts
 * - Hide keyboard button
 * - Safe area aware positioning
 */

interface CodeKeyboardBarProps {
  onInsert?: (text: string) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onHideKeyboard?: () => void;
  onIndent?: () => void;
  onOutdent?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  className?: string;
  visible?: boolean;
}

// Common code symbols organized by frequency of use
const SYMBOLS = [
  { key: 'tab', label: '⇥', insert: '\t', title: 'Tab' },
  { key: 'brackets', label: '{ }', insert: '{}', title: 'Braces' },
  { key: 'parens', label: '( )', insert: '()', title: 'Parentheses' },
  { key: 'square', label: '[ ]', insert: '[]', title: 'Brackets' },
  { key: 'quotes', label: '" "', insert: '""', title: 'Double quotes' },
  { key: 'single', label: "' '", insert: "''", title: 'Single quotes' },
  { key: 'backtick', label: '` `', insert: '``', title: 'Backticks' },
  { key: 'lt', label: '<', insert: '<', title: 'Less than' },
  { key: 'gt', label: '>', insert: '>', title: 'Greater than' },
  { key: 'slash', label: '/', insert: '/', title: 'Slash' },
  { key: 'backslash', label: '\\', insert: '\\', title: 'Backslash' },
  { key: 'semicolon', label: ';', insert: ';', title: 'Semicolon' },
  { key: 'colon', label: ':', insert: ':', title: 'Colon' },
  { key: 'arrow', label: '=>', insert: '=>', title: 'Arrow function' },
  { key: 'pipe', label: '|', insert: '|', title: 'Pipe' },
  { key: 'amp', label: '&', insert: '&', title: 'Ampersand' },
  { key: 'hash', label: '#', insert: '#', title: 'Hash' },
  { key: 'dollar', label: '$', insert: '$', title: 'Dollar sign' },
  { key: 'underscore', label: '_', insert: '_', title: 'Underscore' },
  { key: 'equals', label: '=', insert: '=', title: 'Equals' },
];

export function CodeKeyboardBar({
  onInsert,
  onUndo,
  onRedo,
  onHideKeyboard,
  onIndent,
  onOutdent,
  canUndo = true,
  canRedo = true,
  className,
  visible = true,
}: CodeKeyboardBarProps) {
  const scrollRef = React.useRef<HTMLDivElement>(null);

  if (!visible) return null;

  const handleSymbolClick = (symbol: typeof SYMBOLS[0]) => {
    hapticFeedback('light');
    onInsert?.(symbol.insert);
  };

  const handleAction = (action: () => void, pattern: 'light' | 'medium' = 'light') => {
    hapticFeedback(pattern);
    action();
  };

  return (
    <div
      className={cn(
        'fixed left-0 right-0 bottom-0 z-[60]',
        'bg-gray-100 dark:bg-gray-800',
        'border-t border-gray-200 dark:border-gray-700',
        'pb-safe', // Safe area padding
        className
      )}
      style={{
        // Additional safe area support
        paddingBottom: 'env(safe-area-inset-bottom, 0px)',
      }}
    >
      <div className="flex items-center h-11">
        {/* Undo/Redo */}
        <div className="flex items-center gap-1 px-2 border-r border-gray-200 dark:border-gray-700">
          <button
            onClick={() => onUndo && handleAction(onUndo)}
            disabled={!canUndo}
            className={cn(
              'w-9 h-9 flex items-center justify-center rounded-lg',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-gray-200 dark:hover:bg-gray-700',
              'active:bg-gray-300 dark:active:bg-gray-600',
              'disabled:opacity-30 disabled:pointer-events-none',
              'touch-manipulation'
            )}
            aria-label="Undo"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
            </svg>
          </button>
          <button
            onClick={() => onRedo && handleAction(onRedo)}
            disabled={!canRedo}
            className={cn(
              'w-9 h-9 flex items-center justify-center rounded-lg',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-gray-200 dark:hover:bg-gray-700',
              'active:bg-gray-300 dark:active:bg-gray-600',
              'disabled:opacity-30 disabled:pointer-events-none',
              'touch-manipulation'
            )}
            aria-label="Redo"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
            </svg>
          </button>
        </div>

        {/* Indentation */}
        <div className="flex items-center gap-1 px-2 border-r border-gray-200 dark:border-gray-700">
          <button
            onClick={() => onOutdent && handleAction(onOutdent)}
            className={cn(
              'w-9 h-9 flex items-center justify-center rounded-lg',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-gray-200 dark:hover:bg-gray-700',
              'active:bg-gray-300 dark:active:bg-gray-600',
              'touch-manipulation'
            )}
            aria-label="Decrease indent"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14V5" />
            </svg>
          </button>
          <button
            onClick={() => onIndent && handleAction(onIndent)}
            className={cn(
              'w-9 h-9 flex items-center justify-center rounded-lg',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-gray-200 dark:hover:bg-gray-700',
              'active:bg-gray-300 dark:active:bg-gray-600',
              'touch-manipulation'
            )}
            aria-label="Increase indent"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5v14" />
            </svg>
          </button>
        </div>

        {/* Scrollable symbols */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-x-auto scrollbar-hide"
          style={{
            WebkitOverflowScrolling: 'touch',
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
          }}
        >
          <div className="flex items-center gap-1 px-2 min-w-max">
            {SYMBOLS.map((symbol) => (
              <button
                key={symbol.key}
                onClick={() => handleSymbolClick(symbol)}
                className={cn(
                  'min-w-[36px] h-9 px-2.5 flex items-center justify-center rounded-lg',
                  'text-sm font-mono font-medium',
                  'text-gray-700 dark:text-gray-300',
                  'bg-white dark:bg-gray-700',
                  'border border-gray-200 dark:border-gray-600',
                  'hover:bg-gray-50 dark:hover:bg-gray-600',
                  'active:bg-gray-100 dark:active:bg-gray-500',
                  'touch-manipulation shadow-sm'
                )}
                title={symbol.title}
                aria-label={symbol.title}
              >
                {symbol.label}
              </button>
            ))}
          </div>
        </div>

        {/* Hide keyboard */}
        <div className="flex items-center px-2 border-l border-gray-200 dark:border-gray-700">
          <button
            onClick={() => onHideKeyboard && handleAction(onHideKeyboard, 'medium')}
            className={cn(
              'w-9 h-9 flex items-center justify-center rounded-lg',
              'text-gray-700 dark:text-gray-300',
              'hover:bg-gray-200 dark:hover:bg-gray-700',
              'active:bg-gray-300 dark:active:bg-gray-600',
              'touch-manipulation'
            )}
            aria-label="Hide keyboard"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default CodeKeyboardBar;
