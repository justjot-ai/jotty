'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

/**
 * Material Design 3 Chip
 *
 * Features:
 * - Multiple variants (assist, filter, input, suggestion)
 * - Selected state for filter chips
 * - Dismiss/remove functionality
 * - Leading icon support
 * - Avatar support
 * - Minimum 44px height for touch targets (WCAG 2.5.5 Level AAA)
 * - Ripple effect animation
 * - Mobile-optimized
 */

const chipVariants = cva(
  `inline-flex items-center font-medium transition-all duration-200
   focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-0
   disabled:opacity-38 disabled:cursor-not-allowed disabled:pointer-events-none
   relative overflow-hidden touch-manipulation rounded-md-sm`,
  {
    variants: {
      variant: {
        assist: `bg-surface-3 text-gray-900 hover:bg-surface-4 active:bg-surface-5
                focus-visible:ring-primary-500/30
                dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700`,
        filter: `border border-gray-300 bg-transparent text-gray-700 hover:bg-gray-100
                focus-visible:ring-primary-500/30
                dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800`,
        'filter-selected': `bg-primary-100 text-primary-700 border border-primary-300
                           hover:bg-primary-200 active:bg-primary-300
                           focus-visible:ring-primary-500/30
                           dark:bg-primary-900 dark:text-primary-300 dark:border-primary-700`,
        input: `bg-primary-100 text-primary-700 hover:bg-primary-200
               focus-visible:ring-primary-500/30
               dark:bg-primary-900 dark:text-primary-300`,
        suggestion: `border border-gray-300 bg-transparent text-gray-700 hover:bg-gray-100
                    focus-visible:ring-primary-500/30
                    dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800`,
      },
      size: {
        small: 'text-sm gap-2 px-3 py-2.5 min-h-[44px] sm:text-xs sm:gap-1.5 sm:px-2 sm:py-2 sm:min-h-[36px]',
        medium: 'text-base gap-2.5 px-4 py-3 min-h-[48px] sm:text-sm sm:gap-2 sm:px-3 sm:py-2 sm:min-h-[40px]',
        large: 'text-lg gap-3 px-5 py-3.5 min-h-[52px] sm:text-base sm:gap-2.5 sm:px-4 sm:py-2.5 sm:min-h-[44px]',
      },
    },
    defaultVariants: {
      variant: 'assist',
      size: 'medium',
    },
  }
);

export interface ChipProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof chipVariants> {
  label: string;
  icon?: React.ReactNode;
  avatar?: React.ReactNode;
  onRemove?: () => void;
  selected?: boolean;
}

const Chip = React.forwardRef<HTMLButtonElement, ChipProps>(
  (
    {
      className,
      variant,
      size,
      label,
      icon,
      avatar,
      onRemove,
      selected,
      onClick,
      ...props
    },
    ref
  ) => {
    // Automatically use filter-selected variant if selected
    const finalVariant = selected ? 'filter-selected' : variant;

    // Handle click with optional haptic feedback
    const handleClick = (e: React.MouseEvent<HTMLElement>) => {
      if ('vibrate' in navigator) {
        navigator.vibrate(5);
      }
      onClick?.(e as any);
    };

    // Handle remove with haptic feedback
    const handleRemove = (e: React.MouseEvent<HTMLElement>) => {
      e.stopPropagation();
      if ('vibrate' in navigator) {
        navigator.vibrate(10);
      }
      onRemove?.();
    };

    // Render as button when clickable, otherwise as div
    if (onClick) {
      return (
        <button
          ref={ref}
          className={cn(chipVariants({ variant: finalVariant, size }), className)}
          onClick={handleClick}
          type="button"
          {...props}
        >
        {/* Ripple effect (only for clickable chips) */}
        <span className="absolute inset-0 overflow-hidden rounded-md-sm">
          <span className="absolute inset-0 bg-current opacity-0 scale-0 transition-all duration-600 group-active:opacity-10 group-active:scale-100" />
        </span>

        {/* Avatar (if provided) */}
        {avatar && (
          <span className="flex-shrink-0 -ml-1">
            {avatar}
          </span>
        )}

        {/* Leading Icon */}
        {!avatar && icon && (
          <span className="flex-shrink-0 w-5 h-5 sm:w-4 sm:h-4">
            {icon}
          </span>
        )}

        {/* Label */}
        <span className="truncate max-w-[240px] sm:max-w-[200px]">
          {label}
        </span>

        {/* Remove button (for input chips) */}
        {onRemove && (
          <button
            type="button"
            onClick={handleRemove}
            className="flex-shrink-0 -mr-1 p-1 sm:p-0.5 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors min-w-[44px] min-h-[44px] sm:min-w-0 sm:min-h-0 flex items-center justify-center"
            aria-label="Remove"
          >
            <svg
              className="w-5 h-5 sm:w-4 sm:h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}

        {/* Selected checkmark (for filter chips) */}
        {selected && variant === 'filter' && (
          <span className="flex-shrink-0 w-5 h-5 sm:w-4 sm:h-4 text-primary-700 dark:text-primary-300">
            <svg
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2.5}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </span>
        )}
        </button>
      );
    }

    // Non-clickable chip - render as div
    return (
      <div
        className={cn(chipVariants({ variant: finalVariant, size }), className)}
      >
        {/* Avatar (if provided) */}
        {avatar && (
          <span className="flex-shrink-0 -ml-1">
            {avatar}
          </span>
        )}

        {/* Icon (if provided) */}
        {icon && (
          <span className="flex-shrink-0 w-5 h-5 sm:w-4 sm:h-4">{icon}</span>
        )}

        {/* Label */}
        <span className="truncate max-w-[240px] sm:max-w-[200px]">{label}</span>

        {/* Remove button (optional) */}
        {onRemove && (
          <button
            type="button"
            onClick={handleRemove}
            className="flex-shrink-0 -mr-1 p-1 sm:p-0.5 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors min-w-[44px] min-h-[44px] sm:min-w-0 sm:min-h-0 flex items-center justify-center"
            aria-label="Remove"
          >
            <svg
              className="w-5 h-5 sm:w-4 sm:h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}

        {/* Selected checkmark (for filter chips) */}
        {selected && variant === 'filter' && (
          <span className="flex-shrink-0 w-5 h-5 sm:w-4 sm:h-4 text-primary-700 dark:text-primary-300">
            <svg
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2.5}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </span>
        )}
      </div>
    );
  }
);

Chip.displayName = 'Chip';

/**
 * Chip Group - for managing multiple chips
 */
export function ChipGroup({
  children,
  className,
  wrap = true,
}: {
  children: React.ReactNode;
  className?: string;
  wrap?: boolean;
}) {
  return (
    <div
      className={cn(
        'flex gap-3 sm:gap-2',
        wrap ? 'flex-wrap' : 'overflow-x-auto scrollbar-hide',
        className
      )}
    >
      {children}
    </div>
  );
}

export { Chip, chipVariants };
