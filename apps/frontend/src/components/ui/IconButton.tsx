'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { useRipple } from '@/hooks/useRipple';

/**
 * Material Design 3 Icon Button
 *
 * Features:
 * - Minimum 44x44px touch target (WCAG 2.5.5 Level AAA)
 * - Standard, filled, tonal, and outlined variants
 * - Multiple sizes (small, medium, large)
 * - State layer (hover, focus, active)
 * - Ripple effect animation
 * - Loading state
 * - Mobile-optimized with haptic feedback
 * - Accessibility (ARIA labels, focus indicators)
 */

const iconButtonVariants = cva(
  `inline-flex items-center justify-center font-medium transition-all duration-200
   focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-offset-0
   disabled:opacity-38 disabled:cursor-not-allowed disabled:pointer-events-none
   relative overflow-hidden touch-manipulation rounded-full`,
  {
    variants: {
      variant: {
        standard: `text-gray-700 hover:bg-gray-100 active:bg-gray-200
                  focus-visible:ring-gray-500/30
                  dark:text-gray-300 dark:hover:bg-gray-800 dark:active:bg-gray-700`,
        filled: `bg-primary-500 text-white hover:bg-primary-600 active:bg-primary-700
                shadow-md-2 hover:shadow-md-3
                focus-visible:ring-primary-500/30
                dark:bg-primary-600 dark:hover:bg-primary-700`,
        'filled-tonal': `bg-primary-100 text-primary-700 hover:bg-primary-200 active:bg-primary-300
                        focus-visible:ring-primary-500/30
                        dark:bg-primary-900 dark:text-primary-300 dark:hover:bg-primary-800`,
        outlined: `border-2 border-gray-300 text-gray-700 hover:bg-gray-100 active:bg-gray-200
                  focus-visible:ring-gray-500/30
                  dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800`,
      },
      size: {
        small: 'h-12 w-12 min-h-[48px] min-w-[48px] sm:h-11 sm:w-11 sm:min-h-[44px] sm:min-w-[44px]',
        medium: 'h-12 w-12 min-h-[48px] min-w-[48px] sm:h-11 sm:w-11 sm:min-h-[44px] sm:min-w-[44px]',
        large: 'h-14 w-14 min-h-[56px] min-w-[56px] sm:h-12 sm:w-12 sm:min-h-[48px] sm:min-w-[48px]',
      },
    },
    defaultVariants: {
      variant: 'standard',
      size: 'medium',
    },
  }
);

const iconSizeVariants = cva('transition-transform', {
  variants: {
    size: {
      small: 'w-6 h-6 sm:w-5 sm:h-5',
      medium: 'w-6 h-6 sm:w-6 sm:h-6',
      large: 'w-8 h-8 sm:w-7 sm:h-7',
    },
  },
  defaultVariants: {
    size: 'medium',
  },
});

export interface IconButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof iconButtonVariants> {
  icon?: React.ReactNode;
  loading?: boolean;
  badge?: number | boolean;
}

const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  (
    {
      className,
      variant,
      size,
      icon,
      loading,
      disabled,
      badge,
      children,
      onClick,
      ...props
    },
    ref
  ) => {
    const createRipple = useRipple('rgba(0, 0, 0, 0.2)');

    // Handle click with optional haptic feedback
    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      // Trigger haptic feedback on mobile devices
      if ('vibrate' in navigator) {
        navigator.vibrate(10);
      }

      createRipple(e as any);
      onClick?.(e);
    };

    return (
      <button
        ref={ref}
        className={cn(iconButtonVariants({ variant, size }), className)}
        disabled={disabled || loading}
        onClick={handleClick}
        {...props}
      >
        {/* Loading spinner */}
        {loading && (
          <span className={cn(iconSizeVariants({ size }), 'animate-spin')}>
            <svg
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="3"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </span>
        )}

        {/* Icon */}
        {!loading && icon && (
          <span className={iconSizeVariants({ size })}>
            {icon}
          </span>
        )}

        {/* Children (for custom content) */}
        {!loading && !icon && children}

        {/* Badge indicator */}
        {badge && (
          <span className="absolute -top-1 -right-1 flex items-center justify-center min-w-[20px] h-[20px] px-1 text-xs sm:min-w-[18px] sm:h-[18px] sm:text-[10px] bg-error-500 text-white font-bold rounded-full border-2 border-surface-1 dark:border-gray-900">
            {typeof badge === 'number' && badge > 99 ? '99+' : typeof badge === 'number' ? badge : ''}
          </span>
        )}
      </button>
    );
  }
);

IconButton.displayName = 'IconButton';

export { IconButton, iconButtonVariants };
