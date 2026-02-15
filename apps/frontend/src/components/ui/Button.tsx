'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { useRipple } from '@/hooks/useRipple';

const buttonVariants = cva(
  `inline-flex items-center justify-center whitespace-nowrap rounded-lg
   font-medium transition-all duration-200 focus-visible:outline-none
   focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50
   disabled:cursor-not-allowed active:scale-95 relative overflow-hidden
   touch-manipulation`,
  {
    variants: {
      variant: {
        default: `bg-indigo-600 text-white hover:bg-indigo-700
                 dark:bg-indigo-500 dark:hover:bg-indigo-600
                 focus-visible:ring-indigo-500 dark:focus-visible:ring-indigo-400
                 shadow-sm hover:shadow-md`,
        secondary: `bg-gray-200 text-gray-900 hover:bg-gray-300
                   dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700
                   focus-visible:ring-gray-500`,
        destructive: `bg-red-600 text-white hover:bg-red-700
                     dark:bg-red-500 dark:hover:bg-red-600
                     focus-visible:ring-red-500`,
        ghost: `text-gray-700 dark:text-gray-300 hover:bg-gray-100
               dark:hover:bg-gray-900 focus-visible:ring-gray-500`,
        outline: `border-2 border-gray-300 dark:border-gray-700 text-gray-700
                 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-900
                 focus-visible:ring-gray-500`,
        link: `text-indigo-600 dark:text-indigo-400 underline-offset-4
              hover:underline focus-visible:ring-indigo-500`,
      },
      size: {
        // Mobile-first: Larger touch targets (min 48px for WCAG 2.5.5 Level AA)
        // Desktop: Slightly smaller for more compact UI
        xs: 'h-12 px-3 text-sm sm:h-10 sm:px-2.5 sm:text-xs',
        sm: 'h-12 px-4 text-base sm:h-10 sm:px-3 sm:text-sm',
        md: 'h-12 px-5 text-base sm:h-11 sm:px-4 sm:text-base',
        lg: 'h-14 px-7 text-lg sm:h-12 sm:px-6 sm:text-lg',
        xl: 'h-16 px-9 text-xl sm:h-14 sm:px-8 sm:text-lg',
        icon: 'h-12 w-12 sm:h-11 sm:w-11',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'md',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  loading?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      loading,
      disabled,
      children,
      onClick,
      ...props
    },
    ref
  ) => {
    const createRipple = useRipple();

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      createRipple(e as any);
      onClick?.(e);
    };

    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || loading}
        onClick={handleClick}
        {...props}
      >
        {loading && (
          <span className="inline-block animate-spin mr-2">
            <svg
              className="w-4 h-4"
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
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </span>
        )}
        {children}
      </button>
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
