'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

/**
 * Mobile-First Checkbox Component
 *
 * Features:
 * - Mobile-first responsive design (24px on mobile, 20px on desktop)
 * - 48px minimum touch target on mobile (WCAG 2.5.5 Level AA)
 * - 44px minimum touch target on desktop
 * - Label support with responsive typography
 * - Helper text support
 * - Error states
 * - Indeterminate state
 * - Disabled state
 * - Dark mode support
 * - Touch-optimized with haptic feedback
 * - Keyboard accessible
 */

const checkboxVariants = cva(
  `peer shrink-0 rounded border-2 transition-all duration-200
   focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
   disabled:cursor-not-allowed disabled:opacity-50`,
  {
    variants: {
      variant: {
        default: `border-gray-300 bg-white text-primary-600
                 checked:bg-primary-600 checked:border-primary-600
                 focus-visible:ring-primary-500/30
                 dark:border-gray-700 dark:bg-gray-900
                 dark:checked:bg-primary-600 dark:checked:border-primary-600`,
        error: `border-error-500 bg-white text-error-600
               checked:bg-error-600 checked:border-error-600
               focus-visible:ring-error-500/30
               dark:border-error-600 dark:bg-gray-900`,
      },
      size: {
        default: 'h-6 w-6 sm:h-5 sm:w-5',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface CheckboxProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size' | 'type'>,
    VariantProps<typeof checkboxVariants> {
  label?: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  indeterminate?: boolean;
}

const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  (
    {
      className,
      variant,
      size,
      label,
      helperText,
      error,
      errorMessage,
      indeterminate,
      disabled,
      ...props
    },
    ref
  ) => {
    const finalVariant = error ? 'error' : variant;
    const displayHelperText = error ? errorMessage : helperText;
    const checkboxRef = React.useRef<HTMLInputElement>(null);

    // Merge refs
    React.useImperativeHandle(ref, () => checkboxRef.current!);

    // Handle indeterminate state
    React.useEffect(() => {
      if (checkboxRef.current) {
        checkboxRef.current.indeterminate = indeterminate || false;
      }
    }, [indeterminate]);

    // Handle click with haptic feedback
    const handleClick = () => {
      if ('vibrate' in navigator && !disabled) {
        navigator.vibrate(5);
      }
    };

    return (
      <div className="w-full">
        <div className="flex items-start gap-3 sm:gap-2">
          {/* Checkbox wrapper with minimum touch target */}
          <div
            className="flex items-center justify-center min-w-[48px] min-h-[48px] sm:min-w-[44px] sm:min-h-[44px] -ml-3 sm:-ml-2.5 touch-manipulation"
            onClick={handleClick}
          >
            <input
              ref={checkboxRef}
              type="checkbox"
              className={cn(
                checkboxVariants({ variant: finalVariant, size }),
                'cursor-pointer',
                className
              )}
              disabled={disabled}
              {...props}
            />
          </div>

          {/* Label and helper text */}
          {(label || displayHelperText) && (
            <div className="flex-1 pt-3 sm:pt-2.5">
              {label && (
                <label
                  htmlFor={props.id}
                  className={cn(
                    'block text-base font-medium sm:text-sm',
                    disabled
                      ? 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                      : 'text-gray-900 dark:text-gray-100 cursor-pointer'
                  )}
                  onClick={handleClick}
                >
                  {label}
                  {props.required && (
                    <span className="text-error-500 ml-1">*</span>
                  )}
                </label>
              )}

              {displayHelperText && (
                <p
                  className={cn(
                    'mt-1 text-base sm:text-sm',
                    error
                      ? 'text-error-600 dark:text-error-400'
                      : 'text-gray-600 dark:text-gray-400'
                  )}
                >
                  {displayHelperText}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }
);

Checkbox.displayName = 'Checkbox';

/**
 * CheckboxGroup - for managing multiple checkboxes
 */
export interface CheckboxGroupProps {
  children: React.ReactNode;
  label?: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  className?: string;
}

export function CheckboxGroup({
  children,
  label,
  helperText,
  error,
  errorMessage,
  className,
}: CheckboxGroupProps) {
  const displayHelperText = error ? errorMessage : helperText;

  return (
    <div className={cn('w-full', className)}>
      {/* Group Label */}
      {label && (
        <div className="mb-3 sm:mb-2">
          <p className="text-base font-medium text-gray-900 sm:text-sm dark:text-gray-100">
            {label}
          </p>
        </div>
      )}

      {/* Checkboxes */}
      <div className="space-y-3 sm:space-y-2">{children}</div>

      {/* Helper text or error message */}
      {displayHelperText && (
        <p
          className={cn(
            'mt-2 text-base sm:text-sm',
            error
              ? 'text-error-600 dark:text-error-400'
              : 'text-gray-600 dark:text-gray-400'
          )}
        >
          {displayHelperText}
        </p>
      )}
    </div>
  );
}

export { Checkbox, checkboxVariants };
