'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

/**
 * Mobile-First Select Component
 *
 * Features:
 * - Mobile-first responsive design (larger on mobile, smaller on desktop)
 * - 48px minimum height on mobile (WCAG 2.5.5 Level AA)
 * - 40px height on desktop (sm: breakpoint)
 * - Error states with visual feedback
 * - Helper text support
 * - Label integration
 * - Disabled state
 * - Dark mode support
 * - Touch-optimized
 */

const selectVariants = cva(
  `w-full rounded-md border font-medium transition-all duration-200
   focus:outline-none focus:ring-2 focus:ring-offset-0
   disabled:cursor-not-allowed disabled:opacity-50
   touch-manipulation appearance-none bg-no-repeat`,
  {
    variants: {
      variant: {
        default: `border-gray-300 bg-white text-gray-900
                 focus:border-primary-500 focus:ring-primary-500/20
                 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100
                 dark:focus:border-primary-500`,
        error: `border-error-500 bg-white text-gray-900
               focus:border-error-600 focus:ring-error-500/20
               dark:border-error-600 dark:bg-gray-900 dark:text-gray-100`,
      },
      selectSize: {
        default: `h-12 px-4 py-3 text-base pr-10
                 sm:h-10 sm:px-4 sm:py-2 sm:text-sm sm:pr-9`,
      },
    },
    defaultVariants: {
      variant: 'default',
      selectSize: 'default',
    },
  }
);

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'>,
    VariantProps<typeof selectVariants> {
  label?: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  options: Array<{
    value: string;
    label: string;
    disabled?: boolean;
  }>;
  placeholder?: string;
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      className,
      variant,
      selectSize,
      label,
      helperText,
      error,
      errorMessage,
      options,
      placeholder,
      disabled,
      ...props
    },
    ref
  ) => {
    const finalVariant = error ? 'error' : variant;
    const displayHelperText = error ? errorMessage : helperText;

    return (
      <div className="w-full">
        {/* Label */}
        {label && (
          <label
            htmlFor={props.id}
            className="block mb-2 text-base font-medium text-gray-900 sm:text-sm dark:text-gray-100"
          >
            {label}
            {props.required && <span className="text-error-500 ml-1">*</span>}
          </label>
        )}

        {/* Select Container */}
        <div className="relative">
          <select
            ref={ref}
            className={cn(
              selectVariants({ variant: finalVariant, selectSize }),
              className
            )}
            disabled={disabled}
            style={{
              backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
              backgroundPosition: 'right 0.75rem center',
              backgroundSize: '1.5em 1.5em',
            }}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))}
          </select>

          {/* Focus indicator for better accessibility */}
          <div className="pointer-events-none absolute inset-0 rounded-md ring-0 transition-all duration-200" />
        </div>

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
);

Select.displayName = 'Select';

export { Select, selectVariants };
