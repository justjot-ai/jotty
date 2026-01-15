import * as React from 'react';
import { cn } from '@/lib/utils';

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: string;
  label?: string;
  icon?: React.ReactNode;
  helperText?: string;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, error, label, icon, helperText, id, ...props }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const errorId = error ? `${inputId}-error` : undefined;
    const helperId = helperText ? `${inputId}-helper` : undefined;
    const ariaDescribedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2
                       sm:text-base"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400
                         sm:left-4"
              aria-hidden="true"
            >
              {icon}
            </div>
          )}
          <input
            id={inputId}
            type={type}
            className={cn(
              // Base styles with mobile-first sizing
              `flex w-full rounded-lg border border-gray-300 dark:border-gray-600
               bg-white dark:bg-gray-900 text-gray-900 dark:text-white
               placeholder:text-gray-500 dark:placeholder:text-gray-400
               transition-colors duration-200
               focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400
               focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50`,

              // Mobile-first: Larger touch targets (min 44px for WCAG 2.5.5 Level AA)
              'h-12 px-4 py-3 text-base',

              // Desktop: Slightly smaller, more compact
              'sm:h-10 sm:px-4 sm:py-2 sm:text-sm',

              // Touch manipulation for better mobile performance
              'touch-manipulation',

              // Icon spacing - more space on mobile
              icon && 'pl-11 sm:pl-10',

              // Error state
              error && 'border-red-500 dark:border-red-400 focus:ring-red-500 dark:focus:ring-red-400',

              className
            )}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={ariaDescribedBy}
            ref={ref}
            {...props}
          />
        </div>

        {/* Helper text */}
        {helperText && !error && (
          <p id={helperId} className="mt-1.5 text-xs text-gray-500 dark:text-gray-400 sm:text-sm">
            {helperText}
          </p>
        )}

        {/* Error message */}
        {error && (
          <p
            id={errorId}
            className="mt-1.5 text-sm text-red-600 dark:text-red-400 font-medium
                       sm:text-sm"
            role="alert"
          >
            {error}
          </p>
        )}
      </div>
    );
  }
);
Input.displayName = 'Input';

export { Input };
