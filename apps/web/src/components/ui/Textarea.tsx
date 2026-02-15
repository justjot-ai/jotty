import * as React from 'react';
import { cn } from '@/lib/utils';

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: string;
  label?: string;
  helperText?: string;
  showCharCount?: boolean;
  maxLength?: number;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error, label, helperText, showCharCount, maxLength, id, ...props }, ref) => {
    const textareaId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`;
    const errorId = error ? `${textareaId}-error` : undefined;
    const helperId = helperText ? `${textareaId}-helper` : undefined;
    const ariaDescribedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

    const [charCount, setCharCount] = React.useState(0);

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      if (showCharCount) {
        setCharCount(e.target.value.length);
      }
      props.onChange?.(e);
    };

    React.useEffect(() => {
      if (showCharCount && props.value) {
        setCharCount(String(props.value).length);
      }
    }, [props.value, showCharCount]);

    return (
      <div className="w-full">
        {/* Label and character count */}
        <div className="flex items-center justify-between mb-2">
          {label && (
            <label
              htmlFor={textareaId}
              className="block text-sm font-medium text-gray-700 dark:text-gray-300
                         sm:text-base"
            >
              {label}
            </label>
          )}
          {showCharCount && (
            <span
              className="text-xs text-gray-500 dark:text-gray-400
                         sm:text-sm"
              aria-live="polite"
              aria-atomic="true"
            >
              {charCount}
              {maxLength && ` / ${maxLength}`}
            </span>
          )}
        </div>

        <textarea
          id={textareaId}
          maxLength={maxLength}
          className={cn(
            // Base styles with mobile-first sizing
            `flex w-full rounded-lg border border-gray-300 dark:border-gray-600
             bg-white dark:bg-gray-900 text-gray-900 dark:text-white
             placeholder:text-gray-500 dark:placeholder:text-gray-400
             transition-colors duration-200
             focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400
             focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50
             resize-vertical`,

            // Mobile-first: Larger touch targets and font size
            'min-h-[120px] px-4 py-3 text-base',

            // Desktop: Slightly smaller, more compact
            'sm:min-h-[100px] sm:px-4 sm:py-3 sm:text-sm',

            // Touch manipulation for better mobile performance
            'touch-manipulation',

            // Error state
            error && 'border-red-500 dark:border-red-400 focus:ring-red-500 dark:focus:ring-red-400',

            className
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={ariaDescribedBy}
          ref={ref}
          onChange={handleChange}
          {...props}
        />

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
Textarea.displayName = 'Textarea';

export { Textarea };
