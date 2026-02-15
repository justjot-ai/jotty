'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

/**
 * Mobile-First Label Component
 *
 * Features:
 * - Mobile-first responsive typography (base on mobile, sm on desktop)
 * - Support for required indicator
 * - Support for optional indicator
 * - Error state styling
 * - Disabled state styling
 * - Dark mode support
 * - Multiple variants (default, inline, floating)
 * - Helper text support
 */

const labelVariants = cva(
  'font-medium transition-colors duration-200',
  {
    variants: {
      variant: {
        default: 'block mb-2',
        inline: 'inline-block mr-2',
        floating: 'absolute left-4 transition-all duration-200 pointer-events-none',
      },
      size: {
        default: 'text-base sm:text-sm',
        large: 'text-lg sm:text-base',
        small: 'text-sm sm:text-xs',
      },
      state: {
        default: 'text-gray-900 dark:text-gray-100',
        error: 'text-error-600 dark:text-error-400',
        disabled: 'text-gray-400 dark:text-gray-600 cursor-not-allowed',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
      state: 'default',
    },
  }
);

export interface LabelProps
  extends React.LabelHTMLAttributes<HTMLLabelElement>,
    VariantProps<typeof labelVariants> {
  required?: boolean;
  optional?: boolean;
  error?: boolean;
  disabled?: boolean;
  helperText?: string;
}

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  (
    {
      className,
      variant,
      size,
      state,
      required,
      optional,
      error,
      disabled,
      helperText,
      children,
      ...props
    },
    ref
  ) => {
    // Determine state based on props
    const finalState = error ? 'error' : disabled ? 'disabled' : state;

    return (
      <div className="w-full">
        <label
          ref={ref}
          className={cn(
            labelVariants({ variant, size, state: finalState }),
            className
          )}
          {...props}
        >
          {children}
          {required && (
            <span className="text-error-500 dark:text-error-400 ml-1" aria-label="required">
              *
            </span>
          )}
          {optional && !required && (
            <span className="text-gray-500 dark:text-gray-400 ml-1.5 text-sm sm:text-xs font-normal">
              (Optional)
            </span>
          )}
        </label>

        {/* Helper text */}
        {helperText && (
          <p
            className={cn(
              'mt-1 text-base sm:text-sm',
              finalState === 'error'
                ? 'text-error-600 dark:text-error-400'
                : finalState === 'disabled'
                ? 'text-gray-400 dark:text-gray-600'
                : 'text-gray-600 dark:text-gray-400'
            )}
          >
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Label.displayName = 'Label';

/**
 * FieldLabel - Specialized label for form fields with consistent styling
 */
export interface FieldLabelProps extends LabelProps {
  fieldId: string;
}

export const FieldLabel = React.forwardRef<HTMLLabelElement, FieldLabelProps>(
  ({ fieldId, children, ...props }, ref) => {
    return (
      <Label ref={ref} htmlFor={fieldId} {...props}>
        {children}
      </Label>
    );
  }
);

FieldLabel.displayName = 'FieldLabel';

/**
 * FormLabel - Label with specific styling for form sections
 */
export interface FormLabelProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {
  title: string;
  description?: string;
  required?: boolean;
  optional?: boolean;
}

export const FormLabel = React.forwardRef<HTMLDivElement, FormLabelProps>(
  (
    { className, title, description, required, optional, ...props },
    ref
  ) => {
    return (
      <div ref={ref} className={cn('mb-4 sm:mb-3', className)} {...props}>
        <h3 className="text-lg font-semibold text-gray-900 sm:text-base dark:text-gray-100">
          {title}
          {required && (
            <span className="text-error-500 dark:text-error-400 ml-1">*</span>
          )}
          {optional && !required && (
            <span className="text-gray-500 dark:text-gray-400 ml-2 text-base sm:text-sm font-normal">
              (Optional)
            </span>
          )}
        </h3>
        {description && (
          <p className="mt-1 text-base text-gray-600 sm:text-sm dark:text-gray-400">
            {description}
          </p>
        )}
      </div>
    );
  }
);

FormLabel.displayName = 'FormLabel';

export { Label, labelVariants };
