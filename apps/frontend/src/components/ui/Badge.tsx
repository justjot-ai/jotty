import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  `inline-flex items-center rounded-full font-semibold transition-colors
   touch-manipulation`,
  {
    variants: {
      variant: {
        default:
          'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400',
        secondary:
          'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300',
        success:
          'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400',
        warning:
          'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400',
        destructive:
          'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400',
        outline:
          'border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300',
      },
      size: {
        // Mobile-first: Larger, more readable text and padding
        sm: 'px-2.5 py-1 text-xs sm:px-2 sm:py-0.5 sm:text-[10px]',
        md: 'px-3 py-1.5 text-sm sm:px-2.5 sm:py-1 sm:text-xs',
        lg: 'px-4 py-2 text-base sm:px-3 sm:py-1.5 sm:text-sm',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'md',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, size, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant, size }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
