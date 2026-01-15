import * as React from 'react';
import { cn } from '@/lib/utils';

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    variant?: 'default' | 'outlined' | 'elevated';
    hoverable?: boolean;
    interactive?: boolean;
  }
>(({ className, variant = 'default', hoverable = false, interactive = false, ...props }, ref) => {
  // If card is interactive (has onClick), make it keyboard accessible
  const interactiveProps = interactive || props.onClick ? {
    tabIndex: props.tabIndex ?? 0,
    role: props.role ?? 'button',
  } : {};

  return (
    <div
      ref={ref}
      className={cn(
        'rounded-lg transition-all duration-300 ease-out touch-manipulation',
        variant === 'default' &&
          'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800',
        variant === 'outlined' &&
          'bg-transparent border-2 border-gray-300 dark:border-gray-700',
        variant === 'elevated' &&
          'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-lg hover:shadow-xl',
        hoverable && 'hover:-translate-y-1 hover:shadow-md-3 cursor-pointer',
        (interactive || props.onClick) && 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2',
        className
      )}
      {...interactiveProps}
      {...props}
    />
  );
});
Card.displayName = 'Card';

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      // Mobile-first: More generous padding on mobile
      'px-4 py-3 border-b border-gray-200 dark:border-gray-800',
      'sm:px-6 sm:py-4',
      className
    )}
    {...props}
  />
));
CardHeader.displayName = 'CardHeader';

const CardTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      // Mobile-first: Larger title on mobile for readability
      'text-xl font-semibold text-gray-900 dark:text-white',
      'sm:text-lg',
      className
    )}
    {...props}
  />
));
CardTitle.displayName = 'CardTitle';

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn(
      // Mobile-first: Base text size for better readability on mobile
      'text-base text-gray-600 dark:text-gray-400 mt-1.5',
      'sm:text-sm sm:mt-1',
      className
    )}
    {...props}
  />
));
CardDescription.displayName = 'CardDescription';

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      // Mobile-first: More generous padding on mobile
      'px-4 py-3',
      'sm:px-6 sm:py-4',
      className
    )}
    {...props}
  />
));
CardContent.displayName = 'CardContent';

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      // Mobile-first: More generous padding and spacing on mobile
      'px-4 py-3 border-t border-gray-200 dark:border-gray-800 flex gap-3 justify-end',
      'sm:px-6 sm:py-4 sm:gap-2',
      className
    )}
    {...props}
  />
));
CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
