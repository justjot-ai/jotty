'use client';

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { useRipple } from '@/hooks/useRipple';

/**
 * Material Design 3 Floating Action Button (FAB)
 *
 * Features:
 * - Multiple sizes (small, medium, large)
 * - Extended variant with label
 * - Primary, secondary, tertiary, and surface variants
 * - Elevation on hover
 * - Ripple effect animation
 * - 44x44px minimum touch target for accessibility
 * - Mobile-optimized with haptic feedback support
 */

const fabVariants = cva(
  `inline-flex items-center justify-center font-medium transition-all duration-300
   focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-offset-0
   disabled:opacity-38 disabled:cursor-not-allowed disabled:pointer-events-none
   relative overflow-hidden active:scale-95 touch-manipulation`,
  {
    variants: {
      variant: {
        primary: `bg-primary-500 text-white hover:bg-primary-600
                 shadow-md-3 hover:shadow-md-4 focus-visible:ring-primary-500/30
                 dark:bg-primary-600 dark:hover:bg-primary-700`,
        secondary: `bg-secondary-500 text-white hover:bg-secondary-600
                   shadow-md-3 hover:shadow-md-4 focus-visible:ring-secondary-500/30
                   dark:bg-secondary-600 dark:hover:bg-secondary-700`,
        tertiary: `bg-surface-3 text-primary-700 hover:bg-surface-4
                  shadow-md-2 hover:shadow-md-3 focus-visible:ring-primary-500/30
                  dark:bg-gray-800 dark:text-primary-400 dark:hover:bg-gray-700`,
        surface: `bg-surface-1 text-primary-600 hover:bg-surface-2
                 shadow-md-2 hover:shadow-md-3 focus-visible:ring-primary-500/30
                 dark:bg-gray-900 dark:text-primary-400 dark:hover:bg-gray-800`,
      },
      size: {
        small: 'h-12 w-12 rounded-md-sm min-h-[48px] min-w-[48px] sm:h-11 sm:w-11 sm:min-h-[44px] sm:min-w-[44px]',
        medium: 'h-16 w-16 rounded-md-lg min-h-[64px] min-w-[64px] sm:h-14 sm:w-14 sm:min-h-[56px] sm:min-w-[56px]',
        large: 'h-28 w-28 rounded-md-xl min-h-[112px] min-w-[112px] sm:h-24 sm:w-24 sm:min-h-[96px] sm:min-w-[96px]',
        extended: 'h-16 px-5 rounded-md-lg gap-3 min-h-[64px] sm:h-14 sm:px-4 sm:gap-2 sm:min-h-[56px]',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'medium',
    },
  }
);

const fabIconVariants = cva('transition-transform', {
  variants: {
    size: {
      small: 'w-7 h-7 sm:w-6 sm:h-6',
      medium: 'w-7 h-7 sm:w-6 sm:h-6',
      large: 'w-11 h-11 sm:w-9 sm:h-9',
      extended: 'w-7 h-7 sm:w-6 sm:h-6',
    },
  },
  defaultVariants: {
    size: 'medium',
  },
});

export interface FABProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof fabVariants> {
  icon?: React.ReactNode;
  label?: string;
  loading?: boolean;
  hideOnScroll?: boolean;
  position?: 'bottom-right' | 'bottom-left' | 'bottom-center' | 'top-right' | 'top-left';
}

const FAB = React.forwardRef<HTMLButtonElement, FABProps>(
  (
    {
      className,
      variant,
      size,
      icon,
      label,
      loading,
      disabled,
      hideOnScroll = false,
      position = 'bottom-right',
      children,
      onClick,
      ...props
    },
    ref
  ) => {
    const [isVisible, setIsVisible] = React.useState(true);
    const lastScrollY = React.useRef(0);
    const createRipple = useRipple('rgba(255, 255, 255, 0.3)');

    // Handle hide on scroll
    React.useEffect(() => {
      if (!hideOnScroll) return;

      const handleScroll = () => {
        const currentScrollY = window.scrollY;

        if (currentScrollY > lastScrollY.current && currentScrollY > 100) {
          setIsVisible(false);
        } else {
          setIsVisible(true);
        }

        lastScrollY.current = currentScrollY;
      };

      window.addEventListener('scroll', handleScroll, { passive: true });
      return () => window.removeEventListener('scroll', handleScroll);
    }, [hideOnScroll]);

    // Handle click with optional haptic feedback
    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      // Trigger haptic feedback on mobile devices
      if ('vibrate' in navigator) {
        navigator.vibrate(10);
      }

      createRipple(e as any);
      onClick?.(e);
    };

    // Extended FAB forces extended size
    const finalSize = label ? 'extended' : size;

    // Position classes for fixed positioning
    const positionClasses = {
      'bottom-right': 'fixed bottom-4 right-4 md:bottom-6 md:right-6',
      'bottom-left': 'fixed bottom-4 left-4 md:bottom-6 md:left-6',
      'bottom-center': 'fixed bottom-4 left-1/2 -translate-x-1/2 md:bottom-6',
      'top-right': 'fixed top-4 right-4 md:top-6 md:right-6',
      'top-left': 'fixed top-4 left-4 md:top-6 md:left-6',
    };

    return (
      <button
        ref={ref}
        className={cn(
          fabVariants({ variant, size: finalSize }),
          isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-0',
          'transition-all duration-200',
          className
        )}
        disabled={disabled || loading}
        onClick={handleClick}
        aria-label={label || props['aria-label']}
        {...props}
      >
        {/* Loading spinner */}
        {loading && (
          <span className={cn(fabIconVariants({ size: finalSize }), 'animate-spin')}>
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

        {/* Icon */}
        {!loading && icon && (
          <span className={fabIconVariants({ size: finalSize })}>
            {icon}
          </span>
        )}

        {/* Label for extended FAB */}
        {!loading && label && (
          <span className="text-base font-semibold whitespace-nowrap sm:text-sm">
            {label}
          </span>
        )}

        {/* Children (for custom content) */}
        {!loading && !icon && !label && children}
      </button>
    );
  }
);

FAB.displayName = 'FAB';

/**
 * Fixed FAB with position
 * Wraps the FAB component with fixed positioning
 */
export const FixedFAB = React.forwardRef<HTMLButtonElement, FABProps>(
  ({ position = 'bottom-right', className, ...props }, ref) => {
    const positionClasses = {
      'bottom-right': 'fixed bottom-4 right-4 md:bottom-6 md:right-6 z-[45]',
      'bottom-left': 'fixed bottom-4 left-4 md:bottom-6 md:left-6 z-[45]',
      'bottom-center': 'fixed bottom-4 left-1/2 -translate-x-1/2 md:bottom-6 z-[45]',
      'top-right': 'fixed top-4 right-4 md:top-6 md:right-6 z-[45]',
      'top-left': 'fixed top-4 left-4 md:top-6 md:left-6 z-[45]',
    };

    return (
      <div className={positionClasses[position]}>
        <FAB ref={ref} className={className} {...props} />
      </div>
    );
  }
);

FixedFAB.displayName = 'FixedFAB';

export { FAB, fabVariants };
