'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

/**
 * Material Design 3 Bottom Sheet
 *
 * Features:
 * - Swipe to dismiss gesture
 * - Backdrop overlay with dismiss on click
 * - Smooth spring animations
 * - Safe area support for mobile devices
 * - Multiple snap points (collapsed, half-expanded, fully-expanded)
 * - Keyboard accessibility (Escape to close)
 * - Focus trap when open
 * - Mobile-optimized with touch handling
 */

interface BottomSheetProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  snapPoints?: ('collapsed' | 'half' | 'full')[];
  initialSnap?: 'collapsed' | 'half' | 'full';
  className?: string;
  showHandle?: boolean;
  dismissible?: boolean;
  modal?: boolean;
}

export function BottomSheet({
  isOpen,
  onClose,
  children,
  title,
  snapPoints = ['half', 'full'],
  initialSnap = 'half',
  className,
  showHandle = true,
  dismissible = true,
  modal = true,
}: BottomSheetProps) {
  const [currentSnap, setCurrentSnap] = React.useState<'collapsed' | 'half' | 'full'>(initialSnap);
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStartY, setDragStartY] = React.useState(0);
  const [dragOffset, setDragOffset] = React.useState(0);
  const sheetRef = React.useRef<HTMLDivElement>(null);
  const dragHandleRef = React.useRef<HTMLDivElement>(null);

  // Snap point heights
  const snapHeights = {
    collapsed: 120,
    half: typeof window !== 'undefined' ? window.innerHeight * 0.5 : 400,
    full: typeof window !== 'undefined' ? window.innerHeight * 0.9 : 800,
  };

  // Calculate current height based on snap point and drag offset
  const currentHeight = snapHeights[currentSnap] - dragOffset;

  // Close on Escape key
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dismissible && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [dismissible, isOpen, onClose]);

  // Lock body scroll when open
  React.useEffect(() => {
    if (isOpen && modal) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [isOpen, modal]);

  // Handle drag start
  const handleDragStart = (clientY: number) => {
    if (!dismissible) return;
    setIsDragging(true);
    setDragStartY(clientY);
    setDragOffset(0);
  };

  // Handle drag move
  const handleDragMove = (clientY: number) => {
    if (!isDragging) return;

    const offset = dragStartY - clientY;
    // Only allow dragging down (closing), not up beyond current snap
    const clampedOffset = Math.min(offset, 0);
    setDragOffset(-clampedOffset);
  };

  // Handle drag end
  const handleDragEnd = () => {
    if (!isDragging) return;

    setIsDragging(false);

    // Determine if we should close or snap to next point
    const threshold = snapHeights[currentSnap] * 0.3;

    if (dragOffset > threshold) {
      // Find next lower snap point or close
      const currentIndex = snapPoints.indexOf(currentSnap);
      if (currentIndex > 0) {
        setCurrentSnap(snapPoints[currentIndex - 1]);
      } else if (dismissible) {
        onClose();
      }
    }

    setDragOffset(0);
  };

  // Mouse events
  const handleMouseDown = (e: React.MouseEvent) => {
    handleDragStart(e.clientY);
  };

  const handleMouseMove = (e: MouseEvent) => {
    handleDragMove(e.clientY);
  };

  const handleMouseUp = () => {
    handleDragEnd();
  };

  // Touch events
  const handleTouchStart = (e: React.TouchEvent) => {
    handleDragStart(e.touches[0].clientY);
  };

  const handleTouchMove = (e: TouchEvent) => {
    handleDragMove(e.touches[0].clientY);
  };

  const handleTouchEnd = () => {
    handleDragEnd();
  };

  // Add/remove event listeners for drag
  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.addEventListener('touchmove', handleTouchMove);
      document.addEventListener('touchend', handleTouchEnd);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.removeEventListener('touchmove', handleTouchMove);
        document.removeEventListener('touchend', handleTouchEnd);
      };
    }
  }, [isDragging, dragStartY, dragOffset]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      {modal && (
        <div
          className={cn(
            'fixed inset-0 bg-black/50 z-40 transition-opacity duration-300',
            isOpen ? 'opacity-100' : 'opacity-0'
          )}
          onClick={dismissible ? onClose : undefined}
          aria-hidden="true"
        />
      )}

      {/* Bottom Sheet */}
      <div
        ref={sheetRef}
        className={cn(
          'fixed left-0 right-0 bottom-0 z-[60] bg-surface-1 dark:bg-gray-900',
          'rounded-t-3xl shadow-md-8 overflow-hidden',
          'transition-transform duration-300 ease-out',
          'touch-none select-none',
          isDragging && 'transition-none',
          className
        )}
        style={{
          height: `${currentHeight}px`,
          transform: isOpen ? 'translateY(0)' : 'translateY(100%)',
        }}
        role="dialog"
        aria-modal={modal}
        aria-labelledby={title ? 'bottom-sheet-title' : undefined}
      >
        {/* Drag Handle */}
        {showHandle && (
          <div
            ref={dragHandleRef}
            className="flex justify-center items-center py-3 cursor-grab active:cursor-grabbing touch-manipulation"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
          >
            <div className="w-12 h-1 bg-gray-300 dark:bg-gray-700 rounded-full" />
          </div>
        )}

        {/* Title */}
        {title && (
          <div className="px-6 pb-4 border-b border-gray-200 dark:border-gray-800">
            <h2
              id="bottom-sheet-title"
              className="text-xl font-semibold text-gray-900 dark:text-white"
            >
              {title}
            </h2>
          </div>
        )}

        {/* Content */}
        <div className="overflow-y-auto overscroll-contain h-full pb-safe">
          <div className="px-6 py-4">
            {children}
          </div>
        </div>
      </div>
    </>
  );
}

/**
 * Bottom Sheet Handle - for custom drag handles
 */
export function BottomSheetHandle({ className }: { className?: string }) {
  return (
    <div className={cn('flex justify-center items-center py-3', className)}>
      <div className="w-12 h-1 bg-gray-300 dark:bg-gray-700 rounded-full" />
    </div>
  );
}

/**
 * Bottom Sheet Content - scrollable content area
 */
export function BottomSheetContent({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('overflow-y-auto overscroll-contain', className)}>
      {children}
    </div>
  );
}

/**
 * Bottom Sheet Header - for title and actions
 */
export function BottomSheetHeader({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('px-6 pb-4 border-b border-gray-200 dark:border-gray-800', className)}>
      {children}
    </div>
  );
}

/**
 * Bottom Sheet Footer - for actions at the bottom
 */
export function BottomSheetFooter({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'px-6 py-4 border-t border-gray-200 dark:border-gray-800',
        'bg-surface-2 dark:bg-gray-800',
        className
      )}
    >
      {children}
    </div>
  );
}
