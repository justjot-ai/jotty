import React from 'react';

export interface SkeletonProps {
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  className?: string;
  lines?: number; // For text skeletons
  animate?: 'pulse' | 'shimmer' | 'none';
}

/**
 * Skeleton loading component with multiple variants
 * Used to show placeholder content while data is loading
 */
export function Skeleton({
  variant = 'rectangular',
  width = '100%',
  height = 20,
  className = '',
  lines = 1,
  animate = 'shimmer',
}: SkeletonProps) {
  const baseClasses = 'bg-gray-200 dark:bg-gray-700';

  const animationClasses = {
    pulse: 'animate-pulse',
    shimmer: 'animate-shimmer bg-shimmer-gradient bg-[length:1000px_100%]',
    none: '',
  };

  const variantClasses = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: '',
    rounded: 'rounded-lg',
  };

  const widthStyle = typeof width === 'number' ? `${width}px` : width;
  const heightStyle = typeof height === 'number' ? `${height}px` : height;

  // For text skeletons with multiple lines
  if (variant === 'text' && lines > 1) {
    return (
      <div className={`space-y-2 ${className}`}>
        {Array.from({ length: lines }).map((_, i) => {
          // Make last line shorter (80% width)
          const isLast = i === lines - 1;
          const lineWidth = isLast ? '80%' : widthStyle;

          return (
            <div
              key={i}
              className={`${baseClasses} ${variantClasses[variant]} ${animationClasses[animate]}`}
              style={{ width: lineWidth, height: heightStyle }}
            />
          );
        })}
      </div>
    );
  }

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${animationClasses[animate]} ${className}`}
      style={{ width: widthStyle, height: heightStyle }}
    />
  );
}

/**
 * Skeleton component for card layouts
 */
export function SkeletonCard({ className = '' }: { className?: string }) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-100 dark:border-gray-700 overflow-hidden ${className}`}>
      <div className="p-3 border-b border-gray-50 dark:border-gray-700 flex items-center gap-2">
        <Skeleton variant="circular" width={32} height={32} />
        <div className="flex-1">
          <Skeleton variant="text" width="60%" height={14} className="mb-1.5" />
          <Skeleton variant="text" width="40%" height={10} />
        </div>
      </div>
      <div className="p-3">
        <Skeleton variant="rounded" height={64} />
      </div>
    </div>
  );
}

/**
 * Skeleton component for list items
 */
export function SkeletonListItem({ className = '' }: { className?: string }) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-100 dark:border-gray-700 p-3 flex items-center gap-3 ${className}`}>
      <Skeleton variant="circular" width={40} height={40} />
      <div className="flex-1">
        <Skeleton variant="text" width="70%" height={14} className="mb-2" />
        <Skeleton variant="text" width="50%" height={10} />
      </div>
      <Skeleton variant="rounded" width={60} height={24} />
    </div>
  );
}

/**
 * Skeleton component for stat cards
 */
export function SkeletonStatCard({ className = '' }: { className?: string }) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-100 dark:border-gray-700 p-4 ${className}`}>
      <div className="flex items-center gap-3 mb-3">
        <Skeleton variant="circular" width={40} height={40} />
        <Skeleton variant="text" width="50%" height={14} />
      </div>
      <Skeleton variant="text" width="30%" height={24} className="mb-2" />
      <Skeleton variant="text" width="60%" height={10} />
    </div>
  );
}

/**
 * Skeleton component for table rows
 */
export function SkeletonTableRow({ columns = 4, className = '' }: { columns?: number; className?: string }) {
  return (
    <div className={`border-b border-gray-100 dark:border-gray-700 py-3 flex items-center gap-4 ${className}`}>
      {Array.from({ length: columns }).map((_, i) => (
        <div key={i} className="flex-1">
          <Skeleton variant="text" height={14} />
        </div>
      ))}
    </div>
  );
}

/**
 * Skeleton component for avatar with text
 */
export function SkeletonAvatar({
  size = 40,
  withText = true,
  className = ''
}: {
  size?: number;
  withText?: boolean;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <Skeleton variant="circular" width={size} height={size} />
      {withText && (
        <div className="flex-1">
          <Skeleton variant="text" width="60%" height={14} className="mb-1" />
          <Skeleton variant="text" width="40%" height={10} />
        </div>
      )}
    </div>
  );
}

/**
 * Skeleton component for chart placeholders
 */
export function SkeletonChart({
  height = 200,
  className = ''
}: {
  height?: number;
  className?: string;
}) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-100 dark:border-gray-700 p-4 ${className}`}>
      <Skeleton variant="text" width="40%" height={16} className="mb-4" />
      <Skeleton variant="rounded" height={height} />
    </div>
  );
}

/**
 * Skeleton for full page loading
 */
export function SkeletonPage({ className = '' }: { className?: string }) {
  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <Skeleton variant="text" width={200} height={32} />
        <Skeleton variant="rounded" width={120} height={40} />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <SkeletonStatCard />
        <SkeletonStatCard />
        <SkeletonStatCard />
        <SkeletonStatCard />
      </div>

      {/* Content Area */}
      <div className="space-y-3">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    </div>
  );
}
