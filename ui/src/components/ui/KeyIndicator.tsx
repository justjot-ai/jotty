'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

export type KeyType = 'primary' | 'foreign' | 'composite';

interface KeyIndicatorProps {
  type: KeyType;
  id: string;
  label?: string;
  referencedCollection?: string;
  showFullId?: boolean;
  size?: 'sm' | 'md';
  className?: string;
}

/**
 * KeyIndicator - Visual indicator for primary and foreign keys
 *
 * Shows a compact badge indicating the key type with a truncated ID.
 * Hover reveals the full ID and any referenced collection info.
 */
export function KeyIndicator({
  type,
  id,
  label,
  referencedCollection,
  showFullId = false,
  size = 'sm',
  className,
}: KeyIndicatorProps) {
  const [showTooltip, setShowTooltip] = React.useState(false);

  const formatId = (id: string): string => {
    if (!id) return '-';
    if (showFullId || id.length <= 8) return id;
    return `${id.slice(0, 4)}...${id.slice(-4)}`;
  };

  const getKeyIcon = (): string => {
    switch (type) {
      case 'primary':
        return '🔑';
      case 'foreign':
        return '🔗';
      case 'composite':
        return '🔐';
      default:
        return '🔑';
    }
  };

  const getKeyStyles = (): string => {
    switch (type) {
      case 'primary':
        return 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100';
      case 'foreign':
        return 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100';
      case 'composite':
        return 'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100';
    }
  };

  const getKeyLabel = (): string => {
    if (label) return label;
    switch (type) {
      case 'primary':
        return 'PK';
      case 'foreign':
        return 'FK';
      case 'composite':
        return 'CK';
      default:
        return 'KEY';
    }
  };

  const sizeStyles = size === 'sm'
    ? 'text-[10px] px-1.5 py-0.5 gap-0.5'
    : 'text-xs px-2 py-1 gap-1';

  return (
    <div className="relative inline-flex">
      <button
        type="button"
        className={cn(
          'inline-flex items-center rounded border font-mono transition-colors cursor-help',
          sizeStyles,
          getKeyStyles(),
          className
        )}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          navigator.clipboard.writeText(id);
        }}
        title="Click to copy ID"
      >
        <span className="flex-shrink-0">{getKeyIcon()}</span>
        <span className="font-semibold">{getKeyLabel()}</span>
        <span className="opacity-70">{formatId(id)}</span>
      </button>

      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute bottom-full left-0 mb-1 z-[55] animate-in fade-in duration-150">
          <div className="bg-gray-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg max-w-xs">
            <div className="font-medium mb-1">
              {type === 'primary' ? 'Primary Key' : type === 'foreign' ? 'Foreign Key' : 'Composite Key'}
            </div>
            <div className="font-mono text-[10px] text-gray-300 break-all">
              {id}
            </div>
            {referencedCollection && (
              <div className="mt-1 text-gray-400">
                → References: <span className="text-blue-300">{referencedCollection}</span>
              </div>
            )}
            <div className="mt-1 text-gray-500 text-[10px]">
              Click to copy
            </div>
          </div>
          <div className="w-2 h-2 bg-gray-900 rotate-45 absolute left-3 -bottom-1" />
        </div>
      )}
    </div>
  );
}

interface KeyIndicatorGroupProps {
  keys: Array<{
    type: KeyType;
    id: string;
    label?: string;
    referencedCollection?: string;
  }>;
  size?: 'sm' | 'md';
  className?: string;
}

/**
 * KeyIndicatorGroup - Display multiple key indicators in a row
 */
export function KeyIndicatorGroup({ keys, size = 'sm', className }: KeyIndicatorGroupProps) {
  if (!keys || keys.length === 0) return null;

  return (
    <div className={cn('flex flex-wrap items-center gap-1', className)}>
      {keys.map((key, index) => (
        <KeyIndicator
          key={`${key.type}-${key.id}-${index}`}
          type={key.type}
          id={key.id}
          label={key.label}
          referencedCollection={key.referencedCollection}
          size={size}
        />
      ))}
    </div>
  );
}
