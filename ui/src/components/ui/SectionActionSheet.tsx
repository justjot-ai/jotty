'use client';

import { useState, useEffect } from 'react';
import { BottomSheet, useIsMobile } from '@/components/BottomSheet';
import { triggerHapticFeedback } from '@/lib/swipe-handler';

export interface SectionAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  variant?: 'default' | 'danger' | 'primary';
  disabled?: boolean;
}

export interface SectionActionSheetProps {
  isOpen: boolean;
  onClose: () => void;
  sectionTitle: string;
  sectionType: string;
  sectionIndex: number;
  totalSections: number;
  isBookmarked: boolean;
  // Action handlers
  onToggleBookmark: () => void;
  onDelete: () => void;
  onDuplicate?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onEdit?: () => void;
  onShare?: () => void;
  onCopyContent?: () => void;
  // Custom actions from specific section renderers
  customActions?: SectionAction[];
}

// Icons as inline SVGs for consistency
const Icons = {
  bookmark: (filled: boolean) => (
    <svg className="w-5 h-5" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
    </svg>
  ),
  delete: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  ),
  duplicate: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  ),
  moveUp: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
    </svg>
  ),
  moveDown: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  ),
  edit: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
    </svg>
  ),
  share: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
    </svg>
  ),
  copy: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
    </svg>
  ),
  collection: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
    </svg>
  ),
  menu: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
    </svg>
  ),
};

// Export Icons for use in other components
export { Icons as SectionActionIcons };

// Three-dot menu button to trigger the action sheet
export function SectionMenuButton({
  onClick,
  className = '',
}: {
  onClick: () => void;
  className?: string;
}) {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    triggerHapticFeedback(10);
    onClick();
  };

  return (
    <button
      onClick={handleClick}
      className={`p-2 rounded-lg hover:bg-gray-100 active:bg-gray-200 text-gray-500 hover:text-gray-700 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${className}`}
      aria-label="Section actions"
      title="Section actions"
    >
      {Icons.menu}
    </button>
  );
}

// Action button component used in the action sheet
function ActionButton({
  action,
  onClose,
}: {
  action: SectionAction;
  onClose: () => void;
}) {
  const handleClick = () => {
    triggerHapticFeedback(10);
    action.onClick();
    onClose();
  };

  const variantClasses = {
    default: 'text-gray-700 hover:bg-gray-100 active:bg-gray-200',
    danger: 'text-red-600 hover:bg-red-50 active:bg-red-100',
    primary: 'text-blue-600 hover:bg-blue-50 active:bg-blue-100',
  };

  return (
    <button
      onClick={handleClick}
      disabled={action.disabled}
      className={`w-full flex items-center gap-4 px-4 py-3 text-left transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${variantClasses[action.variant || 'default']}`}
    >
      <span className="flex-shrink-0">{action.icon}</span>
      <span className="font-medium">{action.label}</span>
    </button>
  );
}

// Desktop dropdown menu
function DesktopDropdown({
  isOpen,
  onClose,
  actions,
  position,
}: {
  isOpen: boolean;
  onClose: () => void;
  actions: SectionAction[];
  position: { top: number; right: number };
}) {
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('[data-section-menu]')) {
        onClose();
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('click', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('click', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      data-section-menu
      className="fixed bg-white rounded-xl shadow-xl border border-gray-200 py-2 min-w-[200px] z-50 animate-fadeIn"
      style={{ top: position.top, right: position.right }}
    >
      {actions.map((action, idx) => (
        <ActionButton key={action.id || idx} action={action} onClose={onClose} />
      ))}
    </div>
  );
}

export function SectionActionSheet({
  isOpen,
  onClose,
  sectionTitle,
  sectionType,
  sectionIndex,
  totalSections,
  isBookmarked,
  onToggleBookmark,
  onDelete,
  onDuplicate,
  onMoveUp,
  onMoveDown,
  onEdit,
  onShare,
  onCopyContent,
  customActions = [],
}: SectionActionSheetProps) {
  const isMobile = useIsMobile();

  // Build the list of actions
  const actions: SectionAction[] = [];

  // Bookmark action
  actions.push({
    id: 'bookmark',
    label: isBookmarked ? 'Remove Bookmark' : 'Add Bookmark',
    icon: Icons.bookmark(isBookmarked),
    onClick: onToggleBookmark,
    variant: isBookmarked ? 'primary' : 'default',
  });

  // Edit action
  if (onEdit) {
    actions.push({
      id: 'edit',
      label: 'Edit Section',
      icon: Icons.edit,
      onClick: onEdit,
    });
  }

  // Copy content action
  if (onCopyContent) {
    actions.push({
      id: 'copy',
      label: 'Copy Content',
      icon: Icons.copy,
      onClick: onCopyContent,
    });
  }

  // Duplicate action
  if (onDuplicate) {
    actions.push({
      id: 'duplicate',
      label: 'Duplicate Section',
      icon: Icons.duplicate,
      onClick: onDuplicate,
    });
  }

  // Move up action
  if (onMoveUp) {
    actions.push({
      id: 'moveUp',
      label: 'Move Up',
      icon: Icons.moveUp,
      onClick: onMoveUp,
      disabled: sectionIndex === 0,
    });
  }

  // Move down action
  if (onMoveDown) {
    actions.push({
      id: 'moveDown',
      label: 'Move Down',
      icon: Icons.moveDown,
      onClick: onMoveDown,
      disabled: sectionIndex === totalSections - 1,
    });
  }

  // Share action
  if (onShare) {
    actions.push({
      id: 'share',
      label: 'Share Section',
      icon: Icons.share,
      onClick: onShare,
    });
  }

  // Add custom actions
  actions.push(...customActions);

  // Delete action (always last)
  actions.push({
    id: 'delete',
    label: 'Delete Section',
    icon: Icons.delete,
    onClick: onDelete,
    variant: 'danger',
  });

  // Mobile: Use BottomSheet
  if (isMobile) {
    return (
      <BottomSheet
        isOpen={isOpen}
        onClose={onClose}
        title={sectionTitle}
        maxHeight="70vh"
      >
        <div className="pb-4">
          {/* Section info */}
          <div className="px-4 pb-3 mb-2 border-b border-gray-100">
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
              {sectionType}
            </span>
            <span className="text-xs text-gray-500 ml-2">
              Section {sectionIndex + 1} of {totalSections}
            </span>
          </div>

          {/* Actions */}
          <div className="divide-y divide-gray-100">
            {actions.map((action, idx) => (
              <ActionButton key={action.id || idx} action={action} onClose={onClose} />
            ))}
          </div>
        </div>
      </BottomSheet>
    );
  }

  // Desktop: Render nothing here (dropdown is rendered separately via portal or inline)
  // The desktop version is handled by the parent component positioning
  return null;
}

// Hook to manage action sheet state
export function useSectionActionSheet() {
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, right: 0 });
  const isMobile = useIsMobile();

  const open = (buttonRef?: React.RefObject<HTMLButtonElement | null>) => {
    if (!isMobile && buttonRef?.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + 4,
        right: window.innerWidth - rect.right,
      });
    }
    setIsOpen(true);
  };

  const close = () => setIsOpen(false);
  const toggle = (buttonRef?: React.RefObject<HTMLButtonElement | null>) => {
    if (isOpen) {
      close();
    } else {
      open(buttonRef);
    }
  };

  return {
    isOpen,
    open,
    close,
    toggle,
    dropdownPosition,
    isMobile,
  };
}

// Combined component for easy use
export function SectionActionsMenu({
  sectionTitle,
  sectionType,
  sectionIndex,
  totalSections,
  isBookmarked,
  onToggleBookmark,
  onDelete,
  onDuplicate,
  onMoveUp,
  onMoveDown,
  onEdit,
  onShare,
  onCopyContent,
  customActions = [],
  className = '',
}: Omit<SectionActionSheetProps, 'isOpen' | 'onClose'> & { className?: string }) {
  const { isOpen, close, toggle, dropdownPosition, isMobile } = useSectionActionSheet();

  // Build actions list (same as in SectionActionSheet)
  const actions: SectionAction[] = [];

  actions.push({
    id: 'bookmark',
    label: isBookmarked ? 'Remove Bookmark' : 'Add Bookmark',
    icon: Icons.bookmark(isBookmarked),
    onClick: onToggleBookmark,
    variant: isBookmarked ? 'primary' : 'default',
  });

  if (onEdit) {
    actions.push({
      id: 'edit',
      label: 'Edit Section',
      icon: Icons.edit,
      onClick: onEdit,
    });
  }

  if (onCopyContent) {
    actions.push({
      id: 'copy',
      label: 'Copy Content',
      icon: Icons.copy,
      onClick: onCopyContent,
    });
  }

  if (onDuplicate) {
    actions.push({
      id: 'duplicate',
      label: 'Duplicate Section',
      icon: Icons.duplicate,
      onClick: onDuplicate,
    });
  }

  if (onMoveUp) {
    actions.push({
      id: 'moveUp',
      label: 'Move Up',
      icon: Icons.moveUp,
      onClick: onMoveUp,
      disabled: sectionIndex === 0,
    });
  }

  if (onMoveDown) {
    actions.push({
      id: 'moveDown',
      label: 'Move Down',
      icon: Icons.moveDown,
      onClick: onMoveDown,
      disabled: sectionIndex === totalSections - 1,
    });
  }

  if (onShare) {
    actions.push({
      id: 'share',
      label: 'Share Section',
      icon: Icons.share,
      onClick: onShare,
    });
  }

  actions.push(...customActions);

  actions.push({
    id: 'delete',
    label: 'Delete Section',
    icon: Icons.delete,
    onClick: onDelete,
    variant: 'danger',
  });

  return (
    <>
      <SectionMenuButton onClick={() => toggle()} className={className} />

      {/* Mobile: BottomSheet */}
      {isMobile && (
        <SectionActionSheet
          isOpen={isOpen}
          onClose={close}
          sectionTitle={sectionTitle}
          sectionType={sectionType}
          sectionIndex={sectionIndex}
          totalSections={totalSections}
          isBookmarked={isBookmarked}
          onToggleBookmark={onToggleBookmark}
          onDelete={onDelete}
          onDuplicate={onDuplicate}
          onMoveUp={onMoveUp}
          onMoveDown={onMoveDown}
          onEdit={onEdit}
          onShare={onShare}
          onCopyContent={onCopyContent}
          customActions={customActions}
        />
      )}

      {/* Desktop: Dropdown */}
      {!isMobile && (
        <DesktopDropdown
          isOpen={isOpen}
          onClose={close}
          actions={actions}
          position={dropdownPosition}
        />
      )}
    </>
  );
}
