'use client';

import { createContext, useContext, useState, useCallback, ReactNode, useEffect, useRef } from 'react';
import { useIsMobile } from '@/components/BottomSheet';

// Types for dialog options
interface AlertOptions {
  title?: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  buttonText?: string;
}

interface ConfirmOptions {
  title?: string;
  message: string;
  type?: 'danger' | 'warning' | 'info';
  confirmText?: string;
  cancelText?: string;
}

interface DialogState {
  isOpen: boolean;
  type: 'alert' | 'confirm';
  options: AlertOptions | ConfirmOptions;
  resolve?: (value: boolean) => void;
}

interface DialogContextType {
  alert: (options: AlertOptions | string) => Promise<void>;
  confirm: (options: ConfirmOptions | string) => Promise<boolean>;
}

const DialogContext = createContext<DialogContextType | null>(null);

export function useDialog() {
  const context = useContext(DialogContext);
  if (!context) {
    throw new Error('useDialog must be used within a DialogProvider');
  }
  return context;
}

// Icon components
function SuccessIcon() {
  return (
    <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
      <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    </div>
  );
}

function ErrorIcon() {
  return (
    <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
      <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    </div>
  );
}

function WarningIcon() {
  return (
    <div className="w-12 h-12 rounded-full bg-amber-100 flex items-center justify-center">
      <svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </div>
  );
}

function InfoIcon() {
  return (
    <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    </div>
  );
}

function DangerIcon() {
  return (
    <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
      <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
      </svg>
    </div>
  );
}

function getIcon(type: string, dialogType: 'alert' | 'confirm') {
  if (dialogType === 'confirm') {
    switch (type) {
      case 'danger': return <DangerIcon />;
      case 'warning': return <WarningIcon />;
      default: return <InfoIcon />;
    }
  }
  switch (type) {
    case 'success': return <SuccessIcon />;
    case 'error': return <ErrorIcon />;
    case 'warning': return <WarningIcon />;
    default: return <InfoIcon />;
  }
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [dialog, setDialog] = useState<DialogState>({
    isOpen: false,
    type: 'alert',
    options: { message: '' },
  });
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile on mount and window resize
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 640);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Focus management
  useEffect(() => {
    if (dialog.isOpen) {
      // Save current focus
      previousFocusRef.current = document.activeElement as HTMLElement;

      // Focus dialog after render
      setTimeout(() => {
        const firstButton = dialogRef.current?.querySelector('button');
        firstButton?.focus();
      }, 100);
    } else if (previousFocusRef.current) {
      // Restore focus when dialog closes
      previousFocusRef.current.focus();
      previousFocusRef.current = null;
    }
  }, [dialog.isOpen]);

  // Trap focus within dialog
  useEffect(() => {
    if (!dialog.isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dialog.type === 'alert') {
        handleClose();
      }

      if (e.key === 'Tab') {
        const focusableElements = dialogRef.current?.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusableElements || focusableElements.length === 0) return;

        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [dialog.isOpen, dialog.type]);

  const alert = useCallback((options: AlertOptions | string): Promise<void> => {
    return new Promise((resolve) => {
      const opts = typeof options === 'string' ? { message: options } : options;
      setDialog({
        isOpen: true,
        type: 'alert',
        options: opts,
        resolve: () => resolve(),
      });
    });
  }, []);

  const confirm = useCallback((options: ConfirmOptions | string): Promise<boolean> => {
    return new Promise((resolve) => {
      const opts = typeof options === 'string' ? { message: options } : options;
      setDialog({
        isOpen: true,
        type: 'confirm',
        options: opts,
        resolve,
      });
    });
  }, []);

  const handleClose = useCallback((result: boolean = false) => {
    dialog.resolve?.(result);
    setDialog((prev) => ({ ...prev, isOpen: false }));
  }, [dialog]);

  const handleConfirm = useCallback(() => {
    handleClose(true);
  }, [handleClose]);

  const handleCancel = useCallback(() => {
    handleClose(false);
  }, [handleClose]);

  const options = dialog.options as AlertOptions & ConfirmOptions;
  const type: string = options.type || (dialog.type === 'confirm' ? 'danger' : 'info');
  const dialogTitleId = 'dialog-title';
  const dialogDescId = 'dialog-description';

  return (
    <DialogContext.Provider value={{ alert, confirm }}>
      {children}

      {/* Dialog Modal/Bottom Sheet - Responsive */}
      {dialog.isOpen && (
        <div
          className={`fixed inset-0 z-[100] flex ${isMobile ? 'items-end' : 'items-center justify-center p-4'}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby={options.title ? dialogTitleId : undefined}
          aria-describedby={dialogDescId}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm animate-fadeIn"
            onClick={dialog.type === 'alert' ? () => handleClose() : undefined}
            aria-hidden="true"
          />

          {/* Dialog - Bottom sheet on mobile, centered modal on desktop */}
          <div
            ref={dialogRef}
            className={`relative bg-white shadow-2xl w-full animate-slideUp overflow-hidden ${
              isMobile
                ? 'rounded-t-3xl max-w-full'
                : 'rounded-2xl max-w-sm sm:max-w-md lg:max-w-lg'
            }`}
          >
            {/* Drag Handle - Mobile only */}
            {isMobile && (
              <div className="flex items-center justify-center py-3 px-4">
                <div className="w-12 h-1.5 bg-gray-300 rounded-full" />
              </div>
            )}

            {/* Content - Responsive padding */}
            <div className="p-4 sm:p-6 lg:p-8 text-center">
              {/* Icon */}
              <div className="flex justify-center mb-4" aria-hidden="true">
                {getIcon(type, dialog.type)}
              </div>

              {/* Title - Responsive text */}
              {options.title && (
                <h3 id={dialogTitleId} className="text-base sm:text-lg lg:text-xl font-semibold text-gray-900 mb-2">
                  {options.title}
                </h3>
              )}

              {/* Message - Responsive text */}
              <p id={dialogDescId} className="text-gray-600 text-sm sm:text-base lg:text-base leading-relaxed">
                {options.message}
              </p>
            </div>

            {/* Actions - Responsive padding and layout */}
            <div className={`p-3 sm:p-4 bg-gray-50 ${dialog.type === 'confirm' ? 'flex flex-col-reverse sm:flex-row gap-2 sm:gap-3' : ''}`}>
              {dialog.type === 'confirm' ? (
                <>
                  <button
                    onClick={handleCancel}
                    className="flex-1 px-4 py-3 sm:py-2.5 rounded-lg sm:rounded-xl font-medium text-sm sm:text-base text-gray-700 bg-white border border-gray-200 hover:bg-gray-100 transition min-h-[48px] sm:min-h-[44px]"
                  >
                    {options.cancelText || 'Cancel'}
                  </button>
                  <button
                    onClick={handleConfirm}
                    className={`flex-1 px-4 py-3 sm:py-2.5 rounded-lg sm:rounded-xl font-medium text-sm sm:text-base text-white transition min-h-[48px] sm:min-h-[44px] ${
                      type === 'danger'
                        ? 'bg-red-500 hover:bg-red-600'
                        : type === 'warning'
                        ? 'bg-amber-500 hover:bg-amber-600'
                        : 'bg-blue-500 hover:bg-blue-600'
                    }`}
                  >
                    {options.confirmText || 'Confirm'}
                  </button>
                </>
              ) : (
                <button
                  onClick={() => handleClose()}
                  className={`w-full px-4 py-3 sm:py-2.5 rounded-lg sm:rounded-xl font-medium text-sm sm:text-base text-white transition min-h-[48px] sm:min-h-[44px] ${
                    type === 'success'
                      ? 'bg-green-500 hover:bg-green-600'
                      : type === 'error'
                      ? 'bg-red-500 hover:bg-red-600'
                      : type === 'warning'
                      ? 'bg-amber-500 hover:bg-amber-600'
                      : 'bg-blue-500 hover:bg-blue-600'
                  }`}
                >
                  {options.buttonText || 'OK'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Add animations - Optimized for 60fps with GPU acceleration */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translate3d(0, 20px, 0) scale3d(0.95, 0.95, 1);
          }
          to {
            opacity: 1;
            transform: translate3d(0, 0, 0) scale3d(1, 1, 1);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
          will-change: opacity;
        }
        .animate-slideUp {
          animation: slideUp 0.3s ease-out;
          will-change: transform, opacity;
        }
      `}</style>
    </DialogContext.Provider>
  );
}
