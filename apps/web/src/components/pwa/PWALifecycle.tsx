/**
 * PWA Lifecycle Component
 * =======================
 *
 * Manages service worker registration and displays update notifications.
 * Mounted in root layout.
 */

'use client';

import { useEffect, useState } from 'react';
import { registerServiceWorker } from '@/lib/pwa/serviceWorkerRegistration';

export default function PWALifecycle() {
  const [showUpdateNotification, setShowUpdateNotification] = useState(false);
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    // Register service worker
    registerServiceWorker({
      onSuccess: (registration) => {
        console.log('[PWA] Service worker registered successfully');
      },
      onUpdate: (registration) => {
        console.log('[PWA] New version available');
        setShowUpdateNotification(true);
      },
      onOffline: () => {
        setIsOffline(true);
      },
      onOnline: () => {
        setIsOffline(false);
      },
    });

    // Check if already offline
    if (!navigator.onLine) {
      setIsOffline(true);
    }
  }, []);

  const handleUpdate = () => {
    setShowUpdateNotification(false);
    window.location.reload();
  };

  return (
    <>
      {/* Update Notification */}
      {showUpdateNotification && (
        <div className="fixed bottom-4 right-4 z-50 p-4 bg-emerald-600 text-white rounded-lg shadow-lg max-w-sm animate-slide-up">
          <div className="flex items-start gap-3">
            <div className="flex-1">
              <h3 className="font-semibold mb-1">Update Available</h3>
              <p className="text-sm opacity-90">
                A new version of Jotty AI is available. Update now for the latest features.
              </p>
            </div>
            <button
              onClick={() => setShowUpdateNotification(false)}
              className="text-white opacity-70 hover:opacity-100"
            >
              ×
            </button>
          </div>
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleUpdate}
              className="px-4 py-2 bg-white text-emerald-600 rounded-lg font-semibold hover:bg-gray-100"
            >
              Update Now
            </button>
            <button
              onClick={() => setShowUpdateNotification(false)}
              className="px-4 py-2 bg-emerald-700 rounded-lg hover:bg-emerald-800"
            >
              Later
            </button>
          </div>
        </div>
      )}

      {/* Offline Indicator */}
      {isOffline && (
        <div className="fixed top-0 left-0 right-0 z-50 p-2 bg-yellow-600 text-white text-center text-sm">
          ⚠️ You are offline. Some features may be limited.
        </div>
      )}
    </>
  );
}
