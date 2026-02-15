'use client';

import React, { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

/**
 * Error Boundary that catches client-side errors and displays a fallback UI
 * while keeping the debug panel functional for authorized users
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true };
  }

  async componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to console (will be captured by debug panel if active)
    console.error('ErrorBoundary caught an error:', error);
    console.error('Error component stack:', errorInfo.componentStack);

    this.setState({
      error,
      errorInfo,
    });

    // Report error to backend for tracking and analysis
    try {
      await fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: error.message,
          stack: error.stack,
          componentStack: errorInfo.componentStack,
          url: window.location.href,
          errorType: 'react',
          severity: 'high',
          additionalData: {
            userAgent: navigator.userAgent,
            timestamp: new Date().toISOString(),
          },
        }),
      });
    } catch (reportError) {
      console.error('Failed to report error:', reportError);
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    // Reload the page to reset state
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full bg-white rounded-lg shadow-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <span className="text-5xl">⚠️</span>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Application Error</h1>
                <p className="text-gray-600">A client-side exception has occurred</p>
              </div>
            </div>

            {this.state.error && (
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-2">Error Details</h2>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="font-mono text-sm text-red-800 mb-2">
                    <strong>Message:</strong> {this.state.error.message}
                  </p>
                  {this.state.error.stack && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-sm text-red-700 hover:text-red-900 font-medium">
                        Stack Trace
                      </summary>
                      <pre className="mt-2 text-xs text-red-700 overflow-x-auto bg-white p-3 rounded border border-red-200">
                        {this.state.error.stack}
                      </pre>
                    </details>
                  )}
                  {this.state.errorInfo && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-sm text-red-700 hover:text-red-900 font-medium">
                        Component Stack
                      </summary>
                      <pre className="mt-2 text-xs text-red-700 overflow-x-auto bg-white p-3 rounded border border-red-200">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={this.handleReset}
                className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
              >
                Reload Page
              </button>
              <a
                href="/dashboard"
                className="px-6 py-3 bg-gray-200 text-gray-800 font-semibold rounded-lg hover:bg-gray-300 transition-colors"
              >
                Go to Dashboard
              </a>
            </div>

            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>💡 Tip:</strong> If you have debug access, check the debug panel (🐛 button)
                at the bottom right to see console logs and API calls that may help diagnose this issue.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
