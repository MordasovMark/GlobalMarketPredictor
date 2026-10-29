import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error('[GlobalMarketPredictor] Uncaught render error:', error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0B0E14] flex items-center justify-center">
          <div className="text-center max-w-md px-6">
            <p className="text-2xl font-bold text-white mb-3">Something went wrong</p>
            <p className="text-sm text-gray-400 mb-6">{String(this.state.error?.message || 'Unknown error')}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-5 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
