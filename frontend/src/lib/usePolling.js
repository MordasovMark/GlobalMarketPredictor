import { useEffect, useRef, useState } from 'react';

/**
 * Fetch on mount, then every `intervalMs`. Keeps the last good data when a
 * refresh fails, so a flaky request never blanks the screen.
 *
 * `load` receives an AbortSignal. `deps` re-trigger a fresh load.
 * A 503 (backend still warming up) is retried after `retryMs`.
 */
export function usePolling(load, intervalMs, deps = [], { retryMs = 15_000, enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled, lastSynced: null });
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    let timer = null;
    let controller = null;
    setState((s) => ({ ...s, data: null, loading: true, error: null }));

    const run = async () => {
      controller?.abort();
      controller = new AbortController();
      let wait = intervalMs;
      try {
        const data = await loadRef.current(controller.signal);
        if (!cancelled) setState({ data, error: null, loading: false, lastSynced: Date.now() });
      } catch (err) {
        if (cancelled || err?.name === 'AbortError') return;
        if (err?.status === 503) wait = Math.min(intervalMs, retryMs);
        setState((s) => ({ ...s, loading: false, error: err?.message || 'Request failed', status: err?.status }));
      }
      if (!cancelled && wait) timer = setTimeout(run, wait);
    };
    run();
    return () => {
      cancelled = true;
      controller?.abort();
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, enabled, ...deps]);

  return state;
}
