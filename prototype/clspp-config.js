/**
 * Shared CLS++ UI connection config (home + memory viewer).
 * localStorage: clspp_api_base, clspp_api_key, clspp_ui_backend = prototype | enterprise
 */
(function (global) {
  const LS = {
    API_BASE: 'clspp_api_base',
    API_KEY: 'clspp_api_key',
    BACKEND: 'clspp_ui_backend',
  };

  function defaultBase() {
    try {
      const h = location.hostname;
      if (h === 'localhost' || h === '127.0.0.1')
        return `http://${h}:${location.port || '8080'}`;
    } catch (e) {}
    return 'http://127.0.0.1:8080';
  }

  function apiBase() {
    const v = localStorage.getItem(LS.API_BASE);
    return (v && v.trim()) || defaultBase();
  }

  function apiKey() {
    return localStorage.getItem(LS.API_KEY) || '';
  }

  function uiBackend() {
    return localStorage.getItem(LS.BACKEND) || 'prototype';
  }

  function setConnection(opts) {
    if (opts.apiBase != null) localStorage.setItem(LS.API_BASE, String(opts.apiBase).replace(/\/$/, ''));
    if (opts.apiKey !== undefined) {
      if (opts.apiKey) localStorage.setItem(LS.API_KEY, opts.apiKey);
      else localStorage.removeItem(LS.API_KEY);
    }
    if (opts.backend) localStorage.setItem(LS.BACKEND, opts.backend);
  }

  function headers(extra) {
    const h = Object.assign({ Accept: 'application/json' }, extra || {});
    const k = apiKey();
    if (k) h.Authorization = 'Bearer ' + k;
    return h;
  }

  function fetchApi(path, opts) {
    opts = Object.assign({}, opts || {});
    const url = apiBase().replace(/\/$/, '') + path;
    const extra = opts.headers || {};
    delete opts.headers;
    const h = headers(extra);
    if (opts.body && typeof opts.body === 'string' && !h['Content-Type'])
      h['Content-Type'] = 'application/json';
    opts.headers = h;
    return fetch(url, opts);
  }

  /** Detect which API shape responds at this base URL. */
  async function probeBackend(base, optionalKey) {
    const b = String(base || '').replace(/\/$/, '');
    if (!b) return null;
    try {
      const r = await fetch(b + '/health', { method: 'GET' });
      if (r.ok) {
        const j = await r.json().catch(() => ({}));
        if (j.status === 'ok' && typeof j.namespaces === 'number') return 'prototype';
      }
    } catch (e) {}
    try {
      const h = {};
      const k = optionalKey != null ? optionalKey : apiKey();
      if (k) h.Authorization = 'Bearer ' + k;
      const r = await fetch(b + '/v1/memory/health', { headers: h });
      if (r.ok) return 'enterprise';
    } catch (e) {}
    return null;
  }

  /** Single traces endpoint: list (default) or detail when opts.traceId is set. */
  function memoryTracesUrl(opts) {
    opts = opts || {};
    const path =
      uiBackend() === 'enterprise' ? '/v1/memory/traces' : '/api/memory/traces';
    const q = new URLSearchParams();
    if (opts.traceId) q.set('trace_id', String(opts.traceId).trim());
    if (opts.limit != null) q.set('limit', String(opts.limit));
    const s = q.toString();
    return s ? path + '?' + s : path;
  }

  global.CLSPPConfig = {
    LS,
    apiBase,
    apiKey,
    uiBackend,
    setConnection,
    fetchApi,
    probeBackend,
    defaultBase,
    memoryTracesUrl,
  };
})(typeof window !== 'undefined' ? window : this);
