// CLS++ Theme Engine — shared by popup.html and sidepanel.html
// Themes are CSS custom property overrides. Switching = swap vars on :root.

const CLS_THEMES = [
  {
    id: 'midnight',
    name: 'Midnight',
    preview: { bg: '#0a0a12', accent: '#7c6ef0' },
    vars: {}  // default — no overrides needed
  },
  {
    id: 'aurora',
    name: 'Aurora',
    preview: { bg: '#0d1117', accent: '#58d5a0' },
    vars: {
      '--cls-bg': '#0d1117',
      '--cls-accent': '#58d5a0',
      '--cls-accent-glow': 'rgba(88,213,160,0.25)',
      '--cls-success': '#58d5a0',
      '--cls-success-glow': 'rgba(88,213,160,0.2)',
      '--cls-surface': 'rgba(88,213,160,0.04)',
      '--cls-surface-hover': 'rgba(88,213,160,0.08)',
      '--cls-border': 'rgba(88,213,160,0.1)',
      '--cls-border-focus': 'rgba(88,213,160,0.25)',
      '--cls-bg-image': 'radial-gradient(ellipse at 20% 0%, rgba(88,213,160,0.06) 0%, transparent 60%)'
    }
  },
  {
    id: 'sunset',
    name: 'Sunset',
    preview: { bg: '#1a0f0f', accent: '#f0845d' },
    vars: {
      '--cls-bg': '#1a0f0f',
      '--cls-accent': '#f0845d',
      '--cls-accent-glow': 'rgba(240,132,93,0.25)',
      '--cls-success': '#f0c05d',
      '--cls-surface': 'rgba(240,132,93,0.04)',
      '--cls-surface-hover': 'rgba(240,132,93,0.08)',
      '--cls-border': 'rgba(240,132,93,0.1)',
      '--cls-border-focus': 'rgba(240,132,93,0.2)',
      '--cls-bg-image': 'radial-gradient(ellipse at 80% 0%, rgba(240,132,93,0.06) 0%, transparent 50%)'
    }
  },
  {
    id: 'ocean',
    name: 'Ocean',
    preview: { bg: '#0a1628', accent: '#5db8e0' },
    vars: {
      '--cls-bg': '#0a1628',
      '--cls-accent': '#5db8e0',
      '--cls-accent-glow': 'rgba(93,184,224,0.25)',
      '--cls-success': '#5de0c5',
      '--cls-surface': 'rgba(93,184,224,0.04)',
      '--cls-surface-hover': 'rgba(93,184,224,0.08)',
      '--cls-border': 'rgba(93,184,224,0.1)',
      '--cls-border-focus': 'rgba(93,184,224,0.2)',
      '--cls-bg-image': 'radial-gradient(ellipse at 50% 100%, rgba(93,184,224,0.06) 0%, transparent 50%)'
    }
  },
  {
    id: 'light',
    name: 'Minimal Light',
    preview: { bg: '#fafafa', accent: '#6355e0' },
    vars: {
      '--cls-bg': '#fafafa',
      '--cls-text': '#1a1a2e',
      '--cls-text-muted': 'rgba(26,26,46,0.5)',
      '--cls-text-dim': 'rgba(26,26,46,0.35)',
      '--cls-accent': '#6355e0',
      '--cls-accent-glow': 'rgba(99,85,224,0.15)',
      '--cls-accent-text': '#fff',
      '--cls-success': '#2aa87e',
      '--cls-success-glow': 'rgba(42,168,126,0.15)',
      '--cls-warning': '#d4960a',
      '--cls-danger': '#d94475',
      '--cls-surface': 'rgba(0,0,0,0.03)',
      '--cls-surface-hover': 'rgba(0,0,0,0.06)',
      '--cls-surface-active': 'rgba(0,0,0,0.09)',
      '--cls-border': 'rgba(0,0,0,0.08)',
      '--cls-border-focus': 'rgba(0,0,0,0.18)',
      '--cls-bg-image': 'none'
    }
  }
];

// Default CSS vars (midnight theme) — stored so we can reset cleanly
const CLS_DEFAULT_VARS = {
  '--cls-bg': '#0a0a12',
  '--cls-text': '#f0f0f8',
  '--cls-text-muted': 'rgba(240,240,248,0.5)',
  '--cls-text-dim': 'rgba(240,240,248,0.3)',
  '--cls-accent': '#7c6ef0',
  '--cls-accent-glow': 'rgba(124,110,240,0.25)',
  '--cls-accent-text': '#fff',
  '--cls-success': '#5de0c5',
  '--cls-success-glow': 'rgba(93,224,197,0.2)',
  '--cls-warning': '#f0c05d',
  '--cls-danger': '#f05d9a',
  '--cls-surface': 'rgba(255,255,255,0.04)',
  '--cls-surface-hover': 'rgba(255,255,255,0.08)',
  '--cls-surface-active': 'rgba(255,255,255,0.12)',
  '--cls-border': 'rgba(255,255,255,0.08)',
  '--cls-border-focus': 'rgba(255,255,255,0.2)',
  '--cls-bg-image': 'none',
  '--cls-bg-overlay': 'none'
};

/**
 * Apply a theme by ID. Sets CSS vars on :root and saves to storage.
 */
function clsApplyTheme(themeId) {
  const theme = CLS_THEMES.find(t => t.id === themeId);
  if (!theme) return;

  const root = document.documentElement;

  // Reset all vars to default first
  for (const [key, val] of Object.entries(CLS_DEFAULT_VARS)) {
    root.style.setProperty(key, val);
  }

  // Apply theme overrides
  for (const [key, val] of Object.entries(theme.vars)) {
    root.style.setProperty(key, val);
  }

  // Save
  chrome.storage.local.set({ cls_theme: themeId });
}

/**
 * Load saved theme from storage and apply it. Call on page load.
 */
function clsLoadTheme() {
  chrome.storage.local.get('cls_theme', function (r) {
    clsApplyTheme(r.cls_theme || 'midnight');
  });
}

/**
 * Render theme swatch grid into a container element.
 * @param {HTMLElement} container - Element to render swatches into
 * @param {string} activeId - Currently active theme ID
 */
function clsRenderThemeGrid(container, activeId) {
  container.innerHTML = '';
  CLS_THEMES.forEach(function (theme) {
    var wrap = document.createElement('div');
    wrap.style.textAlign = 'center';

    var swatch = document.createElement('div');
    swatch.className = 'cls-theme-swatch' + (theme.id === activeId ? ' active' : '');
    swatch.title = theme.name;
    swatch.dataset.theme = theme.id;

    var inner = document.createElement('div');
    inner.className = 'cls-theme-swatch-inner';
    inner.style.background = theme.preview.bg;

    var dot = document.createElement('div');
    dot.className = 'cls-theme-swatch-accent';
    dot.style.background = theme.preview.accent;

    inner.appendChild(dot);
    swatch.appendChild(inner);

    var name = document.createElement('div');
    name.className = 'cls-theme-name';
    name.textContent = theme.name;

    wrap.appendChild(swatch);
    wrap.appendChild(name);
    container.appendChild(wrap);

    swatch.addEventListener('click', function () {
      clsApplyTheme(theme.id);
      if (typeof CLSExtAnalytics !== 'undefined') {
        CLSExtAnalytics.track('theme_changed', { theme: theme.id });
      }
      // Update active states
      container.querySelectorAll('.cls-theme-swatch').forEach(function (s) {
        s.classList.toggle('active', s.dataset.theme === theme.id);
      });
    });
  });
}
