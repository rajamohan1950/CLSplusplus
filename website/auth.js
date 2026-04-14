/**
 * CLS++ Auth Utilities
 * Manages user session state across all pages.
 */
(function () {
  'use strict';

  var _user = null;
  var _checked = false;

  window.CLSAuth = {
    /**
     * Fetch current user from /v1/auth/me (cookie sent automatically).
     * Caches the result for the session.
     */
    async getUser() {
      if (_user) return _user;
      if (_checked) return null;
      try {
        var resp = await fetch('/v1/auth/me', { credentials: 'same-origin' });
        _checked = true;
        if (resp.ok) {
          _user = await resp.json();
          window._CLS_USER_ID = _user.id;
          if (window.CLSAnalytics) CLSAnalytics.identify(_user);
          return _user;
        }
      } catch (e) { /* not logged in */ }
      return null;
    },

    /** Returns true if the user has a valid session. */
    async isLoggedIn() {
      return (await this.getUser()) !== null;
    },

    /** Logout: clear cookie and redirect. */
    async logout() {
      try {
        await fetch('/v1/auth/logout', { method: 'POST', credentials: 'same-origin' });
      } catch (e) { /* ignore */ }
      if (window.CLSAnalytics) { CLSAnalytics.track('user_logged_out'); CLSAnalytics.reset(); }
      _user = null;
      _checked = false;
      window._CLS_USER_ID = null;
      window.location.href = '/login.html';
    },

    /**
     * Update the nav bar with auth state.
     * Call this on DOMContentLoaded.
     */
    async updateNav() {
      var container = document.getElementById('nav-auth');
      if (!container) return;

      var user = await this.getUser();
      if (user) {
        var avatar = user.avatar_url
          ? '<img src="' + user.avatar_url + '" class="nav-avatar" alt="">'
          : '<span class="nav-avatar-placeholder">' + (user.name || user.email)[0].toUpperCase() + '</span>';
        var adminLink = user.is_admin
          ? '<div class="nav-admin-wrap">' +
              '<a href="/admin/dashboard.html" class="nav-admin-link">Admin</a>' +
              '<div class="nav-admin-dropdown">' +
                '<div class="nav-dd-group">' +
                  '<div class="nav-dd-heading">Core</div>' +
                  '<a href="/admin/dashboard.html">Admin Dashboard</a>' +
                  '<a href="/admin/permissions.html">Permissions</a>' +
                  '<a href="/dashboard.html">Dashboard</a>' +
                '</div>' +
                '<div class="nav-dd-group">' +
                  '<div class="nav-dd-heading">Content</div>' +
                  '<a href="/memory.html">Memory Viewer</a>' +
                  '<a href="/chat.html">Chat</a>' +
                  '<a href="/profile.html">Profile</a>' +
                  '<a href="/usage.html">Usage</a>' +
                '</div>' +
                '<div class="nav-dd-group">' +
                  '<div class="nav-dd-heading">Developer</div>' +
                  '<a href="/docs.html">API Docs</a>' +
                  '<a href="/integrate.html">Integrate</a>' +
                  '<a href="/getting-started.html">Getting Started</a>' +
                  '<a href="/trace.html">Trace</a>' +
                  '<a href="/tests.html">Tests</a>' +
                  '<a href="/benchmark.html">Benchmarks</a>' +
                  '<a href="/demo.html">Demo</a>' +
                '</div>' +
                '<div class="nav-dd-group">' +
                  '<div class="nav-dd-heading">Info</div>' +
                  '<a href="/support.html">Support</a>' +
                  '<a href="/submit.html">Submit</a>' +
                  '<a href="/install.html">Install</a>' +
                  '<a href="/privacy.html">Privacy</a>' +
                  '<a href="/terms.html">Terms</a>' +
                '</div>' +
              '</div>' +
            '</div>'
          : '';
        container.innerHTML =
          adminLink +
          '<a href="/dashboard.html" class="nav-user">' +
            avatar +
            '<span class="nav-user-name">' + (user.name || user.email.split('@')[0]) + '</span>' +
          '</a>' +
          '<span class="nav-sep">|</span>' +
          '<a href="#" onclick="CLSAuth.logout();return false;" class="nav-logout">Logout</a>';
      } else {
        container.innerHTML =
          '<a href="/login.html" class="btn btn-outline btn-sm">Login</a>' +
          '<a href="/signup.html" class="btn btn-primary btn-sm">Sign Up</a>';
      }
    },

    /** Redirect to login if not authenticated. */
    async requireAuth() {
      if (!(await this.isLoggedIn())) {
        window.location.href = '/login.html?next=' + encodeURIComponent(window.location.pathname);
      }
    },

    /** Redirect if not admin. */
    async requireAdmin() {
      var user = await this.getUser();
      if (!user || !user.is_admin) {
        window.location.href = '/login.html?next=' + encodeURIComponent(window.location.pathname);
      }
    },
  };

  // Auto-update nav on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { CLSAuth.updateNav(); });
  } else {
    CLSAuth.updateNav();
  }
})();
