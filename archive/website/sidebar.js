/**
 * CLS++ Shared Sidebar Component
 * Renders the left navigation sidebar with expandable submenu groups.
 * Usage: include this script then call renderSidebar('pageName') on DOMContentLoaded.
 */
(function () {
  'use strict';

  var NAV_ITEMS = [
    { section: 'Main' },
    { id: 'dashboard', label: 'Dashboard', icon: '\u2302', href: '/dashboard.html' },
    {
      id: 'profile', label: 'Profile', icon: '\u{1F464}', href: '/profile.html',
      children: [
        { id: 'profile-info',    label: 'User Info',    href: '/profile.html#info' },
        { id: 'profile-prefs',   label: 'Preferences',  href: '/profile.html#prefs' },
        { id: 'profile-keys',    label: 'API Keys',     href: '/profile.html#keys' },
        { id: 'profile-billing', label: 'Billing',      href: '/profile.html#billing' },
      ],
    },
    {
      id: 'usage', label: 'Usage', icon: '\u{1F4CA}', href: '/usage.html',
      children: [
        { id: 'usage-overview', label: 'Overview', href: '/usage.html#overview' },
        { id: 'usage-history',  label: 'History',  href: '/usage.html#history' },
      ],
    },
    { section: 'Tools' },
    { id: 'memory', label: 'Memory Viewer', icon: '\u{1F9E0}', href: '/memory.html' },
    { id: 'chat',   label: 'Chat',          icon: '\u{1F4AC}', href: '/chat.html' },
    { id: 'traces', label: 'Traces',        icon: '\u{1F50D}', href: '/trace.html' },
    { section: 'Developer' },
    { id: 'docs',         label: 'API Docs',     icon: '\u{1F4D6}', href: '/docs.html' },
    { id: 'integrations', label: 'Integrations', icon: '\u26A1',    href: '/integrate.html' },
  ];

  window.renderSidebar = async function (activePage) {
    var container = document.querySelector('.app-sidebar');
    if (!container) return;

    var user = (typeof CLSAuth !== 'undefined') ? await CLSAuth.getUser() : null;
    var currentHash = window.location.hash.replace('#', '');

    var html = '';

    // Brand
    html += '<div class="app-sidebar-brand">';
    html += '  <a href="/" class="logo">CLS<span class="logo-plus">++</span></a>';
    html += '</div>';

    // User
    if (user) {
      var avatar = user.avatar_url
        ? '<img src="' + user.avatar_url + '" class="sidebar-avatar" alt="">'
        : '<span class="sidebar-avatar-placeholder">' + (user.name || user.email)[0].toUpperCase() + '</span>';

      html += '<div class="app-sidebar-user">';
      html += '  ' + avatar;
      html += '  <div class="sidebar-user-info">';
      html += '    <div class="sidebar-user-name">' + (user.name || user.email.split('@')[0]) + '</div>';
      html += '    <div class="sidebar-user-email">' + user.email + '</div>';
      html += '  </div>';
      html += '</div>';
    }

    // Nav items
    html += '<nav class="app-sidebar-nav">';
    NAV_ITEMS.forEach(function (item) {
      if (item.section) {
        html += '<div class="sidebar-section-label">' + item.section + '</div>';
        return;
      }

      var isActive = (item.id === activePage);
      var hasChildren = item.children && item.children.length > 0;
      var isExpanded = isActive && hasChildren;

      if (hasChildren) {
        // Expandable parent
        html += '<div class="sidebar-group' + (isExpanded ? ' expanded' : '') + '" data-group="' + item.id + '">';
        html += '  <a href="' + item.href + '" class="sidebar-nav-item expandable' + (isActive ? ' active' : '') + '" data-toggle="' + item.id + '">';
        html += '    <span class="sidebar-icon">' + item.icon + '</span>';
        html += '    <span class="sidebar-label">' + item.label + '</span>';
        html += '    <span class="sidebar-chevron">\u203A</span>';
        html += '  </a>';
        html += '  <div class="sidebar-submenu">';
        item.children.forEach(function (child) {
          var childHash = child.href.split('#')[1] || '';
          var isChildActive = isActive && currentHash === childHash;
          html += '    <a href="' + child.href + '" class="sidebar-sub-item' + (isChildActive ? ' active' : '') + '" data-hash="' + childHash + '">';
          html += '      ' + child.label;
          html += '    </a>';
        });
        html += '  </div>';
        html += '</div>';
      } else {
        // Simple link
        html += '<a href="' + item.href + '" class="sidebar-nav-item' + (isActive ? ' active' : '') + '">';
        html += '  <span class="sidebar-icon">' + item.icon + '</span>';
        html += '  <span class="sidebar-label">' + item.label + '</span>';
        html += '</a>';
      }
    });

    // Admin link for admins
    if (user && user.is_admin) {
      html += '<div class="sidebar-section-label">Admin</div>';
      html += '<a href="/admin/dashboard.html" class="sidebar-nav-item">';
      html += '  <span class="sidebar-icon">\u2699</span>';
      html += '  <span class="sidebar-label">Admin Panel</span>';
      html += '</a>';
    }

    html += '</nav>';

    // Footer with logout
    html += '<div class="app-sidebar-footer">';
    html += '  <button class="sidebar-logout" onclick="CLSAuth.logout();return false;">';
    html += '    <span class="sidebar-icon">\u{1F6AA}</span>';
    html += '    <span>Logout</span>';
    html += '  </button>';
    html += '</div>';

    container.innerHTML = html;

    // Bind expand/collapse on parent items
    container.querySelectorAll('.sidebar-nav-item.expandable').forEach(function (el) {
      el.addEventListener('click', function (e) {
        var group = el.closest('.sidebar-group');
        // If already on the page, just toggle submenu (don't navigate)
        if (group && group.classList.contains('expanded') || el.classList.contains('active')) {
          // If clicking the parent while on that page, toggle expand
          if (el.classList.contains('active')) {
            e.preventDefault();
            group.classList.toggle('expanded');
          }
          // If not active, let navigation happen
        }
      });
    });

    // Bind hash-based sub-item clicks for same-page navigation
    container.querySelectorAll('.sidebar-sub-item').forEach(function (el) {
      el.addEventListener('click', function (e) {
        var href = el.getAttribute('href');
        var pagePath = href.split('#')[0];
        var hash = el.getAttribute('data-hash');

        // Same page — scroll to section
        if (window.location.pathname === pagePath || window.location.pathname === pagePath.replace('.html', '')) {
          e.preventDefault();
          window.location.hash = hash;
          var target = document.getElementById(hash);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }

          // Update active sub-item
          container.querySelectorAll('.sidebar-sub-item').forEach(function (s) { s.classList.remove('active'); });
          el.classList.add('active');
        }
        // Different page — let default navigation happen
      });
    });
  };

  // Update active sub-item on hash change
  window.addEventListener('hashchange', function () {
    var hash = window.location.hash.replace('#', '');
    document.querySelectorAll('.sidebar-sub-item').forEach(function (el) {
      el.classList.toggle('active', el.getAttribute('data-hash') === hash);
    });
  });

  // Mobile sidebar toggle
  document.addEventListener('click', function (e) {
    if (e.target.closest('.sidebar-mobile-toggle')) {
      var sidebar = document.querySelector('.app-sidebar');
      if (sidebar) sidebar.classList.toggle('open');
    }
    // Close sidebar on outside click (mobile)
    if (document.querySelector('.app-sidebar.open') && !e.target.closest('.app-sidebar') && !e.target.closest('.sidebar-mobile-toggle')) {
      document.querySelector('.app-sidebar').classList.remove('open');
    }
  });
})();
