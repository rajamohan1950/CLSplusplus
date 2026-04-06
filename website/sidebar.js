/**
 * CLS++ Shared Sidebar Component
 * Renders the left navigation sidebar on app pages (dashboard, profile, usage).
 * Usage: include this script then call renderSidebar('pageName') on DOMContentLoaded.
 */
(function () {
  'use strict';

  var NAV_ITEMS = [
    { section: 'Main' },
    { id: 'dashboard', label: 'Dashboard',     icon: '\u2302', href: '/dashboard.html' },
    { id: 'profile',   label: 'Profile',       icon: '\u{1F464}', href: '/profile.html' },
    { id: 'usage',     label: 'Usage',         icon: '\u{1F4CA}', href: '/usage.html' },
    { section: 'Tools' },
    { id: 'memory',    label: 'Memory Viewer', icon: '\u{1F9E0}', href: '/memory.html' },
    { id: 'chat',      label: 'Chat',          icon: '\u{1F4AC}', href: '/chat.html' },
    { section: 'Developer' },
    { id: 'docs',         label: 'API Docs',       icon: '\u{1F4D6}', href: '/docs.html' },
    { id: 'integrations', label: 'Integrations',   icon: '\u26A1',    href: '/integrate.html' },
  ];

  window.renderSidebar = async function (activePage) {
    var container = document.querySelector('.app-sidebar');
    if (!container) return;

    var user = (typeof CLSAuth !== 'undefined') ? await CLSAuth.getUser() : null;

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
      var isActive = (item.id === activePage) ? ' active' : '';
      html += '<a href="' + item.href + '" class="sidebar-nav-item' + isActive + '">';
      html += '  <span class="sidebar-icon">' + item.icon + '</span>';
      html += '  <span class="sidebar-label">' + item.label + '</span>';
      html += '</a>';
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
  };

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
