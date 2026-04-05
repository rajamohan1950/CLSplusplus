/**
 * CLS++ Admin Permissions Manager
 * RBAC management: roles, groups, users, scope assignment.
 */
(function () {
  'use strict';

  var _roles = [], _groups = [], _users = [], _allScopes = [];

  async function init() {
    if (typeof CLSAuth !== 'undefined') await CLSAuth.requireAdmin();
    await loadSidebar();
  }

  async function fetchJSON(url, opts) {
    try {
      var r = await fetch(url, Object.assign({ credentials: 'same-origin' }, opts || {}));
      if (r.ok) return await r.json();
      var err = await r.json().catch(function(){ return {}; });
      if (err.detail) alert(err.detail);
    } catch (e) {}
    return null;
  }

  async function postJSON(url, data) {
    return fetchJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
  }

  async function putJSON(url, data) {
    return fetchJSON(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
  }

  async function del(url) {
    return fetchJSON(url, { method: 'DELETE' });
  }

  // ── Sidebar ─────────────────────────────────────────────────────────────

  async function loadSidebar() {
    var [rolesData, groupsData, usersData, scopesData] = await Promise.all([
      fetchJSON('/admin/rbac/roles'),
      fetchJSON('/admin/rbac/groups'),
      fetchJSON('/admin/metrics/users'),
      fetchJSON('/admin/rbac/scopes'),
    ]);

    _roles = (rolesData && rolesData.roles) || [];
    _groups = (groupsData && groupsData.groups) || [];
    _users = (usersData && usersData.users) || [];
    _allScopes = (scopesData && scopesData.scopes) || [];

    renderSidebarRoles();
    renderSidebarGroups();
    renderSidebarUsers(_users);
  }

  function renderSidebarRoles() {
    var el = document.getElementById('sidebar-roles');
    el.innerHTML = '';
    _roles.forEach(function (r) {
      var div = document.createElement('div');
      div.className = 'perm-item';
      div.textContent = r.name + (r.is_system ? ' *' : '');
      div.onclick = function () { showRole(r.id); };
      el.appendChild(div);
    });
  }

  function renderSidebarGroups() {
    var el = document.getElementById('sidebar-groups');
    el.innerHTML = '';
    _groups.forEach(function (g) {
      var div = document.createElement('div');
      div.className = 'perm-item';
      div.textContent = g.name + ' (' + (g.member_count || 0) + ')';
      div.onclick = function () { showGroup(g.id); };
      el.appendChild(div);
    });
  }

  function renderSidebarUsers(users) {
    var el = document.getElementById('sidebar-users');
    el.innerHTML = '';
    users.forEach(function (u) {
      var div = document.createElement('div');
      div.className = 'perm-item';
      div.textContent = u.email;
      div.onclick = function () { showUser(u.id, u.email); };
      el.appendChild(div);
    });
  }

  window.filterUsers = function (q) {
    var filtered = _users.filter(function (u) {
      return u.email.toLowerCase().includes(q.toLowerCase()) || (u.name || '').toLowerCase().includes(q.toLowerCase());
    });
    renderSidebarUsers(filtered);
  };

  function setActive(el) {
    document.querySelectorAll('.perm-item.active').forEach(function (e) { e.classList.remove('active'); });
    if (el) el.classList.add('active');
  }

  // ── Scope grouping ──────────────────────────────────────────────────────

  function renderScopeGrid(scopeSet, checkboxes) {
    var apiScopes = _allScopes.filter(function(s) { return !s.startsWith('page:'); }).sort();
    var pageScopes = _allScopes.filter(function(s) { return s.startsWith('page:'); }).sort();
    var html = '';

    html += '<h3 style="margin:16px 0 8px;font-size:0.9rem;color:var(--accent-light);">API Permissions</h3>';
    html += '<div class="scope-grid">';
    apiScopes.forEach(function(s) {
      var checked = scopeSet.has(s) ? ' checked' : '';
      if (checkboxes) {
        html += '<label class="scope-check"><input type="checkbox" data-scope="' + s + '"' + checked + '> ' + s + '</label>';
      } else {
        var cls = scopeSet.has(s) ? 'eff-scope eff-granted' : 'eff-scope eff-missing';
        html += '<span class="' + cls + '">' + s + '</span> ';
      }
    });
    html += '</div>';

    html += '<h3 style="margin:16px 0 8px;font-size:0.9rem;color:#22c55e;">Page Access</h3>';
    html += '<div class="scope-grid">';
    pageScopes.forEach(function(s) {
      var label = s.replace('page:', '').replace('-', ' ');
      label = label.charAt(0).toUpperCase() + label.slice(1);
      var checked = scopeSet.has(s) ? ' checked' : '';
      if (checkboxes) {
        html += '<label class="scope-check"><input type="checkbox" data-scope="' + s + '"' + checked + '> ' + label + '</label>';
      } else {
        var cls = scopeSet.has(s) ? 'eff-scope eff-granted' : 'eff-scope eff-missing';
        html += '<span class="' + cls + '">' + label + '</span> ';
      }
    });
    html += '</div>';
    return html;
  }

  // ── Role Editor ─────────────────────────────────────────────────────────

  async function showRole(roleId) {
    var role = await fetchJSON('/admin/rbac/roles');
    var r = (role && role.roles || []).find(function (r) { return r.id === roleId; });
    if (!r) return;

    var scopes = (typeof r.scopes === 'string') ? JSON.parse(r.scopes) : (r.scopes || []);
    var main = document.getElementById('perm-main');
    var scopeSet = new Set(scopes);

    var html = '<h2>Role: ' + esc(r.name) + (r.is_system ? ' <span style="color:var(--text-muted);font-size:0.8rem;">(system)</span>' : '') + '</h2>';
    html += '<div class="field-group"><label>Description</label><input id="role-desc" value="' + esc(r.description) + '"></div>';
    html += renderScopeGrid(scopeSet, true);
    html += '<div class="perm-actions">';
    html += '<button class="perm-btn perm-btn-primary" onclick="saveRole(\'' + roleId + '\')">Save</button>';
    if (!r.is_system) html += '<button class="perm-btn perm-btn-danger" onclick="deleteRole(\'' + roleId + '\')">Delete</button>';
    html += '</div>';
    main.innerHTML = html;
  }

  window.saveRole = async function (roleId) {
    var desc = document.getElementById('role-desc').value;
    var scopes = [];
    document.querySelectorAll('.scope-check input:checked').forEach(function (cb) { scopes.push(cb.dataset.scope); });
    await putJSON('/admin/rbac/roles/' + roleId, { description: desc, scopes: scopes });
    await loadSidebar();
    showRole(roleId);
  };

  window.deleteRole = async function (roleId) {
    if (!confirm('Delete this role?')) return;
    await del('/admin/rbac/roles/' + roleId);
    await loadSidebar();
    document.getElementById('perm-main').innerHTML = '<div class="empty-state">Role deleted.</div>';
  };

  window.showCreateRole = async function () {
    var main = document.getElementById('perm-main');
    var html = '<h2>New Role</h2>';
    html += '<div class="field-group"><label>Name</label><input id="new-role-name" placeholder="e.g. data_analyst"></div>';
    html += '<div class="field-group"><label>Description</label><input id="new-role-desc" placeholder="What this role can do"></div>';
    html += renderScopeGrid(new Set(), true);
    html += '<div class="perm-actions"><button class="perm-btn perm-btn-primary" onclick="createRole()">Create</button></div>';
    main.innerHTML = html;
  };

  window.createRole = async function () {
    var name = document.getElementById('new-role-name').value.trim();
    var desc = document.getElementById('new-role-desc').value.trim();
    var scopes = [];
    document.querySelectorAll('.scope-check input:checked').forEach(function (cb) { scopes.push(cb.dataset.scope); });
    if (!name) { alert('Name required'); return; }
    await postJSON('/admin/rbac/roles', { name: name, description: desc, scopes: scopes });
    await loadSidebar();
  };

  // ── Group Editor ────────────────────────────────────────────────────────

  async function showGroup(groupId) {
    var [rolesData, membersData] = await Promise.all([
      fetchJSON('/admin/rbac/groups/' + groupId + '/roles'),
      fetchJSON('/admin/rbac/groups/' + groupId + '/members'),
    ]);
    var grp = _groups.find(function (g) { return g.id === groupId; });
    if (!grp) return;
    var groupRoles = (rolesData && rolesData.roles) || [];
    var members = (membersData && membersData.members) || [];

    var main = document.getElementById('perm-main');
    var html = '<h2>Group: ' + esc(grp.name) + '</h2>';

    // Assigned Roles
    html += '<h3 style="margin-top:24px;">Assigned Roles</h3>';
    html += '<div style="margin:8px 0;">';
    groupRoles.forEach(function (r) {
      html += '<span class="tag tag-role">' + esc(r.name) + '<span class="tag-remove" onclick="removeGroupRole(\'' + groupId + '\',\'' + r.id + '\')">&times;</span></span>';
    });
    if (!groupRoles.length) html += '<span style="color:var(--text-muted);">No roles assigned</span>';
    html += '</div>';
    html += '<div class="add-widget"><select id="add-grp-role">';
    _roles.forEach(function (r) { html += '<option value="' + r.id + '">' + r.name + '</option>'; });
    html += '</select><button onclick="addGroupRole(\'' + groupId + '\')">Add Role</button></div>';

    // Members
    html += '<h3 style="margin-top:24px;">Members (' + members.length + ')</h3>';
    html += '<div style="margin:8px 0;">';
    members.forEach(function (m) {
      html += '<span class="tag tag-group">' + esc(m.email) + '<span class="tag-remove" onclick="removeGroupMember(\'' + groupId + '\',\'' + m.id + '\')">&times;</span></span>';
    });
    if (!members.length) html += '<span style="color:var(--text-muted);">No members</span>';
    html += '</div>';
    html += '<div class="add-widget"><select id="add-grp-member">';
    _users.forEach(function (u) { html += '<option value="' + u.id + '">' + u.email + '</option>'; });
    html += '</select><button onclick="addGroupMember(\'' + groupId + '\')">Add Member</button></div>';

    html += '<div class="perm-actions" style="margin-top:24px;"><button class="perm-btn perm-btn-danger" onclick="deleteGroup(\'' + groupId + '\')">Delete Group</button></div>';
    main.innerHTML = html;
  }

  window.addGroupRole = async function (gid) { await postJSON('/admin/rbac/groups/' + gid + '/roles', { role_id: document.getElementById('add-grp-role').value }); showGroup(gid); };
  window.removeGroupRole = async function (gid, rid) { await del('/admin/rbac/groups/' + gid + '/roles/' + rid); showGroup(gid); };
  window.addGroupMember = async function (gid) { await postJSON('/admin/rbac/groups/' + gid + '/members', { user_id: document.getElementById('add-grp-member').value }); showGroup(gid); await loadSidebar(); };
  window.removeGroupMember = async function (gid, uid) { await del('/admin/rbac/groups/' + gid + '/members/' + uid); showGroup(gid); await loadSidebar(); };
  window.deleteGroup = async function (gid) { if (!confirm('Delete this group?')) return; await del('/admin/rbac/groups/' + gid); await loadSidebar(); document.getElementById('perm-main').innerHTML = '<div class="empty-state">Group deleted.</div>'; };

  window.showCreateGroup = async function () {
    var main = document.getElementById('perm-main');
    main.innerHTML = '<h2>New Group</h2><div class="field-group"><label>Name</label><input id="new-grp-name" placeholder="e.g. Engineering Team"></div><div class="field-group"><label>Description</label><input id="new-grp-desc"></div><div class="perm-actions"><button class="perm-btn perm-btn-primary" onclick="createGroup()">Create</button></div>';
  };

  window.createGroup = async function () {
    var name = document.getElementById('new-grp-name').value.trim();
    if (!name) { alert('Name required'); return; }
    await postJSON('/admin/rbac/groups', { name: name, description: document.getElementById('new-grp-desc').value });
    await loadSidebar();
  };

  // ── User Permission Viewer ──────────────────────────────────────────────

  async function showUser(userId, email) {
    var data = await fetchJSON('/admin/rbac/users/' + userId + '/effective');
    if (!data) return;

    var main = document.getElementById('perm-main');
    var user = _users.find(function (u) { return u.id === userId; });
    var html = '<h2>' + esc(email) + '</h2>';
    if (user) html += '<p style="color:var(--text-muted);">' + esc(user.name || '') + ' &middot; ' + user.tier + (user.is_admin ? ' &middot; <strong style="color:#eab308;">Admin</strong>' : '') + '</p>';

    // Effective Scopes — grouped
    html += '<h3 style="margin-top:24px;">Effective Permissions</h3>';
    var effSet = new Set(data.scopes || []);
    html += renderScopeGrid(effSet, false);

    // Groups
    html += '<h3 style="margin-top:24px;">Group Memberships</h3>';
    html += '<div style="margin:8px 0;">';
    (data.groups || []).forEach(function (g) { html += '<span class="tag tag-group">' + esc(g.name) + '</span>'; });
    if (!(data.groups || []).length) html += '<span style="color:var(--text-muted);">No groups</span>';
    html += '</div>';

    // Direct Roles
    html += '<h3 style="margin-top:16px;">Direct Roles</h3>';
    html += '<div style="margin:8px 0;">';
    (data.direct_roles || []).forEach(function (r) {
      html += '<span class="tag tag-role">' + esc(r.name) + '<span class="tag-remove" onclick="removeUserRole(\'' + userId + '\',\'' + r.id + '\')">&times;</span></span>';
    });
    if (!(data.direct_roles || []).length) html += '<span style="color:var(--text-muted);">No direct roles</span>';
    html += '</div>';
    html += '<div class="add-widget"><select id="add-user-role">';
    _roles.forEach(function (r) { html += '<option value="' + r.id + '">' + r.name + '</option>'; });
    html += '</select><button onclick="addUserRole(\'' + userId + '\')">Assign Role</button></div>';

    // Permission Overrides
    html += '<h3 style="margin-top:24px;">Permission Overrides</h3>';
    html += '<p style="color:var(--text-muted);font-size:0.85rem;">Grant or deny specific scopes for this user (overrides roles).</p>';
    var overrides = data.overrides || [];
    if (overrides.length) {
      overrides.forEach(function (o) {
        var cls = o.granted ? 'granted' : 'denied';
        html += '<div class="scope-check ' + cls + '" style="margin:4px 0;display:inline-flex;">' + o.scope + ' (' + (o.granted ? 'GRANT' : 'DENY') + ') <span class="tag-remove" onclick="removeUserPerm(\'' + userId + '\',\'' + o.id + '\')">&times;</span></div> ';
      });
    }
    html += '<div class="add-widget"><select id="add-perm-scope">';
    _allScopes.forEach(function (s) { html += '<option value="' + s + '">' + s + '</option>'; });
    html += '</select><select id="add-perm-granted"><option value="true">Grant</option><option value="false">Deny</option></select>';
    html += '<button onclick="addUserPerm(\'' + userId + '\')">Add Override</button></div>';

    main.innerHTML = html;
  }

  window.addUserRole = async function (uid) { await postJSON('/admin/rbac/users/' + uid + '/roles', { role_id: document.getElementById('add-user-role').value }); showUser(uid, ''); };
  window.removeUserRole = async function (uid, rid) { await del('/admin/rbac/users/' + uid + '/roles/' + rid); showUser(uid, ''); };
  window.addUserPerm = async function (uid) {
    var scope = document.getElementById('add-perm-scope').value;
    var granted = document.getElementById('add-perm-granted').value === 'true';
    await postJSON('/admin/rbac/users/' + uid + '/permissions', { scope: scope, granted: granted });
    showUser(uid, '');
  };
  window.removeUserPerm = async function (uid, pid) { await del('/admin/rbac/users/' + uid + '/permissions/' + pid); showUser(uid, ''); };

  function esc(s) { var d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
