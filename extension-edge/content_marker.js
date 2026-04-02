// CLS++ — runs on localhost prototype pages (home index.html, memory.html)
// Syncs the extension UID into the page's localStorage so memory.html can use it
document.documentElement.setAttribute('data-clspp-extension', '1');

chrome.runtime.sendMessage({ type: 'GET_UID' }, (resp) => {
  if (resp && resp.uid) {
    localStorage.setItem('clspp_uid', resp.uid);
    window.dispatchEvent(new CustomEvent('clspp-uid', { detail: resp.uid }));
  }
});
