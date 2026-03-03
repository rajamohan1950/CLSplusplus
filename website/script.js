// Mobile nav toggle
document.querySelector('.nav-toggle')?.addEventListener('click', () => {
  const links = document.querySelector('.nav-links');
  links.style.display = links.style.display === 'flex' ? 'none' : 'flex';
});

// Hide video overlay when user has a real video (remove overlay div or set display:none)
// For placeholder: show overlay by default. Remove this when you add your video URL.
const overlay = document.getElementById('video-overlay');
const video = document.getElementById('demo-video');
if (overlay && video) {
  // If video src is still placeholder, show overlay
  const isPlaceholder = video.src.includes('dQw4w9WgXcQ');
  if (isPlaceholder) {
    overlay.classList.add('visible');
  }
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});
