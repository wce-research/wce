// Runs before CSS is loaded to apply a saved preference without a theme flash.
(() => {
  let preference;
  try { preference = localStorage.getItem('wce.theme'); } catch {}
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  document.documentElement.dataset.theme = ['light', 'dark'].includes(preference)
    ? preference : (system.matches ? 'dark' : 'light');
  system.addEventListener('change', event => {
    let saved;
    try { saved = localStorage.getItem('wce.theme'); } catch {}
    if (!['light', 'dark'].includes(saved)) {
      document.documentElement.dataset.theme = event.matches ? 'dark' : 'light';
      document.dispatchEvent(new Event('wce:theme'));
    }
  });
})();
