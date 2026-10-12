// User-stepped illustration. No scientific task or metric is computed in the browser.
document.querySelectorAll('[data-transaction]').forEach(study => {
  const controls = study.querySelector('.transaction-controls');
  const scene = study.querySelector('.transaction-scene');
  const result = study.querySelector('[data-transaction-result]');
  controls.hidden = false;
  scene.hidden = false;
  controls.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      const accepted = button.dataset.decision === 'accept';
      scene.classList.toggle('is-accepted', accepted);
      result.textContent = accepted ? result.dataset.accept : result.dataset.reject;
      controls.querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    });
  });
});
