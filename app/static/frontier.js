// Public presentation controls; no model execution or scientific data rewriting.
const configuration = document.querySelector('.configuration-panel');
if (configuration) {
  const states = {
    primitives: {state: 'A', operator: 'I', transforms: ['translate(65 120)', 'translate(185 120)', 'translate(305 120)']},
    rotation: {state: 'B', operator: 'R', transforms: ['translate(65 120) rotate(45 32 32)', 'translate(185 120) rotate(45 32 32)', 'translate(305 120) rotate(45 32 32)']},
    composition: {state: 'C', operator: 'T ∘ R', transforms: ['translate(165 110) rotate(45 32 32)', 'translate(210 140) rotate(45 32 32)', 'translate(245 105) rotate(45 32 32)']},
  };
  configuration.querySelectorAll('[data-configuration]').forEach(button => {
    button.addEventListener('click', () => {
      const key = button.dataset.configuration;
      const selected = states[key];
      configuration.querySelectorAll('[data-configuration]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      configuration.querySelectorAll('[data-primitive]').forEach((item, index) => item.setAttribute('transform', selected.transforms[index]));
      configuration.querySelector('#configuration-state').textContent = selected.state;
      configuration.querySelector('#configuration-operator').textContent = selected.operator;
      configuration.querySelector('#configuration-description').textContent = configuration.dataset[`description${key[0].toUpperCase()}${key.slice(1)}`];
    });
  });
}
const metric = document.querySelector('#evidence-metric');
if (metric) {
  metric.addEventListener('change', () => {
    document.querySelectorAll('[data-metric-panel]').forEach(panel => { panel.hidden = panel.dataset.metricPanel !== metric.value; });
    document.querySelector('.evidence-current').textContent = metric.selectedOptions[0].textContent;
  });
}
const verdictFilters = document.querySelector('.verdict-filters');
if (verdictFilters) {
  const cards = [...document.querySelectorAll('.hypothesis-card')];
  const counter = verdictFilters.querySelector('.verdict-filter-count');
  function selectVerdict(key) {
    verdictFilters.querySelectorAll('[data-verdict-filter]').forEach(button => {
      const selected = button.dataset.verdictFilter === key;
      button.setAttribute('aria-pressed', String(selected));
      button.classList.toggle('active', selected);
    });
    let visible = 0;
    cards.forEach(card => {
      card.hidden = key !== 'all' && !card.classList.contains('verdict-' + key);
      if (!card.hidden) visible++;
    });
    counter.textContent = `${visible} / ${cards.length} ${counter.dataset.label}`;
  }
  verdictFilters.hidden = false;
  verdictFilters.querySelectorAll('[data-verdict-filter]').forEach(button => {
    button.addEventListener('click', () => selectVerdict(button.dataset.verdictFilter));
  });
  selectVerdict('all');
}
