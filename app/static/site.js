const menu = document.querySelector('.menu');
const navigation = document.querySelector('#navigation');
const groups = [...document.querySelectorAll('.nav-group')];
const desktop = window.matchMedia('(min-width: 1101px)');
const hover = window.matchMedia('(hover: hover)');

function closeGroup(group) {
  group.classList.remove('is-open');
  group.querySelector('.nav-toggle').setAttribute('aria-expanded', 'false');
  group.querySelector('.nav-panel').hidden = true;
}
function closeGroups() { groups.forEach(closeGroup); }
function openGroup(group) {
  groups.filter(item => item !== group).forEach(closeGroup);
  group.classList.add('is-open');
  group.querySelector('.nav-toggle').setAttribute('aria-expanded', 'true');
  group.querySelector('.nav-panel').hidden = false;
}
function closeNavigation() {
  closeGroups();
  navigation.classList.remove('is-open');
  menu.setAttribute('aria-expanded', 'false');
  menu.textContent = menu.dataset.openLabel;
}
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  if (!open) return closeNavigation();
  menu.setAttribute('aria-expanded', 'true');
  navigation.classList.add('is-open');
  menu.textContent = menu.dataset.closeLabel;
});
groups.forEach(group => {
  const toggle = group.querySelector('.nav-toggle');
  group.addEventListener('pointerenter', event => {
    if (desktop.matches && hover.matches && event.pointerType !== 'touch') openGroup(group);
  });
  group.addEventListener('pointerleave', () => {
    if (desktop.matches && !group.contains(document.activeElement)) closeGroup(group);
  });
  group.addEventListener('focusin', event => {
    if (desktop.matches && event.target.matches('a')) openGroup(group);
  });
  group.addEventListener('focusout', event => {
    if (!group.contains(event.relatedTarget)) closeGroup(group);
  });
  toggle.addEventListener('click', () => {
    toggle.getAttribute('aria-expanded') === 'true' ? closeGroup(group) : openGroup(group);
  });
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  const openGroupElement = groups.find(group => group.classList.contains('is-open'));
  if (openGroupElement) {
    closeGroup(openGroupElement);
    openGroupElement.querySelector('.nav-toggle').focus();
  } else if (menu.getAttribute('aria-expanded') === 'true') {
    closeNavigation(); menu.focus();
  }
});
document.addEventListener('pointerdown', event => {
  if (!event.target.closest('.header')) closeNavigation();
});
desktop.addEventListener('change', closeNavigation);

const themeButton = document.querySelector('.theme-toggle');
function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const label = dark ? themeButton.dataset.lightLabel : themeButton.dataset.darkLabel;
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.setAttribute('aria-label', label);
  themeButton.querySelector('.theme-label').textContent = label;
  themeButton.querySelector('.theme-icon').textContent = dark ? '☀' : '☾';
}
themeButton.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('wce.theme', next); } catch {}
  updateThemeButton();
});
document.addEventListener('wce:theme', updateThemeButton);
updateThemeButton();

document.querySelectorAll('[data-language]').forEach(link => {
  link.addEventListener('click', () => {
    document.cookie = `wce_language=${link.dataset.language}; Path=/; Max-Age=31536000; SameSite=Lax${location.protocol === 'https:' ? '; Secure' : ''}`;
  });
});

document.querySelectorAll('[data-filter]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-filter]').forEach(item => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-pressed', String(item === button));
    });
    document.querySelectorAll('[data-topic]').forEach(item => {
      item.hidden = button.dataset.filter !== 'all' && item.dataset.topic !== button.dataset.filter;
    });
  });
});

// Filter the public architecture map locally. All cards remain readable without JS.
const mechanismSearch = document.querySelector('#mechanism-search');
if (mechanismSearch) {
  const controls = mechanismSearch.closest('.atlas-search');
  const cards = [...document.querySelectorAll('[data-mechanism]')];
  const count = controls.querySelector('.search-count');
  const empty = controls.querySelector('.search-empty');
  const normalise = text => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase();
  const index = cards.map(card => ({card, text: normalise(card.textContent)}));
  function filterMechanisms() {
    const terms = normalise(mechanismSearch.value.trim()).split(/\s+/).filter(Boolean);
    let visible = 0;
    for (const {card, text} of index) {
      card.hidden = !terms.every(term => text.includes(term));
      if (!card.hidden) visible++;
    }
    count.textContent = `${visible} / ${cards.length} ${count.dataset.countLabel}`;
    empty.hidden = visible !== 0;
  }
  controls.hidden = false;
  mechanismSearch.addEventListener('input', filterMechanisms);
  filterMechanisms();
}

// Search only the public static source inventory; never fetch repository files.
const sourceSearch = document.querySelector('#source-search');
if (sourceSearch) {
  const controls = sourceSearch.closest('.inventory-search');
  const groups = [...document.querySelectorAll('.inventory-group')];
  const rows = [...document.querySelectorAll('[data-source-module]')];
  const fold = value => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const indexed = rows.map(row => ({row, text: fold(row.textContent)}));
  const counter = controls.querySelector('.search-count');
  const empty = controls.querySelector('.search-empty');
  let initialStates = null;
  function searchSources() {
    const terms = fold(sourceSearch.value.trim()).split(/\s+/).filter(Boolean);
    if (terms.length && !initialStates) {
      initialStates = new Map([...document.querySelectorAll('.inventory-group, .inventory-symbols')].map(node => [node,node.open]));
    }
    let shown = 0;
    indexed.forEach(({row, text}) => {
      row.hidden = !terms.every(term => text.includes(term));
      if (!row.hidden) shown++;
      if (terms.length && !row.hidden) row.querySelectorAll('.inventory-symbols').forEach(node => {node.open = true;});
    });
    groups.forEach(group => {
      group.hidden = ![...group.querySelectorAll('[data-source-module]')].some(row => !row.hidden);
      if (terms.length && !group.hidden) group.open = true;
    });
    if (!terms.length && initialStates) {
      initialStates.forEach((open,node) => {node.open = open;});
      initialStates = null;
    }
    counter.textContent = `${shown} / ${rows.length} ${counter.dataset.countLabel}`;
    empty.hidden = shown !== 0;
  }
  controls.hidden = false;
  sourceSearch.addEventListener('input', searchSources);
  searchSources();
}
