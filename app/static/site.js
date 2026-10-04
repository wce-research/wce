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
    document.cookie = `wce_language=${link.dataset.language}; Path=/; Max-Age=31536000; SameSite=Lax`;
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
