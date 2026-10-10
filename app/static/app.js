import { executeCommand } from '/api.js';

const form = document.querySelector('#form');
const input = document.querySelector('#command');
const send = document.querySelector('#send');
const history = document.querySelector('#history');
const status = document.querySelector('#lab-status');
const counter = document.querySelector('#command-count');
const examples = [...document.querySelectorAll('[data-command]')];
const english = document.documentElement.lang === 'en';
const labels = english ? {
  ready: 'Ready. Choose an example or enter a supported command.',
  waiting: 'Executing the command…', done: 'Command completed. Read the engine response below.',
  error: 'Command unsuccessful. Read the message below.',
  timeout: 'The request timed out. Check the world state before sending another action.',
  network: 'The service could not be reached. Your command has been kept so you can retry.',
  invalid: 'The service returned an unreadable response. Check the world state before retrying.',
} : {
  ready: 'Prêt. Choisissez un exemple ou saisissez une commande acceptée.',
  waiting: 'Exécution de la commande…', done: 'Commande exécutée. Lisez la réponse du moteur ci-dessous.',
  error: 'La commande n’a pas abouti. Lisez le message ci-dessous.',
  timeout: 'Le délai de réponse est dépassé. Vérifiez l’état du monde avant de renvoyer une action.',
  network: 'Le service est inaccessible. Votre commande est conservée pour réessayer.',
  invalid: 'Le service a renvoyé une réponse illisible. Vérifiez l’état du monde avant de réessayer.',
};
function updateCounter() {counter.textContent = `${input.value.length} / 500`;}
input.addEventListener('input', updateCounter);
examples.forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.command;
    updateCounter();input.focus();
  });
});
function addEntry(command, message, error = false) {
  history.querySelector('.initial')?.remove();
  const entry = document.createElement('div');
  entry.className = error ? 'entry error' : 'entry';
  const title = document.createElement('strong');title.textContent = command;
  const answer = document.createElement('span');answer.textContent = message;
  entry.append(title, answer);history.prepend(entry);
  while (history.children.length > 20) history.lastElementChild.remove();
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const command = input.value.trim();
  if (!command || send.disabled) return;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  send.disabled = input.disabled = true;
  examples.forEach(button => {button.disabled = true;});
  form.setAttribute('aria-busy', 'true');status.textContent = labels.waiting;
  try {
    const result = await executeCommand(command, {signal: controller.signal});
    if (typeof result.answer !== 'string') throw new Error(labels.invalid);
    addEntry(command, result.answer);
    input.value = '';status.textContent = labels.done;
  } catch (error) {
    const message = error.name === 'AbortError' ? labels.timeout : error instanceof SyntaxError ? labels.invalid : error instanceof TypeError ? labels.network : error.message || labels.invalid;
    addEntry(command, message, true);status.textContent = labels.error;
  } finally {
    clearTimeout(timer);
    send.disabled = input.disabled = false;
    examples.forEach(button => {button.disabled = false;});
    form.setAttribute('aria-busy', 'false');updateCounter();input.focus();
  }
});
status.textContent = labels.ready;
updateCounter();
