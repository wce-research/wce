import { executeCommand } from '/api.js';

const form = document.querySelector('#form');
const input = document.querySelector('#command');
const send = document.querySelector('#send');
const history = document.querySelector('#history');

document.querySelectorAll('[data-command]').forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.command;
    input.focus();
  });
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  const command = input.value.trim();
  if (!command || send.disabled) return;
  send.disabled = true;
  try {
    const result = await executeCommand(command);
    const entry = document.createElement('div');
    entry.className = 'entry';
    const title = document.createElement('strong');
    title.textContent = command;
    const answer = document.createElement('span');
    answer.textContent = result.answer;
    entry.append(title, answer);
    history.prepend(entry);
    input.value = '';
  } catch (error) {
    const entry = document.createElement('div');
    entry.className = 'entry error';
    entry.textContent = error.message || (document.documentElement.lang === 'en' ? 'The service is temporarily unavailable.' : 'Le service est momentanément inaccessible.');
    history.prepend(entry);
  } finally {
    send.disabled = false;
    input.focus();
  }
});
