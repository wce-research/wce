/** Same-origin API boundary shared by frontend modules.
 * Authentication can later use server-managed HttpOnly cookies; no tokens in storage.
 */
export async function requestJSON(path, {method = 'GET', body, signal} = {}) {
  const response = await fetch(path, {
    method,
    credentials: 'same-origin',
    headers: body === undefined ? {} : {'Content-Type': 'application/json', 'X-WCE-Language': document.documentElement.lang},
    body: body === undefined ? undefined : JSON.stringify(body),
    signal
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
export const executeCommand = (command, {signal} = {}) => requestJSON('/api/command', {
  method: 'POST', body: {command}, signal
});
