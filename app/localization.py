"""Web-only language adapter; the research language engine remains unchanged."""
import re

ERRORS = {
 'Route inconnue.': 'Unknown route.', 'Origine refusée.': 'This origin is not allowed.',
 'JSON requis.': 'JSON content is required.',
 'Requête trop longue ou vide.': 'The request is too large or empty.',
 'La commande doit contenir entre 1 et 500 caractères.': 'The command must contain between 1 and 500 characters.',
 'Requête invalide.': 'Invalid request.', 'Erreur interne.': 'Internal error.',
 'La durée maximale est de 60 secondes.': 'The maximum duration is 60 seconds.',
}

def canonical_command(command, language):
    if language != 'en':return command
    normalized=command.strip().rstrip('.').casefold()
    if normalized in ('where is the ball?', 'where is the ball'):
        return 'Où est la balle ?'
    match=re.fullmatch(r'lift the ball to (\d+(?:[.,]\d+)?) (?:metres?|meters?|m)',normalized)
    if match:return f'Soulève la balle à {match[1].replace(",", ".")} mètres'
    match=re.fullmatch(r'advance time by (\d+(?:[.,]\d+)?) (?:seconds?|s)',normalized)
    if match:return f'Avance le temps de {match[1].replace(",", ".")} secondes'
    return command

def answer_in_language(answer, language):
    if language != 'en' or answer == 'OK.':return answer
    if answer == 'Balle est sur sol.':return 'The ball is on the ground.'
    match=re.fullmatch(r'Balle se trouve à (.+) m de hauteur\.',answer)
    if match:return f'The ball is at a height of {match[1]} m.'
    return 'Engine response (French): ' + answer

def error_in_language(error, language):
    if error in RUNTIME_ERRORS:
        return error if language == 'en' else RUNTIME_ERRORS[error]
    if error == 'The maximum duration is 60 seconds.':
        return error if language == 'en' else 'La durée maximale est de 60 secondes.'
    return ERRORS.get(error,'Invalid request or unsupported command. See the laboratory command guide.') if language == 'en' else error

# Runtime boundary messages are authored in English; this is the French translation layer.
RUNTIME_ERRORS = {
 'Service temporarily unavailable.': 'Service temporairement indisponible.',
 'Invalid public host.': 'Hôte public invalide.',
 'Method not allowed.': 'Méthode non autorisée.',
 'Internal server error.': 'Erreur interne du serveur.',
 'Laboratory session capacity reached; retry after expiry.': 'Capacité des sessions atteinte ; réessayez après expiration.',
 'Laboratory session budget exhausted; start a new browser session or wait for expiry.': 'Budget de la session épuisé ; ouvrez une nouvelle session de navigation ou attendez son expiration.',
 'Laboratory request budget exceeded; retry shortly.': 'Budget de requêtes dépassé ; réessayez dans un instant.',
 'Laboratory busy; retry shortly.': 'Laboratoire occupé ; réessayez dans un instant.',
 'Session already executing; retry shortly.': 'Session déjà en cours d’exécution ; réessayez dans un instant.',
 'Numeric parameters must be finite and within ±1000000.': 'Les paramètres numériques doivent être finis et compris entre −1000000 et 1000000.',
}
