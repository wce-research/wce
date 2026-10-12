"""Explicit, validated runtime configuration; no forwarded-header inference."""
from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit


def integer(env, name, default, minimum, maximum):
    try:
        value = int(env.get(name, default))
    except ValueError as error:
        raise ValueError(f'{name} must be an integer') from error
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = 'development'
    public_origin: str = ''
    threads: int = 4
    connections: int = 128
    sessions: int = 100
    session_ttl: int = 900
    session_lifetime: int = 3600
    commands: int = 1000
    simulation_steps: int = 6000
    history: int = 128
    graph_nodes: int = 64
    api_rate: int = 50
    api_burst: int = 100
    shutdown_seconds: int = 4

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        mode = env.get('WCE_MODE', 'development')
        if mode not in ('development', 'production'):
            raise ValueError('WCE_MODE must be development or production')
        origin = env.get('WCE_PUBLIC_ORIGIN', '')
        if mode == 'production':
            parsed = urlsplit(origin)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                    or parsed.path or parsed.query or parsed.fragment or parsed.netloc != parsed.hostname or len(parsed.hostname) > 253
                    or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+', parsed.hostname)
                    or any(char.isspace() for char in origin)):
                raise ValueError('Production requires WCE_PUBLIC_ORIGIN=https://canonical-host (no path or port)')
        elif origin:
            raise ValueError('WCE_PUBLIC_ORIGIN is only valid in production mode')
        values = {
            'threads': ('WCE_THREADS',4,2,16), 'connections': ('WCE_CONNECTIONS',128,16,256),
            'sessions': ('WCE_MAX_SESSIONS',100,1,100), 'session_ttl': ('WCE_SESSION_TTL',900,10,3600),
            'session_lifetime': ('WCE_SESSION_LIFETIME',3600,10,3600),
            'commands': ('WCE_SESSION_COMMANDS',1000,1,1000),
            'simulation_steps': ('WCE_SESSION_STEPS',6000,1,6000),
            'history': ('WCE_HISTORY_LIMIT',128,1,128), 'graph_nodes': ('WCE_GRAPH_LIMIT',64,2,64),
            'api_rate': ('WCE_API_RATE',50,1,100), 'api_burst': ('WCE_API_BURST',100,1,200),
            'shutdown_seconds': ('WCE_SHUTDOWN_SECONDS',4,1,10),
        }
        result = cls(mode,origin,**{key:integer(env,*spec) for key,spec in values.items()})
        if result.session_ttl > result.session_lifetime or result.threads > result.connections:
            raise ValueError('Invalid session lifetime or connection/thread limits')
        return result
