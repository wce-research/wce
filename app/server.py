"""Single-process Waitress lifecycle with explicit bounded drain."""
import logging
import os
import signal
import threading
import time

from waitress import create_server
from waitress.task import ThreadedTaskDispatcher


class BoundedDispatcher(ThreadedTaskDispatcher):
    """Fixed queue; overflow is closed promptly rather than retained without bound."""
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def add_task(self, task):
        with self.lock:
            if len(self.queue) < self.limit:
                self.queue.append(task)
                self.queue_cv.notify()
                return
        task.cancel()


def build_server(application, host='127.0.0.1', port=0):
    config = application.config
    dispatcher = BoundedDispatcher(config.connections)
    dispatcher.set_thread_count(config.threads)
    try:
        return create_server(application,host=host,port=port,_dispatcher=dispatcher,
            threads=config.threads,connection_limit=config.connections,backlog=config.connections,
            channel_timeout=10,cleanup_interval=1,channel_request_lookahead=0,
            max_request_header_size=8192,max_request_body_size=4096,
            inbuf_overflow=4096,outbuf_overflow=1048576,outbuf_high_watermark=1048576,
            clear_untrusted_proxy_headers=True,trusted_proxy=None,
            url_scheme='https' if config.mode=='production' else 'http',
            asyncore_loop_timeout=1,expose_tracebacks=False,ident='WCE')
    except Exception:
        dispatcher.shutdown(timeout=1)
        raise


def close_server(server, application):
    application.ready = False
    # Stop accepting, but keep the I/O loop alive while active responses drain.
    server.accepting = False
    deadline = time.monotonic()+application.config.shutdown_seconds
    server.task_dispatcher.shutdown(cancel_pending=True,timeout=application.config.shutdown_seconds)
    # Allow already produced output to reach clients, bounded by the same deadline.
    while time.monotonic()<deadline and any(channel.total_outbufs_len for channel in list(server.active_channels.values())):
        time.sleep(.01)
    for channel in list(server.active_channels.values()):
        channel.close()
    server.close()


def serve(application):
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    host = os.environ.get('WCE_HOST','127.0.0.1')
    port = int(os.environ.get('WCE_PORT','8000'))
    application.validate_startup()
    server = build_server(application,host,port)
    stopped = threading.Event()
    def signal_stop(signum, frame):
        application.ready = False
        stopped.set()
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    thread = threading.Thread(target=server.run,name='wce-http',daemon=True)
    thread.start()
    logging.info('WCE listening on %s:%s',server.effective_host,server.effective_port)
    try:
        while not stopped.wait(.1):
            if not thread.is_alive():
                raise RuntimeError('HTTP event loop stopped unexpectedly')
    finally:
        close_server(server,application)
        thread.join(timeout=1)
        logging.info('WCE shutdown complete')
