import threading
import time
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Simulate a shared resource that requires thread-safe access
class SharedCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._counter = 0

    def increment(self):
        with self._lock:
            old_value = self._counter
            # Simulate some processing delay
            time.sleep(0.01)
            self._counter = old_value + 1
            logging.info(f'Counter incremented from {old_value} to {self._counter}')
            return self._counter

    def get(self):
        with self._lock:
            return self._counter

shared_counter = SharedCounter()

# To simulate limited resource pool, e.g., DB connections
# Use a semaphore to limit concurrent access
MAX_CONCURRENT_REQUESTS = 5
concurrency_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


@app.route('/increment', methods=['POST'])
def increment_counter():
    if not concurrency_semaphore.acquire(blocking=False):
        # Too many concurrent requests, return 429 Too Many Requests
        logging.warning('Too many concurrent requests - throttling')
        return jsonify({'error': 'Too many concurrent requests, please try again later.'}), 429

    try:
        # Validate input as an example (idempotency token)
        request_json = request.get_json(force=True, silent=True)
        if request_json is None:
            logging.warning('Invalid JSON payload')
            return jsonify({'error': 'Invalid JSON payload'}), 400

        idempotency_key = request_json.get('idempotency_key')
        if idempotency_key is None or not isinstance(idempotency_key, str):
            logging.warning('Missing or invalid idempotency_key')
            return jsonify({'error': 'Missing or invalid idempotency_key'}), 400

        # For this example, we don't store keys, but in production an idempotency store would be used

        # Critical section: increment shared counter
        new_value = shared_counter.increment()

        return jsonify({'counter': new_value}), 200

    except Exception as e:
        logging.error('Unhandled exception in /increment endpoint', exc_info=e)
        return jsonify({'error': 'Internal server error'}), 500

    finally:
        concurrency_semaphore.release()

if __name__ == '__main__':
    # Run with threaded=True to enable concurrent requests
    app.run(threaded=True, port=5000)
