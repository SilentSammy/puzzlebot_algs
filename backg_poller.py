import threading
from time import sleep
import numpy as np

import threading
from queue import Queue, Full, Empty
import numpy as np

class BackgroundPoller:
    def __init__(self, max_workers=1):
        self._result = None
        self._lock = threading.Lock()
        self._queue = Queue(maxsize=max_workers)
        self._max_workers = max_workers
        self._threads = []
        self._started = False

    def _worker(self):
        while True:
            func = self._queue.get()
            if func is None:
                break  # Sentinel to shut down
            value = func()
            with self._lock:
                self._result = value
            self._queue.task_done()

    def _start_threads(self):
        if not self._started:
            for _ in range(self._max_workers):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self._threads.append(t)
            self._started = True

    def poll(self, func):
        self._start_threads()
        with self._lock:
            result = self._result
        try:
            self._queue.put_nowait(func)
        except Full:
            # Queue is full, refuse to queue more
            pass
        return result

    @staticmethod
    def composite(drawing_frame, annot):
        """Overlay non-black pixels from annot onto drawing_frame in-place."""
        if drawing_frame is not None and annot is not None:
            mask = np.any(annot != 0, axis=2)
            drawing_frame[mask] = annot[mask]

    def poll_with_annotated(self, frame, drawing_frame, func):
        def worker_func():
            annot = np.zeros_like(frame)
            result = func(annot)
            return result, annot

        result = self.poll(worker_func)
        if result is not None:
            value, annotated_frame = result
            self.composite(drawing_frame, annotated_frame)
            return value
        return None
