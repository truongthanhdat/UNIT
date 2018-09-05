import time

class Timer:
    def __init__(self):
        self._tic = time.time()

    def tic(self):
        self._tic = time.time()

    def toc(self):
        return time.time() - self._tic


