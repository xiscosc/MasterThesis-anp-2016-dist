import sys
import time


class Timer:
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        sys.stdout.write('%s...' % self.task)
        sys.stdout.flush()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(' Done! Took %.03f seconds' % self.interval)


class Verbose:
    def __init__(self, start_text, done_text=' Done'):
        self.start_text = start_text
        self.done_text = done_text

    def __enter__(self):
        sys.stdout.write(self.start_text)
        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        print(self.done_text)
