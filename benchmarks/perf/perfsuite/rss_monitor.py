"""Sample another process's resident set from outside it.

    python -m perfsuite.rss_monitor <pid> <period seconds>

Commands arrive one per line on stdin; each is answered with one line on stdout:

    start   reset the peak to the current resident set and sample every <period>   -> "ok"
    stop    stop sampling                                    -> "<peak bytes> <samples taken>"
    quit    (or end of input) exit

The worker cannot sample itself while it featurizes: the Nyxus extension holds the interpreter
lock for the whole call, so a sampling thread inside the worker never runs until the call has
returned and its allocations are gone. This process has its own interpreter, so it keeps sampling.

Between samples the thread sleeps with time.sleep, which keeps to the period on Windows too; a
timed Event.wait or Lock.acquire there rounds up to the 15.6 ms system timer tick.
"""

from __future__ import annotations

import sys
import threading
import time

import psutil


class Monitor:
    """Peak of `rss()` between start() and stop(); sample_once() is one sampling step.

    Every start() and stop() begins a new generation. A sample is read outside the lock, so a
    stop and the next start can both happen while it is read; it only counts toward the
    generation it was read in, so one repeat's peak never lands on the next.
    """

    def __init__(self, rss):
        self._rss = rss
        self._lock = threading.Lock()
        self.active = False
        self.gen = 0
        self.peak = 0
        self.samples = 0

    def start(self):
        base = self._rss()
        with self._lock:
            self.gen += 1
            self.peak, self.samples, self.active = base, 0, True

    def stop(self):
        last = self._rss()
        with self._lock:
            self.gen += 1
            self.active = False
            return max(self.peak, last), self.samples

    def sample_once(self) -> bool:
        """Take one sample if sampling; False when not sampling."""
        with self._lock:
            if not self.active:
                return False
            gen = self.gen
        r = self._rss()
        with self._lock:
            if self.active and self.gen == gen:
                self.samples += 1
                if r > self.peak:
                    self.peak = r
        return True


def sample_loop(mon: Monitor, period: float, wake: threading.Event, done: threading.Event, sleep=time.sleep):
    """Sample while `mon` is active, sleeping `period` between samples; wait on `wake` while it is not."""
    while not done.is_set():
        if not mon.sample_once():
            wake.wait()
            wake.clear()
            continue
        sleep(period)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    proc = psutil.Process(int(argv[0]))
    period = float(argv[1])

    def rss() -> int:
        try:
            return proc.memory_info().rss
        except psutil.Error:
            return 0

    mon = Monitor(rss)
    wake = threading.Event()
    done = threading.Event()
    t = threading.Thread(target=sample_loop, args=(mon, period, wake, done), daemon=True)
    t.start()
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "start":
            mon.start()
            wake.set()
            print("ok", flush=True)
        elif cmd == "stop":
            peak, n = mon.stop()
            print(peak, n, flush=True)
        elif cmd == "quit":
            break
    done.set()
    wake.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
