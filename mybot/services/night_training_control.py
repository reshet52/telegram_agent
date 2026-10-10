"""Wall-clock and thermal limits; no MLX dependency in the Telegram process."""

import ctypes
import sys
import time


class BudgetExpired(Exception):
    pass


def thermal_reader():
    """NSProcessInfo.thermalState (nominal=0, fair=1, serious=2, critical=3)."""
    if sys.platform != 'darwin':
        return lambda: None
    try:
        ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        pointer = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(pointer)
        send_int = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(pointer)
        process = send_object(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
        selector = objc.sel_registerName(b'thermalState')
        def read():
            state = send_int(process, selector)
            return state if state in (0, 1, 2, 3) else None
        read()
        return read
    except (OSError, AttributeError, TypeError):
        return lambda: None


class NightGuard:
    def __init__(self, seconds, *, reserve=900, reader=None, clock=time.monotonic,
                 sleep=time.sleep, progress=None):
        if not 0 < seconds <= 86400:
            raise ValueError('Предел запуска должен быть от 0 до 24 часов.')
        self.clock, self.sleep = clock, sleep
        self.started = clock()
        self.deadline = self.started + seconds
        self.reserve = min(reserve, seconds / 4)
        self.reader = reader or thermal_reader()
        self.progress = progress or (lambda *_: None)
        self.stopping = False
        self.pause_seconds = 0
        self.pause_count = 0
        self.cycle_started = self.started
        self.monitor = 'macos' if self.reader() is not None else 'duty_cycle'

    def expired(self, training=False):
        return self.stopping or self.clock() >= self.deadline - (self.reserve if training else 30)

    def wait(self, training=True):
        if self.expired(training):
            raise BudgetExpired()
        state = self.reader()
        fallback = state is None and self.clock() - self.cycle_started >= 1200
        if state is not None and state < 2:
            return
        if state is None and not fallback:
            return
        start = self.clock()
        self.pause_count += 1
        while True:
            self.progress('cooling', state)
            if self.expired(training):
                self.pause_seconds += self.clock() - start
                raise BudgetExpired()
            if fallback:
                if self.clock() - start >= 300:
                    break
            else:
                state = self.reader()
                if state is not None and state < 2:
                    break
                if state is None:
                    fallback = True
                    self.monitor = 'duty_cycle'
            stop_at = self.deadline - (self.reserve if training else 30)
            self.sleep(min(10, max(.01, stop_at - self.clock())))
        self.pause_seconds += self.clock() - start
        self.cycle_started = self.clock()
        self.progress('training' if training else 'checking', state)
