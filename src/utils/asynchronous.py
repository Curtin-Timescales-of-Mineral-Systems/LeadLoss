import traceback
from enum import Enum
from multiprocessing import Process, Queue, Value
import queue as pyqueue

from PyQt5.QtCore import pyqtSignal, QThread


class SignalType(Enum):
    NEW_TASK = 1,
    PROGRESS = 2,
    COMPLETED = 3,
    CANCELLED = 4,
    ERRORED = 5,
    SKIPPED = 6,


"""
An object that the child process uses to send information to the PyQT thread
"""
class ProcessSignals():
    def __init__(self):
        self.queue = Queue()
        self._halt = Value('i', 0)

    def newTask(self, *args):
        self.queue.put((SignalType.NEW_TASK, *args))

    def progress(self, *args):
        self.queue.put((SignalType.PROGRESS, *args))

    def completed(self, *args):
        self.queue.put((SignalType.COMPLETED, *args))

    def cancelled(self, *args):
        self.queue.put((SignalType.CANCELLED, *args))

    def errored(self, *args):
        self.queue.put((SignalType.ERRORED, *args))

    def halt(self):
        return self._halt.value == 1

    def setHalt(self):
        self._halt.value = 1

    def skipped(self, sample_name, skip_reason):
        self.queue.put((SignalType.SKIPPED, sample_name, skip_reason))


# Can't be part of AsyncTask as this function must be picklable under windows:
# (see https://docs.python.org/2/library/multiprocessing.html#windows)
def wrappedJobFn(jobFn, processSignals, *args):
    try:
        jobFn(processSignals, *args)
    except Exception as e:
        traceback.print_exc()
        processSignals.errored(e)


class AsyncTask(QThread):
    """
    Runs a job in a separate process and forwards messages from the job to the
    main thread through a pyqtSignal.
    """
    msg_from_job = pyqtSignal(object)

    def __init__(self, pyqtSignals, jobFn, *args):
        super().__init__()
        self.jobFn = jobFn
        self.args = args
        self.pyqtSignals = pyqtSignals
        self.processSignals = ProcessSignals()

        self._proc = None
        self.running = True

    def run(self):
        self.running = True

        p = Process(target=wrappedJobFn, args=(self.jobFn, self.processSignals, *self.args))
        self._proc = p
        p.start()

        try:
            while self.running:
                try:
                    output = self.processSignals.queue.get(timeout=0.1)
                except pyqueue.Empty:
                    # If the child died without telling us, stop the thread.
                    if not p.is_alive():
                        # Emit something so UI can unwind cleanly
                        self.pyqtSignals.processingErrored.emit((RuntimeError("Worker process exited unexpectedly"),))
                        break
                    continue

                self._processOutput(output)

        finally:
            # Stop loop
            self.running = False

            # Give the process a moment to exit cleanly; then kill if needed
            try:
                if p.is_alive():
                    p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
                p.join(timeout=5.0)
            except Exception:
                pass

            # Close Queue resources to avoid semaphore/FD leaks
            try:
                self.processSignals.queue.close()
                self.processSignals.queue.cancel_join_thread()
            except Exception:
                pass

    def _processOutput(self, output):
        if output[0] is SignalType.NEW_TASK:
            self.pyqtSignals.processingNewTask.emit(output[1])

        if output[0] is SignalType.PROGRESS:
            self.pyqtSignals.processingProgress.emit(output[1:])
            return

        if output[0] is SignalType.COMPLETED:
            self.pyqtSignals.processingCompleted.emit(output[1:])
            self.running = False
            return

        if output[0] is SignalType.CANCELLED:
            self.pyqtSignals.processingCancelled.emit()
            self.running = False
            return

        if output[0] is SignalType.ERRORED:
            self.pyqtSignals.processingErrored.emit(output[1:])
            self.running = False
            return

        if output[0] is SignalType.SKIPPED:
            sample_name = output[1]
            skip_reason = output[2]
            self.pyqtSignals.processingSkipped.emit(sample_name, skip_reason)
            return

    def halt(self):
        # Ask worker to stop (if it cooperates)
        self.processSignals.setHalt()

        # Also stop our thread loop
        self.running = False

        # If the process keeps running, it will be terminated in finally.
