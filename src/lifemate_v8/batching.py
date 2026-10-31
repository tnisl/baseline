"""Batch ready node requests across independent samples; one owner for vLLM."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Lock
from time import monotonic


class BatchedLLM:
    def __init__(self, model, batch_size=4, wait_ms=10):
        if batch_size < 1 or wait_ms < 0:
            raise ValueError('Invalid batching limits')
        self.model, self.batch_size, self.wait_ms = model, batch_size, wait_ms
        self.queue, self.lock = Queue(), Lock()
        self.closed = False
        self.thread = Thread(target=self._serve, daemon=True)
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.model, name)

    def count(self, messages):
        with self.lock:
            return self.model.count(messages)

    def complete(self, messages, max_new_tokens=512, output_schema=None):
        future = Future()
        if self.closed:
            raise RuntimeError('Batch scheduler closed')
        self.queue.put((future, (messages, max_new_tokens, output_schema), monotonic()))
        return future.result()

    def _serve(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = monotonic() + self.wait_ms / 1000
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get(timeout=max(0, deadline - monotonic()))
                except Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                batch.append(item)
            try:
                # Protect the tokenizer as well as the non-thread-safe engine.
                with self.lock:
                    outputs = self.model.complete_batch([item[1] for item in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError('Batch output cardinality mismatch')
                for (future, _, submitted), output in zip(batch, outputs):
                    output = dict(output, latency_s=monotonic() - submitted)
                    future.set_result(output)
            except BaseException as exc:
                for future, _, _ in batch:
                    future.set_exception(exc)

    def close(self):
        self.closed = True
        self.queue.put(None)
        self.thread.join()


def iter_outcomes(agent, rows, transcripts, workers=1):
    """Yield in input order; each run owns its environment and local state."""
    if workers == 1:
        for row in rows:
            yield agent.run(row, transcripts[row['id']])
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded submission avoids starting timers for the whole dataset.
        pending = {}
        for index, row in enumerate(rows[:workers]):
            pending[index] = pool.submit(agent.run, row, transcripts[row['id']])
        for index in range(len(rows)):
            outcome = pending.pop(index).result()
            next_index = index + workers
            if next_index < len(rows):
                row = rows[next_index]
                pending[next_index] = pool.submit(agent.run, row, transcripts[row['id']])
            yield outcome
