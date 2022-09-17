from threading import Lock, Thread
from typing import Tuple, Union, Any, Callable
from copy import copy

class MessageBroker:
    id: Any
    _mutex: Lock
    _message: Union[Any, None]
    _worker: Thread
    _result: Any
    _started: bool
    _finished: bool
    _func: Callable
    _args: Tuple[str]

    def __init__(self, id, target: Callable):
        self.id = id
        self._mutex = Lock()
        self._message = None
        self._finished = False
        self._started = False
        self._result = None
        self._func = target
        self._args = target.__code__.co_varnames

        assert "set_message" in self._args
        assert "notify_end" in self._args
    
    def run(self, *args, **kwargs):
        kwargs.update({
            "set_message": self._set_message,
            "notify_end": self._set_result,
        })
        net_args_key = [
            elem
            for elem in self._args
            if elem not in kwargs.keys()
        ]
        kwargs.update({
            key: arg
            for key, arg in zip(net_args_key, args)
        })
        with self._mutex:
            self._worker = Thread(
                target = self._func,
                kwargs = kwargs,
            )
            self._worker.start()
            self._started = True
        return

    def _set_message(self, message: str):
        with self._mutex:
            self._message = copy(message)
        return

    def get_message(self):
        with self._mutex:
            message = copy(self._message)
        return message
    
    def _set_result(self, result):
        with self._mutex:
            self._result = copy(result)
            self._finished = True
        return
    
    def get_result(self):
        with self._mutex:
            result = copy(self._result)
        return result
            
    def fetch_finished(self):
        with self._mutex:
            finished_flag = self._finished
        return finished_flag
    
    def fetch_started(self):
        with self._mutex:
            started_flag = self._started
        return started_flag
    