import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class ExecutorRegistry:
    _registry: Dict[str, Type["BaseExecutorBase"]] = {}

    @classmethod
    def register(cls, name):
        def decorator(executor_cls):
            cls._registry[name] = executor_cls
            return executor_cls

        return decorator

    @classmethod
    def create(
        cls, name: str, cfg: Any, test_data: Any, train_data: Any, sampling_params: Any
    ) -> "BaseExecutorBase":
        executor_cls = cls._registry.get(name)
        if executor_cls is None:
            raise ValueError(f"Unknown executor: {name}")
        return executor_cls(cfg, test_data, train_data, sampling_params)


class BaseExecutorBase:
    def __init__(
        self,
        cfg: Any,
        test_data: Any,
        sampling_params: Any,
        train_data: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.test_data = test_data
        self.train_data = train_data
        self.sampling_params = sampling_params

    def execute_steps(self) -> None:
        raise NotImplementedError()
