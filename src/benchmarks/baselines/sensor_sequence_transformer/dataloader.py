"""Independent dataloader for the Sensor-sequence Transformer baseline."""

from __future__ import annotations

from src.benchmarks.baselines._shared import BaselineDataConfig, BaselineTaskData
from src.benchmarks.baselines.sensor_summary_model.dataloader import SensorSummaryDataLoader


class SensorSequenceDataLoader:
    """
    Load fixed-bin pre-target temporal sequences for Sensor-sequence Transformer.

    Sensor-sequence Transformer uses the same leakage-controlled, payload-backed GreenSeeker and UAV-MS
    observations as Sensor-summary model, but consumes the fixed sequence-bin columns rather than
    only summary statistics.
    """

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self._loader = SensorSummaryDataLoader(config)

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Sensor-sequence Transformer train/validation/test frames for one task."""

        return self._loader.load_task(task_name)
