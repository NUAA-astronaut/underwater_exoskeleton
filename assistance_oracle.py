"""Privileged diagnostic path, kept separate from deployable controllers."""
from dataclasses import dataclass
from assistance_controllers import AssistanceController


@dataclass(frozen=True)
class TruthObservation:
    time_s: float
    phase_rad: tuple[float, float]


class Oracle(AssistanceController):
    def update(self, truth: TruthObservation):
        return self.waveform(truth.time_s, truth.phase_rad)
