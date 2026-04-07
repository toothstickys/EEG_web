from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import mne
import numpy as np


ICAScope = Literal["event", "session"]


@dataclass(frozen=True)
class ChannelSpec:
    range_start: int | None = None
    range_end: int | None = None
    selected_names: tuple[str, ...] = ()
    include_aux: bool = False

    @classmethod
    def from_ui(
        cls,
        range_start: int | None,
        range_end: int | None,
        selected_names: list[str] | tuple[str, ...] | None,
        include_aux: bool = False,
    ) -> "ChannelSpec":
        return cls(
            range_start=range_start,
            range_end=range_end,
            selected_names=tuple(selected_names or ()),
            include_aux=include_aux,
        )


@dataclass(frozen=True)
class ProcessedSpec:
    key: str
    band: str = "__all__"


@dataclass
class LabelInfo:
    kind: str
    summary: str
    discrete_label: str | None = None
    discrete_code: str | int | float | None = None
    continuous_values: np.ndarray | None = None
    continuous_step_sec: float | None = None
    multi_scores: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordIndex:
    record_id: str
    dataset: str
    subject: str
    session: str
    event: int
    raw_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    raw_fs: float | None
    processed_fs: float | None
    channel_names: tuple[str, ...]
    extra_channel_names: tuple[str, ...] = ()
    label_summary: str = ""
    source_paths: dict[str, str] = field(default_factory=dict)
    processed_options: tuple[str, ...] = ()
    default_processed_key: str = ""
    processed_kind: str = "signal"
    raw_source_note: str = ""
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset": self.dataset,
            "subject": self.subject,
            "session": self.session,
            "event": self.event,
            "raw_shape": " x ".join(str(v) for v in self.raw_shape),
            "processed_shape": " x ".join(str(v) for v in self.processed_shape),
            "raw_fs": self.raw_fs,
            "processed_fs": self.processed_fs,
            "label_summary": self.label_summary,
            "processed_kind": self.processed_kind,
            "raw_source_note": self.raw_source_note,
            "notes": self.notes,
        }


@dataclass
class SignalBlock:
    data: np.ndarray
    fs: float
    duration_sec: float
    channel_names: list[str]
    label_info: LabelInfo
    start_sec: float = 0.0
    units: str = "uV"
    source_note: str = ""


@dataclass
class FeatureBlock:
    data: np.ndarray
    window_sec: float
    band_names: list[str]
    feature_name: str
    channel_names: list[str]
    label_info: LabelInfo
    source_note: str = ""


@dataclass(frozen=True)
class ICAEventWindow:
    event: int
    start_sec: float
    stop_sec: float
    epoch_index: int | None = None
    label_summary: str = ""


@dataclass
class ICASourceBlock:
    inst: mne.io.BaseRaw | mne.BaseEpochs
    scope: ICAScope
    fs: float
    duration_sec: float
    channel_names: list[str]
    event_windows: list[ICAEventWindow]
    label_info: LabelInfo
    source_note: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ICAComponentInfo:
    index: int
    label: str
    probabilities: dict[str, float]
    suggested_reject: bool = False


@dataclass
class ICAResultBlock:
    scope: ICAScope
    fit_fs: float
    components: list[ICAComponentInfo]
    excluded_indices: list[int]
    source_preview: SignalBlock
    cleaned_preview: SignalBlock
    notes: list[str] = field(default_factory=list)
