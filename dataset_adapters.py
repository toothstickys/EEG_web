from __future__ import annotations

import csv
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from models import ChannelSpec, FeatureBlock, LabelInfo, ProcessedSpec, RecordIndex, SignalBlock


DATASET_ROOT = Path(os.environ.get("EEG_DATASET_ROOT", r"E:\BaiduNetdiskDownload\dataset"))
DEFAULT_ROOT = DATASET_ROOT

SEED_LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}
SEED_IV_LABEL_MAP = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
SEED_IV_SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
SEED_FEATURE_KEYS = ("de_LDS", "de_movingAve", "psd_LDS", "psd_movingAve")
SEED_VIG_FEATURE_KEYS = (
    "5bands:de_LDS",
    "5bands:de_movingAve",
    "5bands:psd_LDS",
    "5bands:psd_movingAve",
    "2hz:de_LDS",
    "2hz:de_movingAve",
    "2hz:psd_LDS",
    "2hz:psd_movingAve",
)
DEAP_EEG_CHANNELS = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]
DEAP_AUX_DISPLAY = ["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Resp", "Plet", "Temp"]
DEAP_RAW_AUX_PICKS = ["EXG1", "EXG2", "EXG3", "EXG4", "GSR1", "Resp", "Plet", "Temp"]
DEAP_PROCESSED_KEY = "processed_eeg"
SEED_VII_PROCESSED_KEY = "eeg_preprocessed"
SEED_VII_SPECIAL_START_TIMES = {
    "14_20221015_1.cnt": "14:25:34",
    "9_20221111_3.cnt": "14:01:27",
}
FIVE_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BANDS_2HZ = [f"{start}-{start + 1}Hz" for start in range(1, 50, 2)]


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _mat_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    return {name: tuple(shape) for name, shape, _ in whosmat(path)}


def _loadmat_variable(path: Path, variable_name: str) -> np.ndarray:
    return loadmat(path, variable_names=[variable_name])[variable_name]


def _read_channel_order(path: Path) -> list[str]:
    frame = pd.read_excel(path, header=None)
    names: list[str] = []
    for value in frame.iloc[:, 0].tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            names.append(text)
    return names


def _score_summary(scores: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.2f}" for key, value in scores.items())


def _make_record_id(dataset: str, subject: str, session: str, event: int) -> str:
    return f"{dataset}::{subject}::{session}::{event}"


def _apply_channel_spec(
    data: np.ndarray,
    channel_names: list[str],
    channel_spec: ChannelSpec,
) -> tuple[np.ndarray, list[str]]:
    if channel_spec.selected_names:
        selected = [name for name in channel_spec.selected_names if name in channel_names]
        if not selected:
            selected = channel_names
        indices = [channel_names.index(name) for name in selected]
    elif channel_spec.range_start is not None and channel_spec.range_end is not None:
        start = max(0, min(channel_spec.range_start, channel_spec.range_end))
        end = min(len(channel_names) - 1, max(channel_spec.range_start, channel_spec.range_end))
        indices = list(range(start, end + 1))
    else:
        indices = list(range(len(channel_names)))

    if data.ndim == 2:
        sliced = data[indices, :]
    else:
        sliced = data[indices, ...]
    return sliced, [channel_names[index] for index in indices]


class DatasetAdapter(ABC):
    dataset_name: str

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, RecordIndex] = {}

    def get_record(self, record_id: str) -> RecordIndex:
        return self.records[record_id]

    def get_band_options(self, record_id: str, processed_key: str) -> list[str]:
        return []

    def get_available_channel_names(self, record_id: str, include_aux: bool = False) -> list[str]:
        record = self.get_record(record_id)
        names = list(record.channel_names)
        if include_aux and record.extra_channel_names:
            names.extend(record.extra_channel_names)
        return names

    @abstractmethod
    def scan(self) -> list[RecordIndex]:
        raise NotImplementedError

    @abstractmethod
    def get_label(self, record_id: str) -> LabelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        raise NotImplementedError

    @abstractmethod
    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> SignalBlock | FeatureBlock:
        raise NotImplementedError


class DEAPAdapter(DatasetAdapter):
    dataset_name = "DEAP"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.raw_dir = root / "data_original"
        self.processed_dat_dir = root / "data_preprocessed_python"
        self.processed_mat_dir = root / "data_preprocessed_matlab"

    @staticmethod
    @lru_cache(maxsize=4)
    def _trial_labels(mat_path: str) -> np.ndarray:
        return loadmat(mat_path, variable_names=["labels"])["labels"]

    @staticmethod
    @lru_cache(maxsize=2)
    def _processed_dat(path: str) -> dict[str, np.ndarray]:
        with open(path, "rb") as handle:
            return pickle.load(handle, encoding="latin1")

    @staticmethod
    @lru_cache(maxsize=2)
    def _open_raw(path: str) -> mne.io.BaseRaw:
        return mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")

    @staticmethod
    @lru_cache(maxsize=16)
    def _trial_bounds(path: str) -> list[tuple[int, int]]:
        raw = DEAPAdapter._open_raw(path)
        events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")
        starts = events[events[:, 2] == 4][:40, 0]
        stops = events[events[:, 2] == 5][:40, 0]
        fs = int(raw.info["sfreq"])
        return [(max(0, int(start - 3 * fs)), int(stop)) for start, stop in zip(starts, stops)]

    def _deap_channel_names(self, include_aux: bool) -> list[str]:
        names = list(DEAP_EEG_CHANNELS)
        if include_aux:
            names.extend(DEAP_AUX_DISPLAY)
        return names

    def scan(self) -> list[RecordIndex]:
        if self.records:
            return list(self.records.values())

        for mat_file in sorted(self.processed_mat_dir.glob("s*.mat")):
            subject = mat_file.stem
            raw_path = self.raw_dir / f"{subject}.bdf"
            dat_path = self.processed_dat_dir / f"{subject}.dat"
            labels = self._trial_labels(str(mat_file))
            processed_shape = _mat_shapes(mat_file).get("data", (40, 40, 8064))
            for trial_idx in range(1, 41):
                scores = labels[trial_idx - 1]
                label_scores = {
                    "Valence": float(scores[0]),
                    "Arousal": float(scores[1]),
                    "Dominance": float(scores[2]),
                    "Liking": float(scores[3]),
                }
                self.records[_make_record_id(self.dataset_name, subject, subject, trial_idx)] = RecordIndex(
                    record_id=_make_record_id(self.dataset_name, subject, subject, trial_idx),
                    dataset=self.dataset_name,
                    subject=subject,
                    session=subject,
                    event=trial_idx,
                    raw_shape=(len(DEAP_EEG_CHANNELS), 63 * 512),
                    processed_shape=(len(DEAP_EEG_CHANNELS), processed_shape[2]),
                    raw_fs=512.0,
                    processed_fs=128.0,
                    channel_names=tuple(DEAP_EEG_CHANNELS),
                    extra_channel_names=tuple(DEAP_AUX_DISPLAY),
                    label_summary=_score_summary(label_scores),
                    source_paths={
                        "raw": str(raw_path),
                        "processed_dat": str(dat_path),
                        "processed_mat": str(mat_file),
                    },
                    processed_options=(DEAP_PROCESSED_KEY,),
                    default_processed_key=DEAP_PROCESSED_KEY,
                    processed_kind="signal",
                )
        return list(self.records.values())

    def get_label(self, record_id: str) -> LabelInfo:
        record = self.get_record(record_id)
        labels = self._trial_labels(record.source_paths["processed_mat"])
        scores = labels[record.event - 1]
        score_map = {
            "Valence": float(scores[0]),
            "Arousal": float(scores[1]),
            "Dominance": float(scores[2]),
            "Liking": float(scores[3]),
        }
        return LabelInfo(kind="multi_score", summary=_score_summary(score_map), multi_scores=score_map)

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        raw = self._open_raw(record.source_paths["raw"])
        start, stop = self._trial_bounds(record.source_paths["raw"])[record.event - 1]
        if channel_spec.include_aux:
            picks = DEAP_EEG_CHANNELS + DEAP_RAW_AUX_PICKS
            data = raw.get_data(picks=picks, start=start, stop=stop)
            channel_names = self._deap_channel_names(True)
            units = "mixed"
        else:
            data = raw.get_data(picks=DEAP_EEG_CHANNELS, start=start, stop=stop, units="uV")
            channel_names = self._deap_channel_names(False)
            units = "uV"
        data, channel_names = _apply_channel_spec(data, channel_names, channel_spec)
        fs = float(raw.info["sfreq"])
        return SignalBlock(
            data=data,
            fs=fs,
            duration_sec=data.shape[1] / fs,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            units=units,
            source_note="DEAP raw EEG from BDF Status 4->5 with 3s baseline.",
        )

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        processed = self._processed_dat(record.source_paths["processed_dat"])
        trial = record.event - 1
        data = processed["data"][trial]
        if channel_spec.include_aux:
            channel_names = self._deap_channel_names(True)
        else:
            data = data[: len(DEAP_EEG_CHANNELS), :]
            channel_names = self._deap_channel_names(False)
        data, channel_names = _apply_channel_spec(data, channel_names, channel_spec)
        return SignalBlock(
            data=np.asarray(data),
            fs=128.0,
            duration_sec=data.shape[1] / 128.0,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="DEAP preprocessed time-domain signals from python .dat.",
        )


class SEEDAdapter(DatasetAdapter):
    dataset_name = "SEED"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.preprocessed_dir = root / "Preprocessed_EEG"
        self.features_dir = root / "ExtractedFeatures"
        self.channel_names = _read_channel_order(root / "channel-order.xlsx")
        self.labels = loadmat(self.features_dir / "label.mat", variable_names=["label"])["label"].reshape(-1)

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_preprocessed(path: str, variable_name: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), variable_name)

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_feature(path: str, variable_name: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), variable_name)

    def scan(self) -> list[RecordIndex]:
        if self.records:
            return list(self.records.values())

        for pre_file in sorted(self.preprocessed_dir.glob("*.mat")):
            if "_" not in pre_file.stem:
                continue
            feature_file = self.features_dir / pre_file.name
            pre_shapes = _mat_shapes(pre_file)
            feature_shapes = _mat_shapes(feature_file)
            subject = pre_file.stem.split("_")[0]
            session = pre_file.stem
            raw_var_names = {
                event: next(name for name in pre_shapes if name.endswith(str(event)) and "_eeg" in name)
                for event in range(1, 16)
            }
            for event in range(1, 16):
                label_code = int(self.labels[event - 1])
                record_id = _make_record_id(self.dataset_name, subject, session, event)
                self.records[record_id] = RecordIndex(
                    record_id=record_id,
                    dataset=self.dataset_name,
                    subject=subject,
                    session=session,
                    event=event,
                    raw_shape=pre_shapes[raw_var_names[event]],
                    processed_shape=feature_shapes[f"de_LDS{event}"],
                    raw_fs=200.0,
                    processed_fs=1.0,
                    channel_names=tuple(self.channel_names),
                    label_summary=SEED_LABEL_MAP[label_code],
                    source_paths={
                        "raw": str(pre_file),
                        "raw_var": raw_var_names[event],
                        "processed": str(feature_file),
                    },
                    processed_options=SEED_FEATURE_KEYS,
                    default_processed_key="de_LDS",
                    processed_kind="feature",
                    raw_source_note="Lowest-level EEG available in dataset (no separate raw file).",
                )
        return list(self.records.values())

    def get_band_options(self, record_id: str, processed_key: str) -> list[str]:
        return ["__all__", *FIVE_BANDS]

    def get_label(self, record_id: str) -> LabelInfo:
        record = self.get_record(record_id)
        code = int(self.labels[record.event - 1])
        return LabelInfo(
            kind="discrete",
            summary=SEED_LABEL_MAP[code],
            discrete_label=SEED_LABEL_MAP[code],
            discrete_code=code,
        )

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        variable_name = record.source_paths["raw_var"]
        data = np.asarray(self._load_preprocessed(record.source_paths["raw"], variable_name))
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return SignalBlock(
            data=data,
            fs=200.0,
            duration_sec=data.shape[1] / 200.0,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note=record.raw_source_note,
        )

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> FeatureBlock:
        record = self.get_record(record_id)
        variable_name = f"{processed_spec.key}{record.event}"
        data = np.asarray(self._load_feature(record.source_paths["processed"], variable_name))
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        window_sec = (record.raw_shape[1] / 200.0) / max(1, data.shape[1])
        return FeatureBlock(
            data=data,
            window_sec=window_sec,
            band_names=list(FIVE_BANDS),
            feature_name=processed_spec.key,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED feature tensor from ExtractedFeatures.",
        )


class SEEDIVAdapter(DatasetAdapter):
    dataset_name = "SEED-IV"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.raw_dir = root / "eeg_raw_data"
        self.feature_dir = root / "eeg_feature_smooth"
        self.channel_names = _read_channel_order(root / "Channel Order.xlsx")

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_raw_trial(path: str, variable_name: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), variable_name)

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_feature_trial(path: str, variable_name: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), variable_name)

    def scan(self) -> list[RecordIndex]:
        if self.records:
            return list(self.records.values())

        for session_dir in sorted(self.raw_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_number = int(session_dir.name)
            for raw_file in sorted(session_dir.glob("*.mat")):
                feature_file = self.feature_dir / session_dir.name / raw_file.name
                raw_shapes = _mat_shapes(raw_file)
                feature_shapes = _mat_shapes(feature_file)
                subject = raw_file.stem.split("_")[0]
                raw_var_names = {
                    event: next(name for name in raw_shapes if name.endswith(f"_eeg{event}"))
                    for event in range(1, 25)
                }
                for event in range(1, 25):
                    label_code = SEED_IV_SESSION_LABELS[session_number][event - 1]
                    record_id = _make_record_id(self.dataset_name, subject, session_dir.name, event)
                    self.records[record_id] = RecordIndex(
                        record_id=record_id,
                        dataset=self.dataset_name,
                        subject=subject,
                        session=session_dir.name,
                        event=event,
                        raw_shape=raw_shapes[raw_var_names[event]],
                        processed_shape=feature_shapes[f"de_LDS{event}"],
                        raw_fs=200.0,
                        processed_fs=0.25,
                        channel_names=tuple(self.channel_names),
                        label_summary=SEED_IV_LABEL_MAP[label_code],
                        source_paths={
                            "raw": str(raw_file),
                            "raw_var": raw_var_names[event],
                            "processed": str(feature_file),
                        },
                        processed_options=SEED_FEATURE_KEYS,
                        default_processed_key="de_LDS",
                        processed_kind="feature",
                    )
        return list(self.records.values())

    def get_band_options(self, record_id: str, processed_key: str) -> list[str]:
        return ["__all__", *FIVE_BANDS]

    def get_label(self, record_id: str) -> LabelInfo:
        record = self.get_record(record_id)
        label_code = SEED_IV_SESSION_LABELS[int(record.session)][record.event - 1]
        return LabelInfo(
            kind="discrete",
            summary=SEED_IV_LABEL_MAP[label_code],
            discrete_label=SEED_IV_LABEL_MAP[label_code],
            discrete_code=label_code,
        )

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        variable_name = record.source_paths["raw_var"]
        data = np.asarray(self._load_raw_trial(record.source_paths["raw"], variable_name))
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return SignalBlock(
            data=data,
            fs=200.0,
            duration_sec=data.shape[1] / 200.0,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED-IV raw trial EEG.",
        )

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> FeatureBlock:
        record = self.get_record(record_id)
        variable_name = f"{processed_spec.key}{record.event}"
        data = np.asarray(self._load_feature_trial(record.source_paths["processed"], variable_name))
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return FeatureBlock(
            data=data,
            window_sec=4.0,
            band_names=list(FIVE_BANDS),
            feature_name=processed_spec.key,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED-IV smoothed EEG features.",
        )


class SEEDVIGAdapter(DatasetAdapter):
    dataset_name = "SEED-VIG"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.raw_dir = root / "Raw_Data"
        self.features_5_dir = root / "EEG_Feature_5Bands"
        self.features_2_dir = root / "EEG_Feature_2Hz"
        self.label_dir = root / "perclos_labels"
        self.channel_names, self.raw_fs = self._vig_channel_metadata(self.raw_dir / "1_20151124_noon_2.mat")

    @staticmethod
    @lru_cache(maxsize=1)
    def _vig_channel_metadata(path: Path) -> tuple[list[str], float]:
        eeg_struct = loadmat(path, variable_names=["EEG"])["EEG"][0, 0]
        names = [str(item[0]) for item in eeg_struct["chn"][0]]
        fs = float(eeg_struct["sample_rate"][0, 0])
        return names, fs

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_raw_file(path: str) -> tuple[np.ndarray, list[str], float]:
        eeg_struct = loadmat(path, variable_names=["EEG"])["EEG"][0, 0]
        data = np.asarray(eeg_struct["data"], dtype=float)
        names = [str(item[0]) for item in eeg_struct["chn"][0]]
        fs = float(eeg_struct["sample_rate"][0, 0])
        return data, names, fs

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_perclos(path: str) -> np.ndarray:
        return loadmat(path, variable_names=["perclos"])["perclos"].reshape(-1)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_feature_file(path: str, variable_name: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), variable_name)

    def scan(self) -> list[RecordIndex]:
        if self.records:
            return list(self.records.values())

        for raw_file in sorted(self.raw_dir.glob("*.mat")):
            stem = raw_file.stem
            subject = stem.split("_")[0]
            perclos = self._load_perclos(str(self.label_dir / raw_file.name))
            for event in range(1, len(perclos) + 1):
                record_id = _make_record_id(self.dataset_name, subject, stem, event)
                self.records[record_id] = RecordIndex(
                    record_id=record_id,
                    dataset=self.dataset_name,
                    subject=subject,
                    session=stem,
                    event=event,
                    raw_shape=(len(self.channel_names), int(self.raw_fs * 8)),
                    processed_shape=(len(self.channel_names), 1, 5),
                    raw_fs=self.raw_fs,
                    processed_fs=1 / 8,
                    channel_names=tuple(self.channel_names),
                    label_summary=f"perclos={perclos[event - 1]:.3f}",
                    source_paths={
                        "raw": str(raw_file),
                        "label": str(self.label_dir / raw_file.name),
                        "5bands": str(self.features_5_dir / raw_file.name),
                        "2hz": str(self.features_2_dir / raw_file.name),
                    },
                    processed_options=SEED_VIG_FEATURE_KEYS,
                    default_processed_key="5bands:de_LDS",
                    processed_kind="feature",
                )
        return list(self.records.values())

    def get_band_options(self, record_id: str, processed_key: str) -> list[str]:
        if processed_key.startswith("2hz:"):
            return ["__all__", *BANDS_2HZ]
        return ["__all__", *FIVE_BANDS]

    def get_label(self, record_id: str) -> LabelInfo:
        record = self.get_record(record_id)
        perclos = self._load_perclos(record.source_paths["label"])
        value = float(perclos[record.event - 1])
        return LabelInfo(
            kind="continuous",
            summary=f"perclos={value:.3f}",
            continuous_values=np.asarray([value]),
            continuous_step_sec=8.0,
            metadata={"perclos": value},
        )

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        data_time_first, channel_names, fs = self._load_raw_file(record.source_paths["raw"])
        start = int((record.event - 1) * fs * 8)
        stop = int(record.event * fs * 8)
        data = data_time_first[start:stop, :].T
        data, channel_names = _apply_channel_spec(data, channel_names, channel_spec)
        return SignalBlock(
            data=data,
            fs=fs,
            duration_sec=data.shape[1] / fs,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED-VIG 8s raw EEG segment aligned to one PERCLOS label.",
        )

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> FeatureBlock:
        record = self.get_record(record_id)
        band_key, feature_key = processed_spec.key.split(":", maxsplit=1)
        path_key = "5bands" if band_key == "5bands" else "2hz"
        band_names = FIVE_BANDS if band_key == "5bands" else BANDS_2HZ
        data = np.asarray(self._load_feature_file(record.source_paths[path_key], feature_key))
        data = data[:, record.event - 1 : record.event, :]
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return FeatureBlock(
            data=data,
            window_sec=8.0,
            band_names=list(band_names),
            feature_name=processed_spec.key,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note=f"SEED-VIG {band_key} feature window.",
        )


class SEEDVIIAdapter(DatasetAdapter):
    dataset_name = "SEED-VII"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.raw_dir = root / "EEG_raw"
        self.preprocessed_dir = root / "EEG_preprocessed"
        self.label_dir = root / "continuous_labels"
        self.save_info_dir = root / "save_info"
        self.channel_names = _read_channel_order(root / "Channel Order.xlsx")
        self.discrete_labels = self._load_discrete_labels(root / "emotion_label_and_stimuli_order.xlsx")

    @staticmethod
    def _load_discrete_labels(path: Path) -> dict[int, str]:
        frame = pd.read_excel(path, header=None)
        labels: dict[int, str] = {}
        for block_start in (0, 2, 4, 6):
            index_row = frame.iloc[block_start].tolist()[1:]
            label_row = frame.iloc[block_start + 1].tolist()[1:]
            for index_value, label_value in zip(index_row, label_row):
                if pd.isna(index_value) or pd.isna(label_value):
                    continue
                labels[int(index_value)] = str(label_value).strip()
        return labels

    @staticmethod
    @lru_cache(maxsize=4)
    def _continuous_labels(path: str) -> dict[str, np.ndarray]:
        mat = loadmat(path)
        return {key: np.asarray(value).reshape(-1) for key, value in mat.items() if not key.startswith("__")}

    @staticmethod
    @lru_cache(maxsize=4)
    def _preprocessed_signal(path: str, key: str) -> np.ndarray:
        return _loadmat_variable(_as_path(path), key)

    @staticmethod
    @lru_cache(maxsize=2)
    def _open_cnt(path: str) -> mne.io.BaseRaw:
        return mne.io.read_raw_cnt(path, preload=False, verbose="ERROR", eog=["HEO", "VEO"], ecg=["ECG"])

    @staticmethod
    @lru_cache(maxsize=64)
    def _trigger_durations(trigger_csv_path: str) -> list[float]:
        with open(trigger_csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        durations: list[float] = []
        for start_row, end_row in zip(rows[0::2], rows[1::2]):
            start_time = datetime.fromisoformat(start_row[1])
            end_time = datetime.fromisoformat(end_row[1])
            durations.append((end_time - start_time).total_seconds())
        return durations

    @staticmethod
    @lru_cache(maxsize=64)
    def _annotation_segments(raw_path: str) -> list[tuple[int, int]]:
        raw = SEEDVIIAdapter._open_cnt(raw_path)
        onsets = raw.annotations.onset
        fs = float(raw.info["sfreq"])
        segments: list[tuple[int, int]] = []
        for start_onset, end_onset in zip(onsets[0::2], onsets[1::2]):
            segments.append((int(round(start_onset * fs)), int(round(end_onset * fs))))
        return segments

    @staticmethod
    @lru_cache(maxsize=64)
    def _csv_segments(raw_path: str, save_info_dir: str) -> list[tuple[int, int]]:
        raw_name = Path(raw_path).stem
        csv_path = Path(save_info_dir) / f"{raw_name}_trigger_info.csv"
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        file_name = f"{raw_name}.cnt"
        fs = 1000
        if file_name in SEED_VII_SPECIAL_START_TIMES:
            recording_start = datetime.strptime(SEED_VII_SPECIAL_START_TIMES[file_name], "%H:%M:%S")
            samples = []
            for _, timestamp in rows:
                actual_time = datetime.strptime(timestamp.split(" ")[-1], "%H:%M:%S.%f")
                samples.append(int(round((actual_time.timestamp() - recording_start.timestamp()) * fs)))
            return [(samples[index], samples[index + 1]) for index in range(0, len(samples), 2)]

        annotation_segments = SEEDVIIAdapter._annotation_segments(raw_path)
        anchor_sample = annotation_segments[0][0]
        anchor_time = datetime.fromisoformat(rows[0][1])
        samples = []
        for _, timestamp in rows:
            event_time = datetime.fromisoformat(timestamp)
            delta_sec = (event_time - anchor_time).total_seconds()
            samples.append(anchor_sample + int(round(delta_sec * fs)))
        return [(samples[index], samples[index + 1]) for index in range(0, len(samples), 2)]

    def scan(self) -> list[RecordIndex]:
        if self.records:
            return list(self.records.values())

        raw_by_subject_session: dict[tuple[str, str], Path] = {}
        duration_by_subject_session: dict[tuple[str, str], list[float]] = {}
        for raw_file in sorted(self.raw_dir.glob("*.cnt")):
            subject, _, session = raw_file.stem.split("_")
            raw_by_subject_session[(subject, session)] = raw_file
            trigger_csv = self.save_info_dir / f"{raw_file.stem}_trigger_info.csv"
            duration_by_subject_session[(subject, session)] = self._trigger_durations(str(trigger_csv))

        for pre_file in sorted(self.preprocessed_dir.glob("*.mat")):
            subject = pre_file.stem
            pre_shapes = _mat_shapes(pre_file)
            continuous_labels = self._continuous_labels(str(self.label_dir / pre_file.name))
            for event in range(1, 81):
                session = str((event - 1) // 20 + 1)
                within_session = (event - 1) % 20
                duration_sec = duration_by_subject_session[(subject, session)][within_session]
                continuous = continuous_labels[str(event)]
                discrete = self.discrete_labels.get(event, "unknown")
                record_id = _make_record_id(self.dataset_name, subject, session, event)
                self.records[record_id] = RecordIndex(
                    record_id=record_id,
                    dataset=self.dataset_name,
                    subject=subject,
                    session=session,
                    event=event,
                    raw_shape=(len(self.channel_names), int(round(duration_sec * 1000))),
                    processed_shape=pre_shapes[str(event)],
                    raw_fs=1000.0,
                    processed_fs=200.0,
                    channel_names=tuple(self.channel_names),
                    label_summary=f"{discrete} | cont_mean={float(np.mean(continuous)):.1f}",
                    source_paths={
                        "raw": str(raw_by_subject_session[(subject, session)]),
                        "processed": str(pre_file),
                        "continuous": str(self.label_dir / pre_file.name),
                    },
                    processed_options=(SEED_VII_PROCESSED_KEY,),
                    default_processed_key=SEED_VII_PROCESSED_KEY,
                    processed_kind="signal",
                )
        return list(self.records.values())

    def get_label(self, record_id: str) -> LabelInfo:
        record = self.get_record(record_id)
        continuous = self._continuous_labels(record.source_paths["continuous"])[str(record.event)]
        discrete = self.discrete_labels.get(record.event, "unknown")
        step_sec = (record.processed_shape[1] / 200.0) / max(1, len(continuous))
        return LabelInfo(
            kind="continuous",
            summary=f"{discrete} | cont_mean={float(np.mean(continuous)):.1f}",
            discrete_label=discrete,
            continuous_values=continuous.astype(float),
            continuous_step_sec=step_sec,
            metadata={"continuous_mean": float(np.mean(continuous))},
        )

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        raw = self._open_cnt(record.source_paths["raw"])
        within_session = (record.event - 1) % 20
        start, stop = self._csv_segments(record.source_paths["raw"], str(self.save_info_dir))[within_session]
        picks = [name for name in raw.ch_names if name not in {"M1", "M2", "ECG", "HEO", "VEO"}]
        data = raw.get_data(picks=picks, start=start, stop=stop, units="uV")
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return SignalBlock(
            data=data,
            fs=1000.0,
            duration_sec=data.shape[1] / 1000.0,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED-VII raw EEG segment cut from CNT using trigger pairs.",
        )

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> SignalBlock:
        record = self.get_record(record_id)
        data = np.asarray(self._preprocessed_signal(record.source_paths["processed"], str(record.event)))
        data, channel_names = _apply_channel_spec(data, list(record.channel_names), channel_spec)
        return SignalBlock(
            data=data,
            fs=200.0,
            duration_sec=data.shape[1] / 200.0,
            channel_names=channel_names,
            label_info=self.get_label(record_id),
            source_note="SEED-VII preprocessed EEG segment.",
        )


class DatasetRegistry:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        self.adapters: dict[str, DatasetAdapter] = {
            "DEAP": DEAPAdapter(root / "DEAP"),
            "SEED": SEEDAdapter(root / "SEED"),
            "SEED-IV": SEEDIVAdapter(root / "SEED_IV"),
            "SEED-VIG": SEEDVIGAdapter(root / "SEED-VIG"),
            "SEED-VII": SEEDVIIAdapter(root / "SEED-VII"),
        }
        self.record_to_dataset: dict[str, str] = {}
        self.records: list[RecordIndex] = self.scan_all()

    def scan_all(self) -> list[RecordIndex]:
        all_records: list[RecordIndex] = []
        self.record_to_dataset.clear()
        for dataset_name, adapter in self.adapters.items():
            records = adapter.scan()
            for record in records:
                self.record_to_dataset[record.record_id] = dataset_name
            all_records.extend(records)
        return all_records

    def get_adapter(self, record_id: str) -> DatasetAdapter:
        return self.adapters[self.record_to_dataset[record_id]]

    def get_record(self, record_id: str) -> RecordIndex:
        return self.get_adapter(record_id).get_record(record_id)

    def get_label(self, record_id: str) -> LabelInfo:
        return self.get_adapter(record_id).get_label(record_id)

    def load_raw(self, record_id: str, channel_spec: ChannelSpec) -> SignalBlock:
        return self.get_adapter(record_id).load_raw(record_id, channel_spec)

    def load_processed(self, record_id: str, processed_spec: ProcessedSpec, channel_spec: ChannelSpec) -> SignalBlock | FeatureBlock:
        return self.get_adapter(record_id).load_processed(record_id, processed_spec, channel_spec)

    def get_band_options(self, record_id: str, processed_key: str) -> list[str]:
        return self.get_adapter(record_id).get_band_options(record_id, processed_key)

    def get_available_channel_names(self, record_id: str, include_aux: bool = False) -> list[str]:
        return self.get_adapter(record_id).get_available_channel_names(record_id, include_aux)

    def filter_records(
        self,
        dataset: str | None = None,
        subject: str | None = None,
        session: str | None = None,
    ) -> list[RecordIndex]:
        records = self.records
        if dataset:
            records = [record for record in records if record.dataset == dataset]
        if subject:
            records = [record for record in records if record.subject == subject]
        if session:
            records = [record for record in records if record.session == session]
        return records

    def dataset_names(self) -> list[str]:
        return list(self.adapters.keys())

    def subject_names(self, dataset: str) -> list[str]:
        return sorted({record.subject for record in self.records if record.dataset == dataset}, key=lambda value: (len(value), value))

    def session_names(self, dataset: str, subject: str | None = None) -> list[str]:
        return sorted(
            {
                record.session
                for record in self.records
                if record.dataset == dataset and (subject is None or record.subject == subject)
            },
            key=lambda value: (len(value), value),
        )
