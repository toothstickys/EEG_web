"""ICA preprocessing, fitting, ICLabel classification, thumbnail generation, and caching."""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA

from models import (
    ICAComponentInfo,
    ICAEventWindow,
    ICAResultBlock,
    ICASourceBlock,
    LabelInfo,
    SignalBlock,
)

ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_ica_cache: dict[str, tuple[ICAResultBlock, ICA, Any]] = {}
_CACHE_MAX = 4


def _cache_key(record_id: str, scope: str) -> str:
    return f"{record_id}::{scope}"


def get_cached_result(record_id: str, scope: str) -> tuple[ICAResultBlock, ICA, Any] | None:
    return _ica_cache.get(_cache_key(record_id, scope))


def _put_cache(record_id: str, scope: str, result: ICAResultBlock, ica: ICA, inst: Any) -> None:
    key = _cache_key(record_id, scope)
    if len(_ica_cache) >= _CACHE_MAX and key not in _ica_cache:
        oldest = next(iter(_ica_cache))
        del _ica_cache[oldest]
    _ica_cache[key] = (result, ica, inst)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_ica_pipeline(
    source: ICASourceBlock,
    record_id: str,
    n_components: int | None = None,
    reject_threshold: float = 0.80,
    current_event_window: ICAEventWindow | None = None,
) -> ICAResultBlock:
    """Run ICA + ICLabel on an ICASourceBlock.

    Steps (following official mne-icalabel example):
    1. Copy inst, pick EEG only
    2. Common average reference
    3. Bandpass 1 Hz – min(100, Nyquist-1) Hz
    4. Resample to 200 Hz if fs > 200
    5. Fit ICA (extended infomax)
    6. ICLabel classification
    7. Build result with source/cleaned preview
    """
    cached = get_cached_result(record_id, source.scope)
    if cached is not None:
        return cached[0]

    inst = source.inst.copy()
    notes: list[str] = list(source.warnings)

    # Ensure data is loaded
    if hasattr(inst, "load_data"):
        inst.load_data(verbose="ERROR")

    n_eeg = len(mne.pick_types(inst.info, eeg=True))
    fs = float(inst.info["sfreq"])
    duration = _inst_duration(inst)

    # Low data / channel warnings
    if duration < 30.0:
        notes.append(f"⚠ Short data ({duration:.1f}s < 30s): ICA results may have low confidence.")
    if n_eeg < 8:
        notes.append(f"⚠ Few EEG channels ({n_eeg} < 8): ICA results may have low confidence.")

    # --- Preprocessing for ICA fitting (on a copy) ---
    filt_inst = inst.copy()

    # 1. Common average reference
    if isinstance(filt_inst, mne.io.BaseRaw):
        filt_inst.set_eeg_reference("average", verbose="ERROR")
    else:
        filt_inst.set_eeg_reference("average", verbose="ERROR")

    # 2. Bandpass filter
    nyquist = fs / 2.0
    h_freq = min(100.0, nyquist - 1.0)
    if h_freq <= 1.0:
        h_freq = nyquist - 0.5
    filt_inst.filter(l_freq=1.0, h_freq=h_freq, verbose="ERROR")

    # 3. Resample if fs > 200
    if fs > 200.0:
        filt_inst.resample(200.0, verbose="ERROR")
        notes.append(f"Resampled from {fs:.0f} Hz to 200 Hz for ICA fitting.")

    # 4. Determine n_components
    n_eeg_filt = len(mne.pick_types(filt_inst.info, eeg=True))
    if n_components is None:
        n_components = min(n_eeg_filt - 1, 15)
    n_components = max(2, min(n_components, n_eeg_filt - 1))

    # 5. Fit ICA
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        max_iter="auto",
    )
    fit_kwargs: dict[str, Any] = {}
    if source.scope == "session":
        fit_kwargs["decim"] = 2
    ica.fit(filt_inst, verbose="ERROR", **fit_kwargs)

    # 6. ICLabel classification
    from mne_icalabel import label_components

    ic_labels = label_components(filt_inst, ica, method="iclabel")
    labels_list: list[str] = ic_labels["labels"]
    y_pred_proba: np.ndarray = ic_labels["y_pred_proba"]

    # 7. Build component info
    components: list[ICAComponentInfo] = []
    excluded: list[int] = []
    for idx, label in enumerate(labels_list):
        probs = {cls: float(y_pred_proba[idx, ci]) for ci, cls in enumerate(ICLABEL_CLASSES)}
        max_prob = max(probs.values())
        suggested = label not in {"brain", "other"} and max_prob >= reject_threshold
        if suggested:
            excluded.append(idx)
        components.append(ICAComponentInfo(
            index=idx,
            label=label,
            probabilities=probs,
            suggested_reject=suggested,
        ))

    # 8. Generate preview windows
    preview_start, preview_stop = _pick_preview_window(source, current_event_window)

    # Source preview (from original unfiltered inst)
    source_preview = _extract_preview(inst, preview_start, preview_stop, "Source EEG (before ICA)")

    # Cleaned preview (apply ICA exclusion to original unfiltered inst)
    reconst = inst.copy()
    ica.apply(reconst, exclude=excluded, verbose="ERROR")
    cleaned_preview = _extract_preview(reconst, preview_start, preview_stop, "Cleaned EEG (after ICA)")

    fit_fs = float(filt_inst.info["sfreq"])

    result = ICAResultBlock(
        scope=source.scope,
        fit_fs=fit_fs,
        components=components,
        excluded_indices=excluded,
        source_preview=source_preview,
        cleaned_preview=cleaned_preview,
        notes=notes,
    )

    _put_cache(record_id, source.scope, result, ica, filt_inst)
    return result


def recompute_exclusions(
    record_id: str,
    scope: str,
    exclude_indices: list[int],
    current_event_window: ICAEventWindow | None = None,
    source: ICASourceBlock | None = None,
) -> ICAResultBlock | None:
    """Re-apply ICA with different excluded components (no re-fitting)."""
    cached = get_cached_result(record_id, scope)
    if cached is None:
        return None
    old_result, ica, filt_inst = cached

    # Need the original unfiltered inst — retrieve from source
    if source is None:
        return None
    inst = source.inst.copy()
    if hasattr(inst, "load_data"):
        inst.load_data(verbose="ERROR")

    preview_start, preview_stop = _pick_preview_window(source, current_event_window)
    source_preview = _extract_preview(inst, preview_start, preview_stop, "Source EEG (before ICA)")

    reconst = inst.copy()
    ica.apply(reconst, exclude=exclude_indices, verbose="ERROR")
    cleaned_preview = _extract_preview(reconst, preview_start, preview_stop, "Cleaned EEG (after ICA)")

    # Rebuild components with updated suggested_reject
    components = []
    for comp in old_result.components:
        components.append(ICAComponentInfo(
            index=comp.index,
            label=comp.label,
            probabilities=comp.probabilities,
            suggested_reject=comp.index in exclude_indices,
        ))

    new_result = ICAResultBlock(
        scope=old_result.scope,
        fit_fs=old_result.fit_fs,
        components=components,
        excluded_indices=list(exclude_indices),
        source_preview=source_preview,
        cleaned_preview=cleaned_preview,
        notes=old_result.notes,
    )
    _put_cache(record_id, scope, new_result, ica, filt_inst)
    return new_result


# ---------------------------------------------------------------------------
# Thumbnail / image generation
# ---------------------------------------------------------------------------

def generate_topomap_base64(record_id: str, scope: str, component_idx: int) -> str | None:
    """Generate a base64-encoded PNG topomap for a single IC."""
    cached = get_cached_result(record_id, scope)
    if cached is None:
        return None
    _, ica, inst = cached
    try:
        fig = ica.plot_components(picks=[component_idx], show=False, verbose="ERROR")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def generate_component_psd_base64(record_id: str, scope: str, component_idx: int) -> str | None:
    """Generate a base64-encoded PNG PSD plot for a single IC."""
    cached = get_cached_result(record_id, scope)
    if cached is None:
        return None
    _, ica, inst = cached
    try:
        sources = ica.get_sources(inst)
        if isinstance(sources, mne.io.BaseRaw):
            data = sources.get_data(picks=[component_idx])[0]
            fs = sources.info["sfreq"]
        else:
            data = sources.get_data()[:, component_idx, :].ravel()
            fs = sources.info["sfreq"]

        from scipy.signal import welch
        freqs, pxx = welch(data, fs=fs, nperseg=min(len(data), int(fs * 2)))

        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.semilogy(freqs, pxx, color="#d44b2f", linewidth=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"IC{component_idx:03d} PSD")
        ax.set_xlim(0, min(100, fs / 2))
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def generate_activation_data(record_id: str, scope: str, component_idx: int, max_seconds: float = 10.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Get activation time series for a single IC. Returns (times, activation) arrays."""
    cached = get_cached_result(record_id, scope)
    if cached is None:
        return None
    _, ica, inst = cached
    try:
        sources = ica.get_sources(inst)
        if isinstance(sources, mne.io.BaseRaw):
            fs = sources.info["sfreq"]
            max_samples = int(max_seconds * fs)
            data = sources.get_data(picks=[component_idx])[:, :max_samples][0]
            times = np.arange(len(data)) / fs
        else:
            fs = sources.info["sfreq"]
            data = sources.get_data()[:, component_idx, :]
            # Use first epoch
            first_epoch = data[0]
            max_samples = int(max_seconds * fs)
            first_epoch = first_epoch[:max_samples]
            times = np.arange(len(first_epoch)) / fs
            data = first_epoch
        return times, data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inst_duration(inst: mne.io.BaseRaw | mne.BaseEpochs) -> float:
    if isinstance(inst, mne.io.BaseRaw):
        return inst.times[-1] if len(inst.times) > 0 else 0.0
    else:
        n_epochs = len(inst)
        epoch_len = inst.times[-1] - inst.times[0] if len(inst.times) > 0 else 0.0
        return n_epochs * epoch_len


def _pick_preview_window(
    source: ICASourceBlock,
    current_event_window: ICAEventWindow | None,
) -> tuple[float, float]:
    """Select the time window for preview plots."""
    duration = _inst_duration(source.inst)
    if source.scope == "event":
        return 0.0, duration
    if current_event_window is not None:
        return current_event_window.start_sec, current_event_window.stop_sec
    return 0.0, min(30.0, duration)


def _extract_preview(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    start_sec: float,
    stop_sec: float,
    note: str,
) -> SignalBlock:
    """Extract a time window from an MNE inst and return as SignalBlock."""
    if isinstance(inst, mne.io.BaseRaw):
        fs = float(inst.info["sfreq"])
        start_samp = max(0, int(start_sec * fs))
        stop_samp = min(inst.n_times, int(stop_sec * fs))
        data = inst.get_data(start=start_samp, stop=stop_samp) * 1e6  # V -> uV
        ch_names = list(inst.ch_names)
    else:
        fs = float(inst.info["sfreq"])
        all_data = inst.get_data()  # (n_epochs, n_channels, n_times)
        # Concatenate epochs for preview
        concat = np.concatenate(all_data, axis=1)
        total_dur = concat.shape[1] / fs
        start_samp = max(0, int(start_sec * fs))
        stop_samp = min(concat.shape[1], int(stop_sec * fs))
        if start_samp >= concat.shape[1]:
            start_samp = 0
            stop_samp = min(concat.shape[1], int(30.0 * fs))
        data = concat[:, start_samp:stop_samp] * 1e6
        ch_names = list(inst.ch_names)

    return SignalBlock(
        data=data,
        fs=fs,
        duration_sec=data.shape[1] / fs,
        channel_names=ch_names,
        label_info=LabelInfo(kind="none", summary=note),
        start_sec=start_sec,
        units="uV",
        source_note=note,
    )
