from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
from plotly.subplots import make_subplots

from dataset_adapters import DATASET_ROOT, DatasetRegistry
from models import ChannelSpec, FeatureBlock, ProcessedSpec, SignalBlock


APP_TITLE = "EEG Raw / Processed Comparator"
REGISTRY = DatasetRegistry(DATASET_ROOT)
DEFAULT_DATASET = REGISTRY.dataset_names()[0]
ACCENT = "#d44b2f"
PANEL = "#fff8f1"
BORDER = "#e2c9b5"
INK = "#2e2220"
MUTED = "#7b655d"
ONLINE_PROCESSING_KEY = "online_processing"
ONLINE_BAND_SPECS = {
    "__raw__": {"label": "No Band / Raw", "short": "raw", "low": None, "high": None},
    "broad_1_50": {"label": "Broad 1-50Hz (Butterworth)", "short": "1-50", "low": 1.0, "high": 50.0},
    "delta_1_4": {"label": "Delta 1-4Hz (Butterworth)", "short": "delta", "low": 1.0, "high": 4.0},
    "theta_4_8": {"label": "Theta 4-8Hz (Butterworth)", "short": "theta", "low": 4.0, "high": 8.0},
    "alpha_8_13": {"label": "Alpha 8-13Hz (Butterworth)", "short": "alpha", "low": 8.0, "high": 13.0},
    "mu_8_12": {"label": "Mu 8-12Hz (Butterworth)", "short": "mu", "low": 8.0, "high": 12.0},
    "beta_13_30": {"label": "Beta 13-30Hz (Butterworth)", "short": "beta", "low": 13.0, "high": 30.0},
    "gamma_30_45": {"label": "Gamma 30-45Hz (Butterworth)", "short": "gamma", "low": 30.0, "high": 45.0},
}
ONLINE_FILTER_SPECS = {
    "lds": {"label": "LDS Smoother"},
    "butter_highpass_1": {"label": "Butterworth Highpass 1Hz"},
    "butter_lowpass_40": {"label": "Butterworth Lowpass 40Hz"},
    "butter_lowpass_30": {"label": "Butterworth Lowpass 30Hz"},
    "notch_50": {"label": "Notch 50Hz"},
    "notch_60": {"label": "Notch 60Hz"},
    "moving_average_5": {"label": "Moving Average (5)"},
    "psd": {"label": "PSD (1s windows)"},
    "de": {"label": "DE (1s windows)"},
    "median_5": {"label": "Median (5)"},
    "savgol_11": {"label": "Savitzky-Golay (11)"},
}
ONLINE_PRESET_SPECS = {
    "custom": {
        "label": "Custom",
        "bands": None,
        "filters": None,
    },
    "de_moving_average": {
        "label": "de_movingAve",
        "bands": ["delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30", "gamma_30_45"],
        "filters": ["de", "moving_average_5"],
    },
    "de_lds": {
        "label": "de_LDS",
        "bands": ["delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30", "gamma_30_45"],
        "filters": ["de", "lds"],
    },
    "psd_moving_average": {
        "label": "psd_movingAve",
        "bands": ["delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30", "gamma_30_45"],
        "filters": ["psd", "moving_average_5"],
    },
    "psd_lds": {
        "label": "psd_LDS",
        "bands": ["delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30", "gamma_30_45"],
        "filters": ["psd", "lds"],
    },
}
ONLINE_FEATURE_KEYS = {"psd", "de"}
ONLINE_SIGNAL_ONLY_KEYS = {"butter_highpass_1", "butter_lowpass_40", "butter_lowpass_30", "notch_50", "notch_60", "median_5", "savgol_11"}
APP_CSS = """
body {
    margin: 0;
    background:
        radial-gradient(circle at top left, #fff4ea 0, #fff4ea 20%, transparent 60%),
        linear-gradient(180deg, #fffaf6 0%, #fff4ed 100%);
    color: #2e2220;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}
.hero { padding: 28px 28px 18px 28px; }
.title { font-size: 30px; font-weight: 800; letter-spacing: 0.02em; }
.subtitle { color: #7b655d; margin-top: 8px; font-size: 14px; }
.top-grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 14px; padding: 0 28px 18px 28px; }
.control-block, .button-stack, .left-column, .right-column, .preview-panel {
    background: #fff8f1;
    border: 1px solid #e2c9b5;
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(131, 91, 61, 0.08);
}
.control-block { padding: 14px; grid-column: span 2; }
.control-block.wide { grid-column: span 3; }
.button-stack {
    grid-column: span 2;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    justify-content: center;
}
.mini-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7b655d;
    margin-bottom: 8px;
    font-weight: 700;
}
.hint-text {
    color: #7b655d;
    font-size: 12px;
    margin-top: 8px;
    line-height: 1.4;
}
.action-btn {
    border: none;
    border-radius: 12px;
    background: #d44b2f;
    color: white;
    font-weight: 700;
    padding: 12px 14px;
    cursor: pointer;
}
.action-btn.secondary { background: #6b5a54; }
.summary-bar {
    margin: 0 28px 18px 28px;
    padding: 12px 16px;
    border-radius: 14px;
    background: #2e2220;
    color: white;
    font-size: 14px;
    font-weight: 600;
}
.content-grid { display: grid; grid-template-columns: 1.1fr 1.9fr; gap: 18px; padding: 0 28px 18px 28px; }
.left-column, .right-column { padding: 16px; }
.section-title { font-size: 15px; font-weight: 800; margin-bottom: 12px; letter-spacing: 0.03em; }
.bottom-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; padding: 0 28px 28px 28px; }
.preview-panel { padding: 16px; }
.preview {
    margin: 0;
    padding: 14px;
    background: #fffefc;
    border: 1px solid #f0ddd0;
    border-radius: 12px;
    overflow: auto;
    white-space: pre-wrap;
    min-height: 190px;
    font-size: 12px;
    line-height: 1.45;
}
.meta-text, .notes-text { color: #7b655d; font-size: 13px; margin-bottom: 10px; line-height: 1.5; }
@media (max-width: 1200px) {
    .top-grid { grid-template-columns: repeat(6, minmax(0, 1fr)); }
    .content-grid, .bottom-grid { grid-template-columns: 1fr; }
    .control-block, .button-stack, .control-block.wide { grid-column: span 3; }
}
@media (max-width: 700px) {
    .top-grid { grid-template-columns: 1fr; }
    .control-block, .button-stack, .control-block.wide { grid-column: span 1; }
}
"""

app = Dash(__name__, title=APP_TITLE)
server = app.server
app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{APP_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""


def format_processed_label(key: str) -> str:
    if key == ONLINE_PROCESSING_KEY:
        return "线上处理"
    if key == "processed_eeg":
        return "Processed EEG"
    if key == "eeg_preprocessed":
        return "Preprocessed EEG"
    if ":" in key:
        group, feature = key.split(":", maxsplit=1)
        group_label = "5Bands" if group == "5bands" else "2Hz"
        return f"{group_label} · {feature}"
    return key


def build_dropdown_options(values: list[str], prefix: str = "") -> list[dict[str, str]]:
    return [{"label": f"{prefix}{value}" if prefix else value, "value": value} for value in values]


def safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    try:
        return signal.sosfiltfilt(sos, data, axis=1)
    except ValueError:
        return signal.sosfilt(sos, data, axis=1)


def safe_filtfilt(b: np.ndarray, a: np.ndarray, data: np.ndarray) -> np.ndarray:
    try:
        return signal.filtfilt(b, a, data, axis=1)
    except ValueError:
        return signal.lfilter(b, a, data, axis=1)


def butter_bandpass(data: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.99)
    if high <= low:
        return data.copy()
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return safe_sosfiltfilt(sos, data)


def butter_highpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    cutoff = min(max(cutoff_hz / nyquist, 1e-5), 0.99)
    sos = signal.butter(order, cutoff, btype="highpass", output="sos")
    return safe_sosfiltfilt(sos, data)


def butter_lowpass(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    cutoff = min(max(cutoff_hz / nyquist, 1e-5), 0.99)
    sos = signal.butter(order, cutoff, btype="lowpass", output="sos")
    return safe_sosfiltfilt(sos, data)


def notch_filter(data: np.ndarray, fs: float, freq_hz: float, quality: float = 30.0) -> np.ndarray:
    if fs <= freq_hz * 2:
        return data.copy()
    b, a = signal.iirnotch(freq_hz, quality, fs)
    return safe_filtfilt(b, a, data)


def moving_average_filter(data: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.vstack([np.convolve(channel, kernel, mode="same") for channel in data])


def moving_average_feature(data: np.ndarray, window: int = 5) -> np.ndarray:
    channels, windows, bands = data.shape
    reshaped = np.transpose(data, (0, 2, 1)).reshape(channels * bands, windows)
    filtered = moving_average_filter(reshaped, window=window)
    return np.transpose(filtered.reshape(channels, bands, windows), (0, 2, 1))


def median_filter(data: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = window if window % 2 == 1 else window + 1
    return signal.medfilt(data, kernel_size=(1, kernel))


def savgol_smooth(data: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    if data.shape[1] < 5:
        return data.copy()
    window = min(window, data.shape[1] if data.shape[1] % 2 == 1 else data.shape[1] - 1)
    window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if window % 2 == 0:
        window += 1
    if window >= data.shape[1]:
        window = data.shape[1] - 1 if data.shape[1] % 2 == 0 else data.shape[1]
    if window < polyorder + 2:
        return data.copy()
    return signal.savgol_filter(data, window_length=window, polyorder=polyorder, axis=1, mode="interp")


def lds_smooth(data: np.ndarray, process_noise: float = 1e-4, measurement_noise: float = 0.1) -> np.ndarray:
    output = np.empty_like(data, dtype=float)
    for index, series in enumerate(data):
        forward = np.empty_like(series, dtype=float)
        estimate = float(series[0])
        variance = 1.0
        forward[0] = estimate
        for sample_index in range(1, series.shape[0]):
            variance += process_noise
            gain = variance / (variance + measurement_noise)
            estimate = estimate + gain * (float(series[sample_index]) - estimate)
            variance = (1.0 - gain) * variance
            forward[sample_index] = estimate

        backward = np.empty_like(series, dtype=float)
        estimate = float(forward[-1])
        variance = 1.0
        backward[-1] = estimate
        for sample_index in range(series.shape[0] - 2, -1, -1):
            variance += process_noise
            gain = variance / (variance + measurement_noise)
            estimate = estimate + gain * (float(forward[sample_index]) - estimate)
            variance = (1.0 - gain) * variance
            backward[sample_index] = estimate
        output[index] = backward
    return output


def lds_feature(data: np.ndarray) -> np.ndarray:
    channels, windows, bands = data.shape
    reshaped = np.transpose(data, (0, 2, 1)).reshape(channels * bands, windows)
    filtered = lds_smooth(reshaped)
    return np.transpose(filtered.reshape(channels, bands, windows), (0, 2, 1))


def extract_window_feature(
    data: np.ndarray,
    fs: float,
    feature_key: str,
    window_sec: float = 1.0,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, float]:
    window_samples = max(1, int(round(fs * window_sec)))
    step_samples = max(1, int(round(fs * step_sec)))
    total_samples = data.shape[1]
    if total_samples <= window_samples:
        windows = data[:, None, :]
    else:
        starts = list(range(0, total_samples - window_samples + 1, step_samples))
        if starts[-1] != total_samples - window_samples:
            starts.append(total_samples - window_samples)
        windows = np.stack([data[:, start : start + window_samples] for start in starts], axis=1)

    if feature_key == "psd":
        feature = np.mean(np.square(windows), axis=2) / max(fs, 1e-12)
        feature = np.log10(np.maximum(feature, 1e-12))
    else:
        variance = np.var(windows, axis=2)
        feature = 0.5 * np.log(2 * np.pi * np.e * np.maximum(variance, 1e-12))
    return feature, window_sec


def apply_online_filter_step(data: np.ndarray, fs: float, filter_key: str) -> np.ndarray:
    if filter_key == "lds":
        return lds_smooth(data)
    if filter_key == "butter_highpass_1":
        return butter_highpass(data, fs, 1.0)
    if filter_key == "butter_lowpass_40":
        return butter_lowpass(data, fs, 40.0)
    if filter_key == "butter_lowpass_30":
        return butter_lowpass(data, fs, 30.0)
    if filter_key == "notch_50":
        return notch_filter(data, fs, 50.0)
    if filter_key == "notch_60":
        return notch_filter(data, fs, 60.0)
    if filter_key == "moving_average_5":
        return moving_average_filter(data, window=5)
    if filter_key == "median_5":
        return median_filter(data, window=5)
    if filter_key == "savgol_11":
        return savgol_smooth(data, window=11, polyorder=3)
    return data


def apply_online_feature_step(data: np.ndarray, filter_key: str) -> np.ndarray:
    if filter_key == "lds":
        return lds_feature(data)
    if filter_key == "moving_average_5":
        return moving_average_feature(data, window=5)
    return data


def build_online_processed_block(raw_block: SignalBlock, band_keys: list[str] | None, filter_keys: list[str] | None) -> SignalBlock | FeatureBlock:
    ordered_bands = [key for key in (band_keys or []) if key in ONLINE_BAND_SPECS]
    ordered_filters = [key for key in (filter_keys or []) if key in ONLINE_FILTER_SPECS]
    if not ordered_bands:
        ordered_bands = ["__raw__"]

    base_data = np.asarray(raw_block.data, dtype=float)
    signal_blocks: list[np.ndarray] = []
    signal_names: list[str] = []
    feature_blocks: list[np.ndarray] = []
    feature_band_names: list[str] = []
    pipeline_lines: list[str] = []
    feature_name: str | None = None
    feature_window_sec: float | None = None

    for band_key in ordered_bands:
        band_spec = ONLINE_BAND_SPECS[band_key]
        current_signal = base_data.copy()
        current_feature: np.ndarray | None = None
        if band_spec["low"] is not None and band_spec["high"] is not None:
            current_signal = butter_bandpass(current_signal, raw_block.fs, float(band_spec["low"]), float(band_spec["high"]))
        pipeline = [band_spec["label"]]
        for filter_key in ordered_filters:
            if current_feature is None and filter_key in ONLINE_FEATURE_KEYS:
                current_feature_2d, current_window_sec = extract_window_feature(current_signal, raw_block.fs, filter_key)
                current_feature = current_feature_2d[:, :, None]
                feature_name = ONLINE_FILTER_SPECS[filter_key]["label"] if feature_name is None else feature_name
                feature_window_sec = current_window_sec if feature_window_sec is None else feature_window_sec
            elif current_feature is None:
                current_signal = apply_online_filter_step(current_signal, raw_block.fs, filter_key)
            elif filter_key in ONLINE_FEATURE_KEYS:
                pipeline.append(f"skip:{ONLINE_FILTER_SPECS[filter_key]['label']}")
                continue
            elif filter_key in ONLINE_SIGNAL_ONLY_KEYS:
                pipeline.append(f"skip:{ONLINE_FILTER_SPECS[filter_key]['label']}")
                continue
            else:
                current_feature = apply_online_feature_step(current_feature, filter_key)
            pipeline.append(ONLINE_FILTER_SPECS[filter_key]["label"])

        if current_feature is None:
            signal_blocks.append(current_signal)
            prefix_needed = len(ordered_bands) > 1 or band_key != "__raw__"
            if prefix_needed:
                signal_names.extend([f"{band_spec['short']} | {name}" for name in raw_block.channel_names])
            else:
                signal_names.extend(list(raw_block.channel_names))
        else:
            feature_blocks.append(current_feature)
            feature_band_names.append(band_spec["short"])
        pipeline_lines.append(" -> ".join(pipeline))

    source_note = "Online processing pipeline: " + " || ".join(pipeline_lines)
    if feature_blocks:
        if signal_blocks:
            source_note += " || mixed signal/feature pipeline requested; feature result shown in processed panel."
        combined_feature = np.concatenate(feature_blocks, axis=2)
        return FeatureBlock(
            data=combined_feature,
            window_sec=feature_window_sec or 1.0,
            band_names=feature_band_names,
            feature_name=feature_name or "Online Feature",
            channel_names=list(raw_block.channel_names),
            label_info=raw_block.label_info,
            source_note=source_note,
        )

    combined_signal = np.vstack(signal_blocks)
    return SignalBlock(
        data=combined_signal,
        fs=raw_block.fs,
        duration_sec=raw_block.duration_sec,
        channel_names=signal_names,
        label_info=raw_block.label_info,
        start_sec=raw_block.start_sec,
        units=raw_block.units,
        source_note=source_note,
    )


def format_online_order(band_keys: list[str] | None, filter_keys: list[str] | None) -> str:
    ordered_bands = [ONLINE_BAND_SPECS[key]["short"] for key in (band_keys or []) if key in ONLINE_BAND_SPECS]
    ordered_filters = [ONLINE_FILTER_SPECS[key]["label"] for key in (filter_keys or []) if key in ONLINE_FILTER_SPECS]
    parts = [f"bands({', '.join(ordered_bands)})" if ordered_bands else "bands(raw)"]
    parts.extend(ordered_filters)
    return " -> ".join(parts)


def format_label_badge(summary: str, discrete_label: str | None) -> html.Div:
    children = []
    if discrete_label:
        children.append(
            html.Span(
                discrete_label,
                style={
                    "display": "inline-block",
                    "padding": "4px 10px",
                    "borderRadius": "999px",
                    "background": "#ffe6d7",
                    "color": ACCENT,
                    "fontWeight": 700,
                    "marginRight": "10px",
                },
            )
        )
    children.append(html.Span(summary, style={"color": MUTED}))
    return html.Div(children, style={"display": "flex", "alignItems": "center", "gap": "6px"})


def downsample_signal(data: np.ndarray, target_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[1] <= target_points:
        indices = np.arange(data.shape[1])
        return data, indices
    step = int(np.ceil(data.shape[1] / target_points))
    indices = np.arange(0, data.shape[1], step)
    return data[:, indices], indices


def build_signal_figure(block: SignalBlock, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if block.data.size == 0:
        fig.update_layout(template="plotly_white", title=title)
        return fig

    data = block.data.astype(float)
    centered = data - np.median(data, axis=1, keepdims=True)
    view_data, indices = downsample_signal(centered)
    x = block.start_sec + (indices / block.fs)
    amplitude = float(np.nanpercentile(np.abs(view_data), 95))
    spacing = amplitude * 3.0 if amplitude > 0 else 25.0
    offsets = np.arange(view_data.shape[0] - 1, -1, -1) * spacing

    for idx, channel_name in enumerate(block.channel_names):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=view_data[idx] + offsets[idx],
                mode="lines",
                line={"width": 1.2},
                name=channel_name,
                hovertemplate=f"{channel_name}<br>t=%{{x:.3f}}s<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_yaxes(
        tickvals=list(offsets),
        ticktext=list(block.channel_names),
        secondary_y=False,
        showgrid=False,
        zeroline=False,
    )

    if block.label_info.continuous_values is not None:
        values = np.asarray(block.label_info.continuous_values, dtype=float).reshape(-1)
        if values.size:
            if values.size == 1:
                label_x = np.asarray([block.duration_sec / 2], dtype=float)
            else:
                step_sec = block.label_info.continuous_step_sec or (block.duration_sec / values.size)
                label_x = np.arange(values.size, dtype=float) * step_sec
            fig.add_trace(
                go.Scatter(
                    x=label_x,
                    y=values,
                    mode="lines+markers",
                    name="label",
                    line={"color": ACCENT, "shape": "hv", "width": 2},
                    marker={"size": 5},
                    hovertemplate="label=%{y:.3f}<br>t=%{x:.2f}s<extra></extra>",
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Label", secondary_y=True, showgrid=False, zeroline=False)
    else:
        fig.update_yaxes(secondary_y=True, visible=False)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=430,
        hovermode="x unified",
        margin={"l": 80, "r": 40, "t": 70, "b": 55},
        paper_bgcolor=PANEL,
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "color": INK},
    )
    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor="#f0e3d8")
    return fig


def build_feature_figure(block: FeatureBlock, band: str) -> go.Figure:
    band = band or "__all__"
    data = np.asarray(block.data, dtype=float)
    title = f"Processed | {block.feature_name} | {block.label_info.summary}"
    if band == "__all__":
        if data.shape[1] == 1:
            z = data[:, 0, :]
            x = block.band_names
            x_title = "Band"
            title = f"{title} | all bands"
        else:
            z = data.mean(axis=2)
            x = np.arange(data.shape[1]) * block.window_sec
            x_title = "Time (s)"
            title = f"{title} | mean across bands"
    else:
        band_index = block.band_names.index(band)
        z = data[:, :, band_index]
        x = np.arange(data.shape[1]) * block.window_sec
        x_title = "Time (s)"
        title = f"{title} | {band}"

    if z.ndim == 1:
        z = z[:, None]
        x = [band]
        x_title = "Band"

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=x,
                y=block.channel_names,
                colorscale="Turbo",
                colorbar={"title": "Value"},
                hovertemplate="x=%{x}<br>channel=%{y}<br>value=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=430,
        margin={"l": 80, "r": 40, "t": 70, "b": 55},
        paper_bgcolor=PANEL,
        plot_bgcolor="white",
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "color": INK},
    )
    fig.update_xaxes(title_text=x_title, showgrid=False)
    fig.update_yaxes(title_text="Channel", autorange="reversed")
    return fig


def format_array_preview(array: np.ndarray, max_channels: int = 6, max_time: int = 12, max_bands: int = 5) -> str:
    if array.ndim == 2:
        snippet = array[:max_channels, :max_time]
    elif array.ndim == 3:
        snippet = array[:max_channels, :max_time, :max_bands]
    else:
        snippet = array[:max_time]
    return np.array2string(snippet, precision=3, suppress_small=True, threshold=10_000)


def signal_meta(block: SignalBlock) -> str:
    return (
        f"shape={block.data.shape} | fs={block.fs:.2f}Hz | duration={block.duration_sec:.2f}s | "
        f"channels={len(block.channel_names)} | units={block.units}"
    )


def feature_meta(block: FeatureBlock) -> str:
    return (
        f"shape={block.data.shape} | window={block.window_sec:.2f}s | "
        f"bands={len(block.band_names)} | channels={len(block.channel_names)}"
    )


def records_for_view(dataset: str, subject: str | None, session: str | None):
    return REGISTRY.filter_records(dataset=dataset, subject=subject, session=session)


def record_option_label(record) -> str:
    return f"Event {record.event} | {record.label_summary}"


def csv_bytes_for_record(record_id: str, channel_spec: ChannelSpec) -> bytes:
    block = REGISTRY.load_raw(record_id, channel_spec)
    frame = pd.DataFrame(block.data, index=block.channel_names)
    frame.index.name = "channel"
    buffer = io.StringIO()
    buffer.write(f"# {record_id}\n")
    buffer.write(f"# label: {block.label_info.summary}\n")
    frame.to_csv(buffer)
    return buffer.getvalue().encode("utf-8")


def zip_bytes_for_records(record_ids: list[str], channel_spec: ChannelSpec) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record_id in record_ids:
            record = REGISTRY.get_record(record_id)
            file_name = f"{record.dataset}_{record.subject}_{record.session}_event{record.event}.csv"
            archive.writestr(file_name, csv_bytes_for_record(record_id, channel_spec))
    return buffer.getvalue()


def filter_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Dataset", className="mini-label"),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        options=build_dropdown_options(REGISTRY.dataset_names()),
                        value=DEFAULT_DATASET,
                        clearable=False,
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Div("Subject", className="mini-label"),
                    dcc.Dropdown(id="subject-dropdown", clearable=False),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Div("Session / File", className="mini-label"),
                    dcc.Dropdown(id="session-dropdown", clearable=False),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Div("Event", className="mini-label"),
                    dcc.Dropdown(id="record-dropdown", clearable=False),
                ],
                className="control-block wide",
            ),
            html.Div(
                [
                    html.Div("Processed", className="mini-label"),
                    dcc.Dropdown(id="processed-dropdown", clearable=False),
                ],
                className="control-block wide",
            ),
            html.Div(
                [
                    html.Div("Band", className="mini-label"),
                    dcc.Dropdown(id="band-dropdown", clearable=False),
                ],
                id="feature-band-wrap",
                className="control-block",
            ),
            html.Div(
                [
                    html.Div("Online Preset", className="mini-label"),
                    dcc.Dropdown(
                        id="online-preset-dropdown",
                        clearable=False,
                        options=[{"label": spec["label"], "value": key} for key, spec in ONLINE_PRESET_SPECS.items()],
                        value="custom",
                    ),
                    html.Div("一键填充常见组合，例如 de_LDS、psd_movingAve。", className="hint-text"),
                ],
                id="online-preset-wrap",
                className="control-block",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div("Online Band Order", className="mini-label"),
                    dcc.Dropdown(
                        id="online-band-dropdown",
                        multi=True,
                        options=[{"label": spec["label"], "value": key} for key, spec in ONLINE_BAND_SPECS.items()],
                        value=["alpha_8_13"],
                    ),
                    html.Div("左到右为执行和显示顺序。留空表示不额外做 band 处理。", className="hint-text"),
                ],
                id="online-band-wrap",
                className="control-block wide",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div("Online Filter Order", className="mini-label"),
                    dcc.Dropdown(
                        id="online-filter-dropdown",
                        multi=True,
                        options=[{"label": spec["label"], "value": key} for key, spec in ONLINE_FILTER_SPECS.items()],
                        value=["lds"],
                    ),
                    html.Div("左到右依次执行，可组合 PSD、DE、LDS、Moving Average、Butterworth、Notch 等步骤。", className="hint-text"),
                ],
                id="online-filter-wrap",
                className="control-block wide",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div("Channel Range", className="mini-label"),
                    html.Div(
                        [
                            dcc.Dropdown(id="range-start-dropdown", clearable=False, placeholder="start"),
                            dcc.Dropdown(id="range-end-dropdown", clearable=False, placeholder="end"),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                    ),
                ],
                className="control-block wide",
            ),
            html.Div(
                [
                    html.Div("Channel Multi-Select", className="mini-label"),
                    dcc.Dropdown(id="channel-dropdown", multi=True, placeholder="leave empty = all channels"),
                ],
                className="control-block wide",
            ),
            html.Div(
                [
                    html.Div("DEAP Aux", className="mini-label"),
                    dcc.Checklist(
                        id="include-aux-toggle",
                        options=[{"label": "Include peripheral channels", "value": "include_aux"}],
                        value=[],
                    ),
                ],
                id="include-aux-wrap",
                className="control-block",
            ),
            html.Div(
                [
                    html.Button("Export Current Raw CSV", id="export-current-btn", n_clicks=0, className="action-btn"),
                    html.Button("Export Filtered ZIP", id="export-batch-btn", n_clicks=0, className="action-btn secondary"),
                ],
                className="button-stack",
            ),
            dcc.Download(id="download-current"),
            dcc.Download(id="download-batch"),
        ],
        className="top-grid",
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("EEG Raw / Processed Comparator", className="title"),
                html.Div(
                    f"Source: {DATASET_ROOT} | startup index records: {len(REGISTRY.records)}",
                    className="subtitle",
                ),
            ],
            className="hero",
        ),
        filter_panel(),
        html.Div(id="summary-text", className="summary-bar"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Indexed Records", className="section-title"),
                        dash_table.DataTable(
                            id="record-table",
                            columns=[
                                {"name": "Event", "id": "event"},
                                {"name": "Raw", "id": "raw_shape"},
                                {"name": "Processed", "id": "processed_shape"},
                                {"name": "Raw Hz", "id": "raw_fs"},
                                {"name": "Proc Hz", "id": "processed_fs"},
                                {"name": "Label", "id": "label_summary"},
                                {"name": "record_id", "id": "record_id", "hidden": True},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=14,
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": "#f7e6d9", "fontWeight": 700},
                            style_cell={
                                "backgroundColor": PANEL,
                                "border": f"1px solid {BORDER}",
                                "color": INK,
                                "fontFamily": "IBM Plex Sans, Segoe UI, sans-serif",
                                "fontSize": "13px",
                                "padding": "8px",
                                "textAlign": "left",
                            },
                        ),
                    ],
                    className="left-column",
                ),
                html.Div(
                    [
                        html.Div(id="label-badge"),
                        dcc.Graph(id="raw-graph"),
                        dcc.Graph(id="processed-graph"),
                    ],
                    className="right-column",
                ),
            ],
            className="content-grid",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Raw Preview", className="section-title"),
                        html.Div(id="raw-meta", className="meta-text"),
                        html.Pre(id="raw-preview", className="preview"),
                    ],
                    className="preview-panel",
                ),
                html.Div(
                    [
                        html.Div("Processed Preview", className="section-title"),
                        html.Div(id="processed-meta", className="meta-text"),
                        html.Pre(id="processed-preview", className="preview"),
                    ],
                    className="preview-panel",
                ),
                html.Div(
                    [
                        html.Div("Notes", className="section-title"),
                        html.Div(id="notes-panel", className="notes-text"),
                    ],
                    className="preview-panel",
                ),
            ],
            className="bottom-grid",
        ),
    ]
)


@app.callback(
    Output("subject-dropdown", "options"),
    Output("subject-dropdown", "value"),
    Input("dataset-dropdown", "value"),
    State("subject-dropdown", "value"),
)
def sync_subjects(dataset: str, current_subject: str | None):
    subjects = REGISTRY.subject_names(dataset)
    value = current_subject if current_subject in subjects else (subjects[0] if subjects else None)
    return build_dropdown_options(subjects), value


@app.callback(
    Output("session-dropdown", "options"),
    Output("session-dropdown", "value"),
    Input("dataset-dropdown", "value"),
    Input("subject-dropdown", "value"),
    State("session-dropdown", "value"),
)
def sync_sessions(dataset: str, subject: str | None, current_session: str | None):
    sessions = REGISTRY.session_names(dataset, subject)
    value = current_session if current_session in sessions else (sessions[0] if sessions else None)
    return build_dropdown_options(sessions), value


@app.callback(
    Output("record-dropdown", "options"),
    Output("record-dropdown", "value"),
    Output("record-table", "data"),
    Output("summary-text", "children"),
    Input("dataset-dropdown", "value"),
    Input("subject-dropdown", "value"),
    Input("session-dropdown", "value"),
    State("record-dropdown", "value"),
)
def sync_records(dataset: str, subject: str | None, session: str | None, current_record: str | None):
    records = records_for_view(dataset, subject, session)
    options = [{"label": record_option_label(record), "value": record.record_id} for record in records]
    valid_ids = {record.record_id for record in records}
    selected_record = current_record if current_record in valid_ids else (records[0].record_id if records else None)
    summary = f"{dataset} | subject={subject or '-'} | session={session or '-'} | indexed records={len(records)}"
    return options, selected_record, [record.to_row() for record in records], summary


@app.callback(
    Output("processed-dropdown", "options"),
    Output("processed-dropdown", "value"),
    Output("band-dropdown", "options"),
    Output("band-dropdown", "value"),
    Output("band-dropdown", "disabled"),
    Output("feature-band-wrap", "style"),
    Output("online-preset-wrap", "style"),
    Output("online-band-wrap", "style"),
    Output("online-filter-wrap", "style"),
    Output("range-start-dropdown", "options"),
    Output("range-start-dropdown", "value"),
    Output("range-end-dropdown", "options"),
    Output("range-end-dropdown", "value"),
    Output("channel-dropdown", "options"),
    Output("channel-dropdown", "value"),
    Output("include-aux-wrap", "style"),
    Input("record-dropdown", "value"),
    Input("include-aux-toggle", "value"),
    State("processed-dropdown", "value"),
    State("band-dropdown", "value"),
    State("range-start-dropdown", "value"),
    State("range-end-dropdown", "value"),
    State("channel-dropdown", "value"),
)
def sync_record_dependent_controls(
    record_id: str | None,
    include_aux_values: list[str] | None,
    current_processed: str | None,
    current_band: str | None,
    current_start: int | None,
    current_end: int | None,
    current_channels: list[str] | None,
):
    if not record_id:
        return [], None, [], None, True, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, [], None, [], None, [], [], {"display": "none"}

    include_aux = "include_aux" in (include_aux_values or [])
    record = REGISTRY.get_record(record_id)
    processed_keys = list(record.processed_options)
    if ONLINE_PROCESSING_KEY not in processed_keys:
        processed_keys.append(ONLINE_PROCESSING_KEY)
    processed_options = [{"label": format_processed_label(key), "value": key} for key in processed_keys]
    default_processed = record.default_processed_key if current_processed != ONLINE_PROCESSING_KEY else ONLINE_PROCESSING_KEY
    processed_value = current_processed if current_processed in processed_keys else default_processed

    band_values = [] if processed_value == ONLINE_PROCESSING_KEY else REGISTRY.get_band_options(record_id, processed_value)
    band_options = [{"label": "All Bands" if value == "__all__" else value, "value": value} for value in band_values]
    band_disabled = len(band_values) == 0
    band_value = current_band if current_band in band_values else ("__all__" if band_values else None)
    feature_band_style = {"display": "block"} if band_values and processed_value != ONLINE_PROCESSING_KEY else {"display": "none"}
    online_style = {"display": "block"} if processed_value == ONLINE_PROCESSING_KEY else {"display": "none"}

    channel_names = REGISTRY.get_available_channel_names(record_id, include_aux=include_aux)
    range_options = [{"label": f"{index + 1}: {name}", "value": index} for index, name in enumerate(channel_names)]
    start_value = current_start if current_start is not None and 0 <= current_start < len(channel_names) else 0
    end_value = current_end if current_end is not None and 0 <= current_end < len(channel_names) else len(channel_names) - 1
    selected_channels = [name for name in (current_channels or []) if name in channel_names]
    aux_style = {"display": "block"} if record.dataset == "DEAP" else {"display": "none"}
    return (
        processed_options,
        processed_value,
        band_options,
        band_value,
        band_disabled,
        feature_band_style,
        online_style,
        online_style,
        online_style,
        range_options,
        start_value,
        range_options,
        end_value,
        [{"label": name, "value": name} for name in channel_names],
        selected_channels,
        aux_style,
    )


@app.callback(
    Output("online-band-dropdown", "value"),
    Output("online-filter-dropdown", "value"),
    Input("online-preset-dropdown", "value"),
    State("online-band-dropdown", "value"),
    State("online-filter-dropdown", "value"),
)
def sync_online_preset(
    preset_key: str | None,
    current_bands: list[str] | None,
    current_filters: list[str] | None,
):
    if not preset_key or preset_key == "custom":
        return current_bands, current_filters
    preset = ONLINE_PRESET_SPECS[preset_key]
    return list(preset["bands"] or []), list(preset["filters"] or [])


@app.callback(
    Output("label-badge", "children"),
    Output("raw-graph", "figure"),
    Output("processed-graph", "figure"),
    Output("raw-meta", "children"),
    Output("processed-meta", "children"),
    Output("raw-preview", "children"),
    Output("processed-preview", "children"),
    Output("notes-panel", "children"),
    Input("record-dropdown", "value"),
    Input("processed-dropdown", "value"),
    Input("band-dropdown", "value"),
    Input("online-preset-dropdown", "value"),
    Input("online-band-dropdown", "value"),
    Input("online-filter-dropdown", "value"),
    Input("range-start-dropdown", "value"),
    Input("range-end-dropdown", "value"),
    Input("channel-dropdown", "value"),
    Input("include-aux-toggle", "value"),
)
def render_record(
    record_id: str | None,
    processed_key: str | None,
    band: str | None,
    online_preset: str | None,
    online_bands: list[str] | None,
    online_filters: list[str] | None,
    range_start: int | None,
    range_end: int | None,
    selected_channels: list[str] | None,
    include_aux_values: list[str] | None,
):
    if not record_id or not processed_key:
        empty = go.Figure()
        empty.update_layout(template="plotly_white")
        return "", empty, empty, "", "", "", "", ""

    include_aux = "include_aux" in (include_aux_values or [])
    channel_spec = ChannelSpec.from_ui(range_start, range_end, selected_channels, include_aux=include_aux)
    raw_block = REGISTRY.load_raw(record_id, channel_spec)
    if processed_key == ONLINE_PROCESSING_KEY:
        processed_block = build_online_processed_block(raw_block, online_bands, online_filters)
    else:
        processed_block = REGISTRY.load_processed(record_id, ProcessedSpec(key=processed_key, band=band or "__all__"), channel_spec)

    raw_figure = build_signal_figure(raw_block, f"Raw | {raw_block.label_info.summary}")
    if isinstance(processed_block, SignalBlock):
        processed_figure = build_signal_figure(processed_block, f"Processed | {processed_block.label_info.summary}")
        processed_meta_text = signal_meta(processed_block)
        processed_preview_text = format_array_preview(processed_block.data)
        processed_note = processed_block.source_note
    else:
        processed_figure = build_feature_figure(processed_block, band or "__all__")
        processed_meta_text = feature_meta(processed_block)
        processed_preview_text = format_array_preview(processed_block.data)
        processed_note = processed_block.source_note

    badge = format_label_badge(raw_block.label_info.summary, raw_block.label_info.discrete_label)
    notes = html.Div(
        [
            html.Div(f"Raw source: {raw_block.source_note or '-'}"),
            html.Div(f"Processed source: {processed_note or '-'}"),
            html.Div(f"Online preset: {ONLINE_PRESET_SPECS.get(online_preset or 'custom', ONLINE_PRESET_SPECS['custom'])['label'] if processed_key == ONLINE_PROCESSING_KEY else '-'}"),
            html.Div(f"Online order: {format_online_order(online_bands, online_filters) if processed_key == ONLINE_PROCESSING_KEY else '-'}"),
            html.Div("Processed preview below is the current processed result array."),
            html.Div(f"Current record: {record_id}"),
        ]
    )
    return (
        badge,
        raw_figure,
        processed_figure,
        signal_meta(raw_block),
        processed_meta_text,
        format_array_preview(raw_block.data),
        processed_preview_text,
        notes,
    )


@app.callback(
    Output("download-current", "data"),
    Input("export-current-btn", "n_clicks"),
    State("record-dropdown", "value"),
    State("range-start-dropdown", "value"),
    State("range-end-dropdown", "value"),
    State("channel-dropdown", "value"),
    State("include-aux-toggle", "value"),
    prevent_initial_call=True,
)
def export_current(
    n_clicks: int,
    record_id: str | None,
    range_start: int | None,
    range_end: int | None,
    selected_channels: list[str] | None,
    include_aux_values: list[str] | None,
):
    if not n_clicks or not record_id:
        return no_update
    include_aux = "include_aux" in (include_aux_values or [])
    channel_spec = ChannelSpec.from_ui(range_start, range_end, selected_channels, include_aux=include_aux)
    record = REGISTRY.get_record(record_id)
    filename = f"{record.dataset}_{record.subject}_{record.session}_event{record.event}.csv"
    return dcc.send_bytes(csv_bytes_for_record(record_id, channel_spec), filename)


@app.callback(
    Output("download-batch", "data"),
    Input("export-batch-btn", "n_clicks"),
    State("record-table", "data"),
    State("dataset-dropdown", "value"),
    State("subject-dropdown", "value"),
    State("session-dropdown", "value"),
    State("range-start-dropdown", "value"),
    State("range-end-dropdown", "value"),
    State("channel-dropdown", "value"),
    State("include-aux-toggle", "value"),
    prevent_initial_call=True,
)
def export_batch(
    n_clicks: int,
    table_data: list[dict[str, object]] | None,
    dataset: str,
    subject: str | None,
    session: str | None,
    range_start: int | None,
    range_end: int | None,
    selected_channels: list[str] | None,
    include_aux_values: list[str] | None,
):
    if not n_clicks or not table_data:
        return no_update
    include_aux = "include_aux" in (include_aux_values or [])
    channel_spec = ChannelSpec.from_ui(range_start, range_end, selected_channels, include_aux=include_aux)
    record_ids = [str(row["record_id"]) for row in table_data if row.get("record_id")]
    filename = f"{dataset}_{subject or 'all'}_{session or 'all'}_raw_export.zip"
    return dcc.send_bytes(zip_bytes_for_records(record_ids, channel_spec), filename)


if __name__ == "__main__":
    app.run(
        debug=False,
        host=os.environ.get("EEG_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("EEG_WEB_PORT", "8050")),
    )
