# EEG Web 使用文档

这是一个本地运行的 `Dash + Plotly` EEG 数据查看器，用来对比 `raw` 和 `processed` 数据，支持：

- 浏览 `DEAP / SEED / SEED-VIG / SEED-VII / SEED_IV`
- 查看每条事件的 `raw / processed` shape
- 上下双图对比
- 通道范围选择和通道多选
- `线上处理`
- 在线预设：`de_movingAve / de_LDS / psd_movingAve / psd_LDS`
- 导出当前事件 raw CSV
- 按筛选结果批量导出 ZIP
- **ICA / ICLabel 分析页面** (`/ica`)

## 1. 环境要求

- Windows
- Python `3.13` 已验证可运行
- 数据集目录完整

当前项目依赖见 [requirements.txt](f:/EEG_web/requirements.txt)：

```txt
numpy==2.2.3
pandas==2.3.3
scipy==1.16.3
mne==1.11.0
dash==4.1.0
plotly==6.6.0
openpyxl==3.1.5
mne-icalabel>=0.7
scikit-learn>=1.3
onnxruntime>=1.16
```

> **ICA 依赖说明**：`mne-icalabel` 需要 `scikit-learn`（用于 extended infomax ICA）和 `onnxruntime`（用于 ICLabel 推理）。如果环境里已有 `torch`，`mne-icalabel` 也可以使用它作为后端。

安装：

```powershell
cd f:\EEG_web
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 数据怎么放

程序默认读取：

```txt
E:\BaiduNetdiskDownload\dataset
```

默认目录下必须包含这 5 个一级子目录，名字要和下面完全一致：

```txt
E:\BaiduNetdiskDownload\dataset
├─ DEAP
├─ SEED
├─ SEED-VIG
├─ SEED-VII
└─ SEED_IV
```

如果你的数据不在这个位置，不需要改代码，直接在运行前设置环境变量：

```powershell
$env:EEG_DATASET_ROOT = "D:\your\dataset"
```

程序只认“数据集根目录”，不需要额外拷贝到项目目录里。

### 每个数据集的关键目录

程序当前会读取这些位置：

- `DEAP`
  - `data_original`
  - `data_preprocessed_python`
  - `data_preprocessed_matlab`
- `SEED`
  - `Preprocessed_EEG`
  - `ExtractedFeatures`
  - `channel-order.xlsx`
- `SEED_IV`
  - `eeg_raw_data`
  - `eeg_feature_smooth`
  - `Channel Order.xlsx`
- `SEED-VIG`
  - `Raw_Data`
  - `EEG_Feature_5Bands`
  - `EEG_Feature_2Hz`
  - `perclos_labels`
- `SEED-VII`
  - `EEG_raw`
  - `EEG_preprocessed`
  - `continuous_labels`
  - `save_info`
  - `emotion_label_and_stimuli_order.xlsx`
  - `Channel Order.xlsx`

## 3. 怎么运行

在 PowerShell 中：

```powershell
cd f:\EEG_web
.\.venv\Scripts\Activate.ps1
python app.py
```

默认启动地址：

```txt
http://127.0.0.1:8050
```

如果你要改 host 或 port：

```powershell
$env:EEG_WEB_HOST = "127.0.0.1"
$env:EEG_WEB_PORT = "8051"
python app.py
```

## 4. 页面怎么用

顶部从左到右主要是：

- `Dataset`：选择数据集
- `Subject`：选择被试
- `Session / File`：选择 session 或文件
- `Event`：选择事件片段
- `Processed`：选择官方 processed 或 `线上处理`
- `Band`：当选的是数据集自带 feature 时，选择显示哪个 band
- `Online Preset`：线上处理预设
- `Online Band Order`：在线 band 顺序
- `Online Filter Order`：在线 filter / feature 顺序
- `Channel Range`：按通道范围选
- `Channel Multi-Select`：按名字多选
- `DEAP Aux`：是否包含 DEAP 外围生理通道

左侧表格显示：

- 当前筛选下所有事件
- 每条事件的 `raw_shape / processed_shape / raw_fs / processed_fs / label_summary`

右侧图像显示：

- 上图：`raw`
- 下图：`processed`
  - 如果是时域信号，显示堆叠折线
  - 如果是特征，显示热力图

底部显示：

- `Raw Preview`
- `Processed Preview`
- `Notes`

## 5. 线上处理怎么用

把 `Processed` 切换成：

```txt
线上处理
```

这时下图和 `Processed Preview` 都不再使用数据集自带 processed，而是使用当前 raw 片段实时处理后的结果。

### Online Preset

当前提供这些一键预设：

- `Custom`
- `de_movingAve`
- `de_LDS`
- `psd_movingAve`
- `psd_LDS`

这些预设会自动填充：

- `Online Band Order`
- `Online Filter Order`

### Online Band Order

支持：

- `No Band / Raw`
- `Broad 1-50Hz`
- `Delta 1-4Hz`
- `Theta 4-8Hz`
- `Alpha 8-13Hz`
- `Mu 8-12Hz`
- `Beta 13-30Hz`
- `Gamma 30-45Hz`

如果选多个 band，会按顺序分别处理，并在下图里按 band 堆叠或组合显示。

### Online Filter Order

支持：

- `LDS Smoother`
- `Butterworth Highpass 1Hz`
- `Butterworth Lowpass 40Hz`
- `Butterworth Lowpass 30Hz`
- `Notch 50Hz`
- `Notch 60Hz`
- `Moving Average (5)`
- `PSD (1s windows)`
- `DE (1s windows)`
- `Median (5)`
- `Savitzky-Golay (11)`

说明：

- 顺序按下拉框中从左到右执行
- `PSD / DE` 会把时域信号转成特征窗口
- 在 `PSD / DE` 之后，`LDS / Moving Average` 会作用在特征窗口上
- `Processed Preview` 显示的是线上处理之后的真实结果数组

## 6. 导出

支持两个按钮：

- `Export Current Raw CSV`
  - 导出当前事件的 raw 数据
- `Export Filtered ZIP`
  - 按当前表格筛选结果批量导出 raw CSV

注意：

- 导出的是 `raw` 数据，不是 processed
- 导出会受当前通道选择影响

## 7. 当前实现里的数据说明

- `SEED` 数据集没有单独 raw 文件
  - 当前实现把 `Preprocessed_EEG` 作为最低层 EEG 来源来显示 raw
- `SEED-VII`
  - raw 片段是从 `EEG_raw/*.cnt` 中按 `save_info/*_trigger_info.csv` 切出来的
- `SEED-VIG`
  - 连续标签来自 `perclos_labels`
- `DEAP`
  - raw 片段来自 `Status 4 -> 5` 事件，并向前补 3 秒基线

## 8. 常见问题

### 1) 页面启动很慢

首次启动会扫描全部数据集并建立索引，数据量较大，十几秒是正常的。

### 2) 提示找不到数据集

先确认：

- 根目录存在
- 5 个一级目录名字完全一致
- 如果不在默认路径，已经设置 `EEG_DATASET_ROOT`

### 3) Excel 读取失败

请确认已安装：

```powershell
pip install openpyxl
```

### 4) `Processed Preview` 显示的是不是线上处理后的结果

是。当前实现里：

- 官方 processed 模式：显示官方 processed 数据
- `线上处理` 模式：显示线上处理后的结果

## 9. 主要文件

- [app.py](f:/EEG_web/app.py)
  - Web UI、图像、线上处理、导出、ICA 页面
- [dataset_adapters.py](f:/EEG_web/dataset_adapters.py)
  - 五个数据集的扫描、加载、标签映射和 ICA 数据源装配
- [models.py](f:/EEG_web/models.py)
  - 数据结构定义（含 ICA 相关类型）
- [ica_pipeline.py](f:/EEG_web/ica_pipeline.py)
  - ICA 预处理、拟合、ICLabel 分类、图像生成、缓存

## 10. ICA / ICLabel 页面

### 入口

在主页面点击 **ICA View** 按钮，会自动跳转到 `/ica` 页面，并带上当前选定的 record、dataset、subject、session 上下文。

### 使用方法

1. 选择 **Scope**：
   - `Current Event`：只用当前事件的 EEG 做 ICA
   - `Current Session / File`：用整个 session 或文件的 EEG 做 ICA
2. 调整 **Reject Threshold**（默认 0.80）
3. 点击 **Run ICA**
4. 查看结果：
   - 摘要卡片：scope、采样率、通道数、IC 数量、建议剔除数
   - 组件表：IC 序号、分类标签、各类概率、建议剔除标记
   - 点击表格行查看组件详情（topomap、概率条形图、激活时序、PSD）
   - 清理前后对比图（原始 EEG vs 剔除 IC 后的 EEG）

### Session vs Event 差异

| 数据集 | Event 范围 | Session 范围 |
|--------|-----------|-------------|
| DEAP | 单 trial BDF 片段 (30 EEG ch) | 整段 `.bdf` EEG |
| SEED | 单 `*_eeg{N}` trial | 当前 session 15 个 trial → `EpochsArray` |
| SEED-IV | 单 trial raw | 当前 session 24 个 trial → `EpochsArray` |
| SEED-VIG | 8s 原始片段 | 整段 `EEG.data` 连续记录 |
| SEED-VII | CNT 事件片段 | 当前 subject-session 整段 `.cnt` EEG |

### ICLabel 预处理假设

按官方 `mne-icalabel` 示例固定：

- 平均参考 (common average reference)
- 带通 1 Hz – min(100 Hz, Nyquist - 1 Hz)
- 采样率 > 200 Hz 时重采样到 200 Hz
- `ICA(method="infomax", fit_params={"extended": True}, random_state=97, max_iter="auto")`
- Session 范围额外使用 `decim=2`

ICLabel 输出 7 类：`brain`、`muscle artifact`、`eye blink`、`heart beat`、`line noise`、`channel noise`、`other`。

默认建议剔除：类别不是 `brain` / `other` 且最大概率 ≥ 阈值（默认 0.80）。

### 限制与注意

- ICA 只使用 EEG 通道；DEAP 的外围生理通道和 SEED-VII 的 M1/M2/ECG/HEO/VEO 不进入 ICA
- 短片段 (<30s) 或少通道 (<8) 时会显示低置信度警告
- ICA 计算可能耗时较长（session 级别可达 30-120 秒），请耐心等待

## 11. 常见问题补充

### 5) 页面报 callback 错误或显示异常

这通常是"回调 schema 失配"：浏览器端缓存了旧的回调依赖图，但服务端已经更新。

**自动处理**：页面内置了 schema 版本检查，检测到失配时会自动强制刷新。

**手动处理**：
1. 停止服务器 (`Ctrl+C`)
2. 重启 `python app.py`
3. 在浏览器中按 `Ctrl+Shift+R` 强制硬刷新

### 6) ICA 页面报依赖错误

确认已安装 ICA 相关依赖：

```powershell
pip install mne-icalabel scikit-learn onnxruntime
```

或者直接：

```powershell
pip install -r requirements.txt
```
