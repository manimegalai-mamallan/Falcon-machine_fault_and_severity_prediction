# VibeDiag - Vibration Fault Diagnosis

Streamlit dashboard for rotating-machinery fault diagnosis. Three model
backends (1D CNN, 2D CNN, MiniRocket) plug into a single dashboard.

## What it does

For any uploaded vibration recording, the app:

1. Predicts the fault class: `Normal`, `Unbalance`, `Misalignment`, or `Looseness`
2. Predicts severity: `Low` / `Medium` / `High`
3. Renders a 12-panel dashboard with FFT, spectrogram, axis coupling, and
   maintenance guidance.

## Project layout

```
vibediag/
├── app/
│   ├── streamlit_app.py        the dashboard UI
│   ├── plots.py                plotly figure builders
│   └── model_registry.py       loads the selected model
│
├── models/
│   ├── shared/                 used by all 3 models + the app
│   │   ├── config.py
│   │   ├── data_utils.py
│   │   ├── recommendations.py
│   │   ├── dashboard_builders.py
│   │   ├── predictor_contract.py
│   │   ├── setup_data.py
│   │   └── build_references.py
│   ├── cnn2d/                  2D CNN -- working reference implementation
│   ├── cnn1d/                  1D CNN -- teammate's slot (stub for now)
│   └── minirocket/             MiniRocket -- teammate's slot (stub for now)
│
├── MMS_Data/                   (gitignored) drop your .jsonl files here
├── test_data/                  (gitignored) populated by setup_data.py
├── requirements.txt
└── README.md
```

## One-time setup

```bash
git clone <your-repo-url>
cd vibediag
python -m venv venv
source venv/bin/activate           # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For NVIDIA GPU support, install PyTorch with CUDA from
https://pytorch.org/get-started/locally/ before installing requirements.

## Workflow

```bash
# 1. Place your MMS_Data/ folder (with all 30 .jsonl files) at the
#    project root, alongside README.md and requirements.txt.

# 2. Move 6 held-out files into test_data/
python -m models.shared.setup_data

# 3. Train the 2D CNN  (~45-60 min on GPU, ~3 hours on CPU)
python -m models.cnn2d.train

# 4. Build healthy-reference FFTs (used by the dashboard's comparison panel)
python -m models.shared.build_references

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. Pick a model from the selector,
upload a `.jsonl` from `test_data/`, and the dashboard renders.

## Adding the 1D CNN and MiniRocket

The selector knows about all three models, but only `2D CNN` is
implemented. To add the others:

1. Replace `models/cnn1d/predict.py` (or `models/minirocket/predict.py`)
   with a real implementation.
2. Place the trained weights in `models/cnn1d/artifacts/` (or
   `models/minirocket/artifacts/`).
3. The selector picks them up automatically.

The contract every `predict.py` must follow is in
`models/shared/predictor_contract.py`. The reference implementation is
`models/cnn2d/predict.py`.

The key insight: each model only needs to produce per-window probability
arrays. Everything else (FFT, spectrogram, axis coupling, recommendations)
is computed by the shared `build_full_dashboard_dict()` helper. So all
three models produce identical-shaped dashboard dicts, and the Streamlit
app code never needs to know which model is in use.

## How the model selector works

`app/model_registry.py` defines the registry of models. Each entry has:

- `key`         — internal id
- `label`       — display name in the UI
- `description` — shown under the selector
- `module`      — python module path to import
- `param_count` — for the UI label

The selector calls `is_available(key)` to decide which models can be
selected. If a model's artifacts haven't been built, the selector greys it
out and shows "not loaded" rather than crashing.

## Why this design

- **Single-fault softmax** matches the use case: exactly one of
  {Normal, Unbalance, Misalignment, Looseness} is active per recording.
- **Multi-task shared backbone** (2D CNN) -- both fault and severity heads
  share features, improving data efficiency.
- **STFT spectrograms** expose the frequency-domain structure (1x, 2x, 3x
  shaft harmonics) that defines vibration faults.
- **RPM conditioning** lets one model generalize across operating speeds.
- **Held-out test set** (6 files, ~20%) gives an honest accuracy estimate.
- **Dashboard data API** -- `predict.py` returns one dict the Streamlit app
  layouts directly. No DSP code in the UI.
- **Modular per-model folders** -- teammates can add their model without
  reading or touching the app code.
