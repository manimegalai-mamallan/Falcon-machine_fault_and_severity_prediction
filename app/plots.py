"""
Plotly figure builders for the v3 dashboard (dark theme, all 12 sections).

Every plot uses transparent background so the streamlit page background
shows through.

Color choices are slightly muted versions of the v2 palette to read well
on a dark slate background without burning the eyes.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Dark-friendly color palette
# ---------------------------------------------------------------------------
COLOR_X        = "#5BA0E8"   # softer blue
COLOR_Y        = "#D69042"   # softer amber
COLOR_Z        = "#9B92E8"   # softer purple
COLOR_BASELINE = "#7C7B73"   # neutral gray
COLOR_MEASURED = "#E07050"   # softer coral
COLOR_PRED     = "#E58060"   # accent
COLOR_HEALTHY  = "#3FB58F"   # softer green / teal

CLASS_COLORS = {
    "Normal":       "#7C7B73",   # neutral gray
    "Unbalance":    "#D69042",   # amber
    "Misalignment": "#3FB58F",   # green
    "Looseness":    "#9B92E8",   # purple
}

# Fill colors (rgba, 18% alpha)
FILL_X        = "rgba(91,160,232,0.18)"
FILL_Y        = "rgba(214,144,66,0.18)"
FILL_Z        = "rgba(155,146,232,0.18)"
FILL_BASELINE = "rgba(124,123,115,0.22)"
FILL_MEASURED = "rgba(224,112,80,0.18)"
FILL_HEALTHY  = "rgba(63,181,143,0.18)"

# Plotly grid/axis colors for dark mode
GRID  = "rgba(255,255,255,0.06)"
ZERO  = "rgba(255,255,255,0.10)"
TEXT  = "#c8c5bd"
MUTED = "#8a877f"


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------
def _base_layout(height: int = 260, **kwargs) -> dict:
    layout = dict(
        height=height,
        margin=dict(l=44, r=20, t=22, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=12, color=TEXT),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1d2028", bordercolor="rgba(255,255,255,0.1)",
                        font_size=12, font_color=TEXT),
    )
    # Apply dark grid colors to any axes the caller specifies
    layout.update(kwargs)
    return layout


def _dark_axis(**kwargs) -> dict:
    """Standard dark-mode axis dict."""
    base = dict(
        gridcolor=GRID,
        zerolinecolor=ZERO,
        linecolor="rgba(255,255,255,0.15)",
        tickcolor="rgba(255,255,255,0.2)",
        title_font_color=TEXT,
        tickfont=dict(color=MUTED, size=11),
    )
    base.update(kwargs)
    return base


# ===========================================================================
# 1. Main spectrum -- annotated FFT with healthy baseline
# ===========================================================================
def main_spectrum_with_baseline(freq: dict) -> go.Figure:
    fig = go.Figure()
    freqs = freq["freqs_hz"]
    measured_y = freq["magnitude_y"]
    hr = freq.get("healthy_reference")

    if hr is not None:
        fig.add_trace(go.Scatter(
            x=hr["freqs"], y=hr["magnitude_y"],
            mode="lines", line=dict(color=COLOR_BASELINE, width=1.2),
            fill="tozeroy", fillcolor=FILL_BASELINE,
            name="healthy baseline",
            hovertemplate="%{x:.1f} Hz<br>baseline %{y:.4f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=freqs, y=measured_y,
        mode="lines", line=dict(color=COLOR_MEASURED, width=1.8),
        name="measured today",
        hovertemplate="%{x:.1f} Hz<br>measured %{y:.4f}<extra></extra>",
    ))

    y_max = float(max(np.max(measured_y),
                      np.max(hr["magnitude_y"]) if hr is not None else 0)) * 1.15
    if y_max <= 0:
        y_max = 1.0

    for marker in freq["markers"].values():
        is_diag = bool(marker.get("is_diagnostic", False))
        color = COLOR_MEASURED if is_diag else "rgba(255,255,255,0.25)"
        width = 1.5 if is_diag else 0.5
        fig.add_shape(
            type="line", x0=marker["freq_hz"], x1=marker["freq_hz"],
            y0=0, y1=y_max,
            line=dict(color=color, width=width, dash="dash"),
        )
        fig.add_annotation(
            x=marker["freq_hz"], y=y_max,
            text=(f"<b>{marker['harmonic']}× ({marker['freq_hz']:.0f} Hz)</b>"
                  if is_diag else
                  f"{marker['harmonic']}× ({marker['freq_hz']:.0f} Hz)"),
            showarrow=False, yanchor="bottom",
            font=dict(size=10, color=color if is_diag else MUTED),
        )

    annotation = freq.get("annotation")
    if annotation is not None and freq.get("healthy_reference") is not None:
        x_diag = annotation["freq_hz"]
        idx = int(np.argmin(np.abs(freqs - x_diag)))
        lo = max(0, idx - 10); hi = min(len(measured_y), idx + 11)
        peak_y = float(np.max(measured_y[lo:hi]))
        callout_x = x_diag + (freq.get("display_max_hz", 200.0) * 0.18)
        callout_y = peak_y + (y_max - peak_y) * 0.45
        fig.add_annotation(
            x=callout_x, y=callout_y, ax=x_diag, ay=peak_y,
            xref="x", yref="y", axref="x", ayref="y",
            text=(f"<b>{annotation['label']}</b><br>"
                  f"<span style='font-size:10px;color:#dab2a6'>"
                  f"{annotation['sublabel']}</span>"),
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor=COLOR_MEASURED,
            bgcolor="#3a1f15", bordercolor=COLOR_MEASURED,
            borderwidth=1, borderpad=6,
            font=dict(size=11, color="#f0d5cb"),
            align="left",
        )

    fig.update_layout(**_base_layout(
        height=320,
        xaxis=_dark_axis(title="Frequency (Hz)",
                         range=[0, freq["display_max_hz"]]),
        yaxis=_dark_axis(title="Magnitude", rangemode="tozero"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT)),
    ))
    return fig


# ===========================================================================
# 2. Class-signature comparison -- 4 mini-FFTs (one per class) with sample
# ===========================================================================
def class_signature_comparison(freq: dict, signatures: dict, target_class: str) -> go.Figure:
    """For one class, plot the class-mean Y-axis FFT (gray fill) with the
    uploaded sample's Y-axis FFT (red line) overlaid.
    """
    sample_freqs = freq["freqs_hz"]
    sample_y     = freq["magnitude_y"]

    sig = signatures.get(target_class)
    fig = go.Figure()
    if sig is not None:
        fig.add_trace(go.Scatter(
            x=sig["freqs"], y=sig["mag_y"],
            mode="lines", line=dict(color=COLOR_BASELINE, width=1),
            fill="tozeroy", fillcolor=FILL_BASELINE,
            hovertemplate=f"%{{x:.1f}} Hz<br>{target_class} mean %{{y:.4f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=sample_freqs, y=sample_y,
        mode="lines", line=dict(color=COLOR_MEASURED, width=1.4),
        hovertemplate="%{x:.1f} Hz<br>your sample %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        height=140,
        xaxis=_dark_axis(range=[0, freq["display_max_hz"]],
                         showticklabels=True, tickfont=dict(size=9)),
        yaxis=_dark_axis(showticklabels=False, rangemode="tozero"),
        margin=dict(l=10, r=10, t=10, b=28),
    ))
    return fig


# ===========================================================================
# 3. Per-axis mini-spectrum (X, Y, Z each in their own card)
# ===========================================================================
def per_axis_mini_spectrum(freq: dict, axis: str) -> go.Figure:
    color_map = {"x": COLOR_X, "y": COLOR_Y, "z": COLOR_Z}
    fill_map  = {"x": FILL_X, "y": FILL_Y, "z": FILL_Z}
    mag_map   = {"x": freq["magnitude_x"], "y": freq["magnitude_y"], "z": freq["magnitude_z"]}
    color = color_map[axis]; mag = mag_map[axis]; fill = fill_map[axis]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq["freqs_hz"], y=mag,
        mode="lines", line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x:.1f} Hz<br>%{y:.4f}<extra></extra>",
    ))
    for marker in freq["markers"].values():
        if marker.get("is_diagnostic"):
            fig.add_shape(
                type="line", x0=marker["freq_hz"], x1=marker["freq_hz"],
                y0=0, y1=float(np.max(mag)) * 1.05,
                line=dict(color=COLOR_MEASURED, width=1.2, dash="dash"),
            )
    fig.update_layout(**_base_layout(
        height=140,
        xaxis=_dark_axis(range=[0, freq["display_max_hz"]],
                         tickfont=dict(size=9)),
        yaxis=_dark_axis(showticklabels=False, rangemode="tozero"),
        margin=dict(l=10, r=10, t=10, b=28),
    ))
    return fig


# ===========================================================================
# 4. Spectrogram (the model's actual input)
# ===========================================================================
def spectrogram_panel(spec: dict) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=spec["magnitude_log"],
        x=spec["times_seconds"], y=spec["frequencies_hz"],
        colorscale="Cividis", showscale=False,
        hovertemplate="t=%{x:.1f}s<br>f=%{y:.0f}Hz<br>%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        height=240,
        xaxis=_dark_axis(title="Time (s)"),
        yaxis=_dark_axis(title="Frequency (Hz)"),
    ))
    return fig


# ===========================================================================
# 5. Per-window prediction trace
# ===========================================================================
def per_window_trace(per_window: dict) -> go.Figure:
    fig = go.Figure()
    indices = per_window["indices"]
    for cls, probs in per_window["fault_probs"].items():
        fig.add_trace(go.Scatter(
            x=indices, y=probs, mode="lines",
            line=dict(color=CLASS_COLORS.get(cls, "#888"), width=1.6),
            name=cls,
            hovertemplate=f"window %{{x}}<br>{cls}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1,
                  annotation_text="threshold 0.5",
                  annotation_position="bottom right",
                  annotation_font_size=10,
                  annotation_font_color=MUTED)
    fig.update_layout(**_base_layout(
        height=220, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT)),
        xaxis=_dark_axis(title="Window index"),
        yaxis=_dark_axis(title="Probability", range=[0, 1.05]),
    ))
    return fig


# ===========================================================================
# 6. Confusion matrix
# ===========================================================================
def confusion_matrix_heatmap(matrix, labels, title: str = "") -> go.Figure:
    matrix = np.asarray(matrix, dtype=int)
    text = [[str(v) if v else "·" for v in row] for row in matrix]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=[
            [0.0, "rgba(91,160,232,0.05)"],
            [0.4, "rgba(91,160,232,0.45)"],
            [1.0, "rgba(91,160,232,0.95)"],
        ],
        showscale=False, text=text, texttemplate="%{text}",
        textfont=dict(size=14, family="ui-monospace", color="#e8e6df"),
        hovertemplate="true %{y}<br>predicted %{x}<br>count: %{z}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        height=280,
        xaxis=_dark_axis(title="Predicted", side="bottom"),
        yaxis=_dark_axis(title="Actual", autorange="reversed"),
        margin=dict(l=70, r=20, t=20, b=50),
    ))
    return fig


# ===========================================================================
# 7. Training history (loss + accuracy curves)
# ===========================================================================
def training_history_curves(history: list) -> go.Figure:
    """Plot loss + accuracy curves. Accepts both the new cascaded
    history format (key 'acc') and the older multi-task format
    (key 'fault_acc')."""
    def _acc(entry: dict) -> float:
        # Cascaded models save 'acc'; older multi-task saved 'fault_acc'.
        return entry.get("acc", entry.get("fault_acc", 0.0))

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss   = [h["val"]["loss"]   for h in history]
    train_acc  = [_acc(h["train"]) for h in history]
    val_acc    = [_acc(h["val"])   for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers",
                             line=dict(color=COLOR_X, width=1.5),
                             marker=dict(size=4),
                             name="train loss",
                             hovertemplate="epoch %{x}<br>train loss %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers",
                             line=dict(color=COLOR_MEASURED, width=1.5),
                             marker=dict(size=4),
                             name="val loss",
                             hovertemplate="epoch %{x}<br>val loss %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers",
                             line=dict(color=COLOR_HEALTHY, width=1.5, dash="dot"),
                             marker=dict(size=4),
                             name="train acc",
                             yaxis="y2",
                             hovertemplate="epoch %{x}<br>train acc %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers",
                             line=dict(color=COLOR_Z, width=1.5, dash="dot"),
                             marker=dict(size=4),
                             name="val acc",
                             yaxis="y2",
                             hovertemplate="epoch %{x}<br>val acc %{y:.3f}<extra></extra>"))
    fig.update_layout(**_base_layout(
        height=260, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT)),
        xaxis=_dark_axis(title="Epoch"),
        yaxis=_dark_axis(title="Loss", rangemode="tozero"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right",
                    range=[0, 1.05], gridcolor="rgba(0,0,0,0)",
                    title_font_color=TEXT,
                    tickfont=dict(color=MUTED, size=11)),
    ))
    return fig


# ===========================================================================
# 8. Per-class accuracy bar chart
# ===========================================================================
def per_class_accuracy_bars(per_class_dict: dict) -> go.Figure:
    classes = list(per_class_dict.keys())
    accs    = [(per_class_dict[c]["accuracy"] or 0.0) * 100
               for c in classes]
    counts  = [per_class_dict[c]["total"] for c in classes]
    text    = [f"{a:.0f}%" if c > 0 else "n/a" for a, c in zip(accs, counts)]
    colors  = [CLASS_COLORS.get(c, COLOR_X) for c in classes]
    hover   = [f"{c}: {per_class_dict[c]['correct']}/{per_class_dict[c]['total']}"
               for c in classes]

    fig = go.Figure(data=go.Bar(
        x=classes, y=accs, marker_color=colors,
        text=text, textposition="outside",
        textfont=dict(color=TEXT, size=12),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        height=240,
        xaxis=_dark_axis(),
        yaxis=_dark_axis(title="Accuracy (%)", range=[0, 110]),
    ))
    return fig


# ===========================================================================
# 9. Healthy vs Your Machine — business-friendly FFT comparison
# ===========================================================================
def healthy_vs_yours_fft(freq: dict, top: bool) -> go.Figure:
    """Build ONE of the two stacked plots for the 'Healthy vs Yours' view.

    top=True:  the healthy reference (gray, fills under the curve)
    top=False: your measurement today (orange, fills under the curve)

    Plain-English axis labels — no Hz, no harmonics, no jargon.
    Used in tandem (call twice — once with top=True, once with top=False) to
    produce a stacked comparison.
    """
    freqs = freq["freqs_hz"]
    hr = freq.get("healthy_reference")

    fig = go.Figure()
    if top:
        # Healthy reference (gray)
        if hr is not None:
            fig.add_trace(go.Scatter(
                x=hr["freqs"], y=hr["magnitude_y"],
                mode="lines",
                line=dict(color=COLOR_BASELINE, width=1.4),
                fill="tozeroy", fillcolor=FILL_BASELINE,
                name="healthy",
                hovertemplate="vibration speed: %{x:.0f}<br>strength: %{y:.4f}<extra></extra>",
            ))
        else:
            # Synthesize a placeholder flat-ish line so the plot isn't empty
            import numpy as np
            placeholder = np.full_like(freqs, 0.05, dtype=np.float32)
            fig.add_trace(go.Scatter(
                x=freqs, y=placeholder,
                mode="lines",
                line=dict(color=COLOR_BASELINE, width=1.4, dash="dot"),
                fill="tozeroy", fillcolor=FILL_BASELINE,
                hovertemplate="(healthy reference unavailable for this speed)<extra></extra>",
            ))
    else:
        # Your measurement today (orange)
        fig.add_trace(go.Scatter(
            x=freqs, y=freq["magnitude_y"],
            mode="lines",
            line=dict(color=COLOR_MEASURED, width=1.6),
            fill="tozeroy", fillcolor=FILL_MEASURED,
            name="your machine",
            hovertemplate="vibration speed: %{x:.0f}<br>strength: %{y:.4f}<extra></extra>",
        ))

    # Set y-axis to a shared range so both plots are visually comparable.
    # Use the larger of (your magnitude, healthy magnitude) so the healthy
    # plot doesn't look artificially flat.
    import numpy as np
    yours_max  = float(np.max(freq["magnitude_y"])) if len(freq["magnitude_y"]) else 1.0
    healthy_max = (
        float(np.max(hr["magnitude_y"])) if hr is not None and len(hr["magnitude_y"])
        else 0.0
    )
    y_max = max(yours_max, healthy_max) * 1.10
    if y_max <= 0:
        y_max = 1.0

    fig.update_layout(**_base_layout(
        height=180,
        xaxis=_dark_axis(
            title="How fast the machine vibrates →" if not top else None,
            range=[0, freq["display_max_hz"]],
            showticklabels=not top,
        ),
        yaxis=_dark_axis(
            title="How strong",
            range=[0, y_max],
            showticklabels=False,
        ),
        margin=dict(l=44, r=20, t=10, b=40 if not top else 10),
    ))
    return fig


# ===========================================================================
# 10. Pattern Match — your sample overlaid on each class signature
# ===========================================================================
def pattern_match_panel(freq: dict, signatures: dict, target_class: str) -> go.Figure:
    """One mini panel for the pattern-match grid (4 panels total).

    Shows the class-mean Y-axis FFT (gray fill) with your sample's FFT
    (orange line) overlaid. Visually obvious whether your shape tracks
    the reference shape.
    """
    sample_freqs = freq["freqs_hz"]
    sample_y     = freq["magnitude_y"]
    sig          = signatures.get(target_class)

    fig = go.Figure()

    # Class signature in gray (filled)
    if sig is not None:
        fig.add_trace(go.Scatter(
            x=sig["freqs"], y=sig["mag_y"],
            mode="lines",
            line=dict(color=COLOR_BASELINE, width=1),
            fill="tozeroy", fillcolor=FILL_BASELINE,
            hovertemplate=f"vibration speed: %{{x:.0f}}<br>{target_class} pattern: %{{y:.4f}}<extra></extra>",
        ))

    # Your sample in orange (line, no fill — keeps the gray pattern visible)
    fig.add_trace(go.Scatter(
        x=sample_freqs, y=sample_y,
        mode="lines",
        line=dict(color=COLOR_MEASURED, width=1.6),
        hovertemplate="vibration speed: %{x:.0f}<br>your sample: %{y:.4f}<extra></extra>",
    ))

    # Use a shared y-range so all 4 panels are comparable
    import numpy as np
    yours_max = float(np.max(sample_y)) if len(sample_y) else 1.0
    sig_max   = float(np.max(sig["mag_y"])) if sig is not None and len(sig["mag_y"]) else 0.0
    y_max     = max(yours_max, sig_max) * 1.10
    if y_max <= 0:
        y_max = 1.0

    fig.update_layout(**_base_layout(
        height=130,
        xaxis=_dark_axis(
            range=[0, freq["display_max_hz"]],
            showticklabels=True,
            tickfont=dict(size=9),
        ),
        yaxis=_dark_axis(
            range=[0, y_max],
            showticklabels=False,
        ),
        margin=dict(l=10, r=10, t=10, b=28),
    ))
    return fig


# ===========================================================================
# 11. Orbit plot — Nyquist-style X vs Y vibration scatter
# ===========================================================================
def orbit_plot(signal) -> go.Figure:
    """Shaft orbit plot: X-axis vibration vs Y-axis vibration.

    The shape of the orbit is diagnostic for vibration analysis:
      - Healthy / pure unbalance: roughly circular orbit
      - Misalignment:             elliptical or figure-8 orbit
      - Looseness:                chaotic, ragged orbit

    Args:
        signal: (3, N) numpy array with X/Y/Z channels
    """
    import numpy as np
    # Subsample to ~2000 points for plotting performance
    n_total = signal.shape[1]
    n_plot  = min(2000, n_total)
    idx = np.linspace(0, n_total - 1, n_plot, dtype=int)
    x = signal[0, idx]
    y = signal[1, idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=COLOR_MEASURED, width=1.2),
        opacity=0.85,
        hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
        name="shaft orbit",
    ))

    max_abs = float(max(abs(x).max(), abs(y).max())) * 1.10
    if max_abs <= 0:
        max_abs = 1.0

    fig.update_layout(**_base_layout(
        height=380,
        xaxis=_dark_axis(
            title="X axis vibration (g)",
            range=[-max_abs, max_abs],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.10)",
        ),
        yaxis=_dark_axis(
            title="Y axis vibration (g)",
            range=[-max_abs, max_abs],
            zeroline=True, zerolinecolor="rgba(255,255,255,0.10)",
            scaleanchor="x", scaleratio=1,
        ),
        margin=dict(l=55, r=20, t=12, b=45),
    ))
    return fig


# ===========================================================================
# 12. Bode-style spectrum — log-log magnitude plot
# ===========================================================================
def bode_style_spectrum(freq: dict) -> go.Figure:
    """Log-log frequency response plot — the vibration-analysis equivalent
    of a Bode magnitude plot.

    Uses the same FFT data as `main_spectrum_with_baseline`, but renders
    both axes on logarithmic scales which is the standard format for
    frequency-response analysis in industrial vibration monitoring.
    """
    import numpy as np
    freqs = freq["freqs_hz"]
    mag_y = freq["magnitude_y"]

    # Clip to log-friendly range
    f_min = 1.0
    mask  = freqs >= f_min
    f_plot = freqs[mask]
    m_plot = np.maximum(mag_y[mask], 1e-4)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_plot, y=m_plot, mode="lines",
        line=dict(color=COLOR_MEASURED, width=1.5),
        fill="tozeroy", fillcolor=FILL_MEASURED,
        hovertemplate="frequency: %{x:.1f} Hz<br>magnitude: %{y:.4f}<extra></extra>",
    ))

    # Mark the diagnostic harmonic if present
    markers = freq.get("markers", {})
    for label, m in markers.items():
        f_hz = m.get("freq_hz")
        if f_hz is None or f_hz < f_min:
            continue
        is_diag = m.get("is_diagnostic", False)
        line_color = COLOR_PRED if is_diag else "rgba(255,255,255,0.18)"
        line_width = 1.8 if is_diag else 1.0
        fig.add_vline(
            x=f_hz, line_color=line_color, line_width=line_width,
            line_dash="dash",
            annotation_text=f"{label} ({f_hz:.0f} Hz)",
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color=line_color,
        )

    fig.update_layout(**_base_layout(
        height=260,
        xaxis=_dark_axis(
            title="Frequency (Hz, log scale)",
            type="log", range=[0, np.log10(freq["display_max_hz"])],
        ),
        yaxis=_dark_axis(
            title="Magnitude (log scale)",
            type="log",
        ),
        margin=dict(l=60, r=20, t=12, b=45),
    ))
    return fig


# ===========================================================================
# 13. Annotated diagnostic spectrum (replaces pattern match panels)
# ===========================================================================
def annotated_diagnostic_spectrum(freq: dict, predicted_fault: str) -> go.Figure:
    """Single clean FFT spectrum with shaft harmonic markers labeled.

    The diagnostic harmonic for the predicted fault is highlighted in accent
    color; other harmonics are shown as muted reference lines.

      Fault            Diagnostic harmonic
      ─────────────────────────────────────
      Unbalance        1× rotation
      Misalignment     2× rotation
      Looseness        3× rotation
      Normal           none (no peaks expected)
    """
    freqs = freq["freqs_hz"]
    mag_y = freq["magnitude_y"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=mag_y, mode="lines",
        line=dict(color=COLOR_X, width=1.6),
        fill="tozeroy", fillcolor=FILL_X,
        hovertemplate="frequency: %{x:.1f} Hz<br>amplitude: %{y:.4f}<extra></extra>",
        name="spectrum",
    ))

    # Mark the harmonic markers — diagnostic one highlighted
    markers = freq.get("markers", {})
    for label, m in markers.items():
        f_hz = m.get("freq_hz")
        if f_hz is None:
            continue
        is_diag = m.get("is_diagnostic", False)
        line_color = COLOR_PRED if is_diag else "rgba(255,255,255,0.25)"
        line_width = 2.0 if is_diag else 1.0
        ann_color = COLOR_PRED if is_diag else "rgba(255,255,255,0.55)"
        ann_text = (
            f"<b>{label}</b> · {f_hz:.0f} Hz"
            if is_diag else f"{label} · {f_hz:.0f} Hz"
        )
        fig.add_vline(
            x=f_hz, line_color=line_color, line_width=line_width,
            line_dash="dash",
            annotation_text=ann_text,
            annotation_position="top",
            annotation_font_size=10 if is_diag else 9,
            annotation_font_color=ann_color,
        )

    fig.update_layout(**_base_layout(
        height=320,
        xaxis=_dark_axis(
            title="Frequency (Hz)",
            range=[0, freq.get("display_max_hz", 200)],
        ),
        yaxis=_dark_axis(
            title="Amplitude (g)",
        ),
        margin=dict(l=55, r=20, t=28, b=45),
    ))
    return fig