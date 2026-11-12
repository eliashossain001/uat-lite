"""
Generate two separate, polished Plotly interactive diagrams:
  1. accuracy_slopegraph.html  — AccOrig vs AccNeg slopegraph
  2. stability_signals.html    — Diverging bar chart for Δ signals
"""

import plotly.graph_objects as go

# ── DATA ──────────────────────────────────────────────────────────────────────
methods = ["Base", "UAT-LITE", "UAT-LITE+TS"]
acc_orig  = [0.705, 0.690, 0.690]
acc_neg   = [0.495, 0.495, 0.495]
d_conf    = [-0.0831, -0.0796, -0.0766]
d_margin  = [-0.1422, -0.1316, -0.1224]
d_entropy = [+0.1524, +0.1387, +0.1172]
d_err     = [+0.2100, +0.1950, +0.1950]

COLORS = ["#4C72B0", "#C44E52", "#DD8452"]   # muted blue, red, orange
BG     = "#FAFAFA"
PAPER  = "#FFFFFF"
FONT   = "Inter, Arial, sans-serif"

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Accuracy Slopegraph
# ════════════════════════════════════════════════════════════════════════════
fig1 = go.Figure()

# Background band to show the "drop zone"
fig1.add_shape(
    type="rect",
    x0=-0.3, x1=1.3, y0=0.48, y1=0.51,
    fillcolor="rgba(200,200,200,0.15)",
    line_width=0,
    layer="below",
)

for i, (m, c) in enumerate(zip(methods, COLORS)):
    dash = ["solid", "dash", "dot"][i]
    sym  = ["circle", "square", "triangle-up"][i]

    # Line + markers
    fig1.add_trace(go.Scatter(
        x=[0, 1],
        y=[acc_orig[i], acc_neg[i]],
        mode="lines+markers",
        name=m,
        line=dict(color=c, width=3, dash=dash),
        marker=dict(size=12, color=c, symbol=sym,
                    line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>{m}</b><br>"
            "Condition: %{text}<br>"
            "Accuracy: <b>%{y:.3f}</b><extra></extra>"
        ),
        text=["Original", "Negated"],
        legendgroup=m,
    ))

    # Left-side value label
    v_off = [0.000, 0.007, -0.007][i]
    fig1.add_annotation(
        x=-0.05, y=acc_orig[i] + v_off,
        text=f"<b>{acc_orig[i]:.3f}</b>",
        showarrow=False,
        font=dict(color=c, size=13, family=FONT),
        xanchor="right",
    )

# Right-side convergence label
fig1.add_annotation(
    x=1.05, y=0.495,
    text="<b>0.495</b>",
    showarrow=False,
    font=dict(color="#555", size=13, family=FONT),
    xanchor="left",
)
fig1.add_annotation(
    x=1.05, y=0.522,
    text="All converge →",
    showarrow=False,
    font=dict(color="#999", size=11, family=FONT, style="italic"),
    xanchor="left",
)

# Delta labels on the lines (midpoint)
for i, (m, c) in enumerate(zip(methods, COLORS)):
    drop = acc_neg[i] - acc_orig[i]
    v_off = [0.012, 0.000, -0.012][i]
    fig1.add_annotation(
        x=0.5,
        y=(acc_orig[i] + acc_neg[i]) / 2 + v_off,
        text=f"{drop:+.3f}",
        showarrow=False,
        font=dict(color=c, size=10, family=FONT),
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=2,
    )

fig1.update_layout(
    title=dict(
        text="Accuracy Under Negation: AccOrig → AccNeg",
        font=dict(size=20, family=FONT, color="#222"),
        x=0.5, xanchor="center",
        pad=dict(b=20),
    ),
    xaxis=dict(
        tickvals=[0, 1],
        ticktext=["Original", "Negated"],
        tickfont=dict(size=15, family=FONT, color="#333"),
        range=[-0.4, 1.6],
        showgrid=False,
        showline=True,
        linecolor="#ccc",
        linewidth=1.5,
        zeroline=False,
        fixedrange=True,
    ),
    yaxis=dict(
        title=dict(text="Accuracy", font=dict(size=14, family=FONT)),
        range=[0.44, 0.74],
        tickformat=".2f",
        tickfont=dict(size=12, family=FONT),
        gridcolor="rgba(0,0,0,0.07)",
        gridwidth=1,
        showline=False,
        zeroline=False,
    ),
    plot_bgcolor=BG,
    paper_bgcolor=PAPER,
    height=480,
    width=700,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=13, family=FONT),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ddd", borderwidth=1,
    ),
    margin=dict(l=80, r=100, t=90, b=60),
    hovermode="x unified",
)

fig1.write_html(
    "results/figures/negation_accuracy_slopegraph.html",
    include_plotlyjs="cdn",
    config={"displayModeBar": False},
)
print("saved results/figures/negation_accuracy_slopegraph.html")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stability Signals Diverging Bar Chart
# ════════════════════════════════════════════════════════════════════════════
fig2 = go.Figure()

signals     = ["ΔConf", "ΔMargin", "ΔEntropy", "ΔErr (see caption)"]
all_vals    = [d_conf, d_margin, d_entropy, d_err]
is_deemph   = [False, False, False, True]
OFFSET      = 0.28          # y-spacing between signal groups
BAR_THICK   = 0.10

for i, (m, c) in enumerate(zip(methods, COLORS)):
    pattern = ["", "/", "."][i]
    off = (i - 1) * BAR_THICK * 1.15   # centre group around 0

    for j, (sig, vals, deemph) in enumerate(zip(signals, all_vals, is_deemph)):
        v      = vals[i]
        alpha  = 0.22 if deemph else 0.88
        y_pos  = j * OFFSET + off

        fig2.add_trace(go.Bar(
            x=[v],
            y=[y_pos],
            orientation="h",
            width=BAR_THICK * 0.92,
            name=m,
            legendgroup=m,
            showlegend=(j == 0),
            marker=dict(
                color=f"rgba({','.join(str(int(c.lstrip('#')[k:k+2], 16)) for k in (0,2,4))},{alpha})",
                line=dict(color=c, width=1.2 if not deemph else 0.4),
                pattern_shape=pattern,
                pattern_fillmode="overlay",
                pattern_fgcolor=c,
                pattern_fgopacity=0.5,
            ),
            hovertemplate=(
                f"<b>{m}</b><br>"
                f"{sig}: <b>%{{x:+.4f}}</b><extra></extra>"
            ),
        ))

        # Inline value label (not for de-emph)
        if not deemph:
            ha  = "left"  if v >= 0 else "right"
            off2 = 0.003  if v >= 0 else -0.003
            fig2.add_annotation(
                x=v + off2, y=y_pos,
                text=f"{v:+.4f}",
                showarrow=False,
                font=dict(size=10, color=c, family=FONT),
                xanchor=ha,
                yanchor="middle",
            )

# Signal group y-tick positions and labels
ytick_vals = [j * OFFSET for j in range(len(signals))]

# Zero line
fig2.add_shape(
    type="line",
    x0=0, x1=0,
    y0=-OFFSET * 0.5,
    y1=(len(signals) - 1) * OFFSET + OFFSET * 0.5,
    line=dict(color="#444", width=1.5, dash="solid"),
    layer="above",
)

# Direction arrows / labels
fig2.add_annotation(
    x=-0.005, y=(len(signals) - 1) * OFFSET + OFFSET * 0.55,
    text="◀  less uncertain / more confident",
    showarrow=False,
    font=dict(size=10.5, color="#888", family=FONT, style="italic"),
    xanchor="right",
)
fig2.add_annotation(
    x=0.005, y=(len(signals) - 1) * OFFSET + OFFSET * 0.55,
    text="more uncertain / less confident  ▶",
    showarrow=False,
    font=dict(size=10.5, color="#888", family=FONT, style="italic"),
    xanchor="left",
)

# Callout: UAT-LITE+TS has smallest ΔEntropy
best_entropy_y = 2 * OFFSET + (2 - 1) * BAR_THICK * 1.15   # UAT-LITE+TS row
fig2.add_annotation(
    x=d_entropy[2] + 0.008,
    y=best_entropy_y + 0.04,
    text="<b>UAT-LITE+TS</b>: smallest<br>uncertainty spike ✓",
    showarrow=True,
    arrowhead=2, arrowcolor="#555", arrowwidth=1.5, ax=70, ay=-40,
    font=dict(size=11, color="#333", family=FONT),
    bgcolor="rgba(255,248,220,0.92)",
    bordercolor="#cca",
    borderwidth=1,
    borderpad=5,
)

# Shade background for ΔErr group to further de-emphasize
fig2.add_shape(
    type="rect",
    x0=-0.26, x1=0.26,
    y0=3 * OFFSET - OFFSET * 0.45,
    y1=3 * OFFSET + OFFSET * 0.45,
    fillcolor="rgba(220,220,220,0.18)",
    line_width=0,
    layer="below",
)

fig2.update_layout(
    title=dict(
        text="Label-Free Stability Signals  (Δ = negated − original)",
        font=dict(size=20, family=FONT, color="#222"),
        x=0.5, xanchor="center",
        pad=dict(b=20),
    ),
    xaxis=dict(
        title=dict(text="Δ value", font=dict(size=14, family=FONT)),
        range=[-0.26, 0.30],
        tickformat="+.2f",
        tickfont=dict(size=12, family=FONT),
        gridcolor="rgba(0,0,0,0.07)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#ccc",
    ),
    yaxis=dict(
        tickvals=ytick_vals,
        ticktext=[f"<b>{s}</b>" for s in signals],
        tickfont=dict(size=13, family=FONT),
        showgrid=False,
        showline=False,
        zeroline=False,
        fixedrange=True,
    ),
    barmode="overlay",
    plot_bgcolor=BG,
    paper_bgcolor=PAPER,
    height=560,
    width=750,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=13, family=FONT),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ddd", borderwidth=1,
    ),
    margin=dict(l=160, r=60, t=120, b=70),
    hovermode="closest",
)

fig2.write_html(
    "results/figures/negation_stability_signals.html",
    include_plotlyjs="cdn",
    config={"displayModeBar": False},
)
print("saved results/figures/negation_stability_signals.html")