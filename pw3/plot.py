import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# ===================== PART 1 =====================
d1 = pd.read_csv("pw3/part1_recovery.csv")

fig1 = go.Figure()

# True function
fig1.add_trace(go.Scatter(
    x=d1.x, y=d1.y_clean,
    mode='lines',
    name='True function',
    line=dict(color='black', width=2)
))

# Noisy observations
fig1.add_trace(go.Scatter(
    x=d1.x, y=d1.y_noisy,
    mode='markers',
    name='Observations (noisy)',
    marker=dict(color='royalblue', size=3, opacity=0.6)
))

# Regression estimate
fig1.add_trace(go.Scatter(
    x=d1.x, y=d1.y_pred,
    mode='lines',
    name='Regression estimate',
    line=dict(color='red', width=2)
))

fig1.update_layout(
    title='Part 1: Regression recovery (Nadaraya–Watson, parabolic kernel)',
    xaxis_title='x',
    yaxis_title='y',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
)
fig1.write_html("part1_plot.html")
fig1.show()

# ===================== PART 2 =====================
d2 = pd.read_csv("pw3/part2_recovery.csv")

fig2 = go.Figure()

# Listings (scatter)
fig2.add_trace(go.Scatter(
    x=d2.x, y=d2.y,
    mode='markers',
    name='Listings',
    marker=dict(color='royalblue', size=8, opacity=0.8,
                line=dict(width=1, color='darkblue'))
))

# Regression estimate
fig2.add_trace(go.Scatter(
    x=d2.x, y=d2.y_pred,
    mode='lines',
    name='Regression estimate',
    line=dict(color='red', width=2)
))

fig2.update_layout(
    title='Part 2: Regression recovery on real data',
    xaxis_title='x (feature)',
    yaxis_title='y (price)',
    template='plotly_white',
    hovermode='closest',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
)
fig2.write_html("part2_plot.html")
fig2.show()

# ===================== BONUS: side-by-side =====================
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Part 1: synthetic data', 'Part 2: real data')
)
for tr in fig1.data:
    fig.add_trace(tr, row=1, col=1)
for tr in fig2.data:
    fig.add_trace(tr, row=1, col=2)
fig.update_layout(
    title='Nadaraya–Watson regression: both parts',
    template='plotly_white',
    height=500
)
fig.write_html("regression_both.html")
fig.show()