import os
import glob
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import plotly
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Point directly to your train folder structure from your folder screenshot
BASE_DIR = "./data/grasp-and-lift-eeg-detection"
DATA_DIR = os.path.join(BASE_DIR, "train","train")

# Event columns standard to WAY-GAL
EVENT_COLS = [
    'HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 
    'LiftOff', 'Replace', 'BothReleased'
]

def parse_filename_info(filepath):
    """Extracts Subject and Series ID from file names (e.g., subj1_series1_data.csv)."""
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    # Handles both 'sub1' and 'subj1' naming formats
    sub_id = int(parts[0].replace('subj', '').replace('sub', ''))
    series_id = int(parts[1].replace('series', ''))
    return sub_id, series_id

def scan_dataset():
    """Scans DATA_DIR for available subject and series files."""
    data_files = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))
    hierarchy = {}
    for f in data_files:
        try:
            s_id, ser_id = parse_filename_info(f)
            if s_id not in hierarchy:
                hierarchy[s_id] = []
            hierarchy[s_id].append(ser_id)
        except Exception:
            continue
    
    for s_id in hierarchy:
        hierarchy[s_id].sort()
    return dict(sorted(hierarchy.items()))

def load_series_data(subject, series):
    """Loads and merges EEG channel data and event flags for a given subject & series with robust parsing."""
    data_path = os.path.join(DATA_DIR, f"subj{subject}_series{series}_data.csv")
    events_path = os.path.join(DATA_DIR, f"subj{subject}_series{series}_events.csv")
    
    # Fallback check if files use 'sub' instead of 'subj'
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, f"sub{subject}_series{series}_data.csv")
        events_path = os.path.join(DATA_DIR, f"sub{subject}_series{series}_events.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Could not find data file at {data_path}")
        return None, None, []
        
    try:
        df_data = pd.read_csv(data_path)
        
        # Drop 'id' or unwanted string columns if present
        if 'id' in df_data.columns:
            df_data = df_data.drop(columns=['id'])
            
        # Ensure all EEG columns are coerced to numeric (handles any accidental string/headers corruption)
        eeg_cols = list(df_data.columns)
        for col in eeg_cols:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0.0)
            
        if os.path.exists(events_path):
            df_events = pd.read_csv(events_path)
            if 'id' in df_events.columns:
                df_events = df_events.drop(columns=['id'])
            events_available = [c for c in EVENT_COLS if c in df_events.columns]
            for col in events_available:
                df_events[col] = pd.to_numeric(df_events[col], errors='coerce').fillna(0)
            df_merged = pd.concat([df_data[eeg_cols], df_events[events_available]], axis=1)
        else:
            df_merged = df_data[eeg_cols]
            events_available = []
            
        return df_merged, eeg_cols, events_available
    except Exception as e:
        print(f"Error loading CSV for Subject {subject} Series {series}: {e}")
        return None, None, []

def detect_trials(df, events_available):
    """
    Segments a continuous series into discrete trials by tracking 'HandStart' onset/offsets.
    Returns list of trial dicts with start/end indices.
    """
    if 'HandStart' not in events_available:
        # Fallback split into 1000-frame chunks if event masks aren't found
        n_frames = len(df)
        chunk_size = 1000
        return [{'trial_num': i+1, 'start': i*chunk_size, 'end': min((i+1)*chunk_size, n_frames)} 
                for i in range(int(np.ceil(n_frames/chunk_size)))]
    
    # Identify trial boundaries based on HandStart transition flags
    hs = df['HandStart'].values
    diff = np.diff(np.pad(hs, (1, 1), 'constant', constant_values=0))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    trials = []
    # Add buffer padding around active trial windows (e.g., 250 frames / 0.5s)
    buffer = 250
    for idx, (s, e) in enumerate(zip(starts, ends)):
        padded_start = max(0, s - buffer)
        padded_end = min(len(df), e + buffer)
        trials.append({
            'trial_num': idx + 1,
            'start': int(padded_start),
            'end': int(padded_end),
            'onset': int(s),
            'offset': int(e)
        })
    return trials

@app.route('/')
def index():
    hierarchy = scan_dataset()
    return render_template_string(HTML_TEMPLATE, hierarchy=hierarchy)

@app.route('/api/load_series')
def api_load_series():
    subject = request.args.get('subject', type=int)
    series = request.args.get('series', type=int)
    
    df, eeg_cols, events_available = load_series_data(subject, series)
    if df is None:
        return jsonify({'error': 'File not found'}), 404
        
    trials = detect_trials(df, events_available)
    
    return jsonify({
        'total_frames': len(df),
        'eeg_channels': eeg_cols,
        'events': events_available,
        'trials': trials
    })

@app.route('/api/plot_data')
def api_plot_data():
    subject = request.args.get('subject', type=int)
    series = request.args.get('series', type=int)
    start_idx = request.args.get('start', default=0, type=int)
    end_idx = request.args.get('end', default=2000, type=int)
    channels = request.args.getlist('channels')
    
    df, eeg_cols, events_available = load_series_data(subject, series)
    if df is None:
        return jsonify({'error': 'Data not found'}), 404
        
    # Clamp bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(df), end_idx)
    sub_df = df.iloc[start_idx:end_idx]
    time_axis = np.arange(start_idx, end_idx) / 500.0  # 500 Hz sampling rate

    fig = go.Figure()
    
    # Plot selected EEG channels with vertical offsets (stacked continuous plot)
    selected_chans = channels if channels else eeg_cols[:8]
    spacing = 150  # Offset spacing in µV
    
    for i, ch in enumerate(selected_chans):
        if ch in sub_df.columns:
            offset = i * spacing
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=sub_df[ch] + offset,
                mode='lines',
                name=ch,
                line=dict(width=1.2)
            ))

    # Overlay event activation bands across the timeline
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B', '#B10DC9', '#FFDC00']
    for idx, ev in enumerate(events_available):
        if ev in sub_df.columns:
            ev_active = sub_df[ev].values
            active_indices = np.where(ev_active == 1)[0]
            if len(active_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=time_axis[active_indices],
                    y=[-(idx + 1) * 30] * len(active_indices),
                    mode='markers',
                    marker=dict(symbol='square', size=4, color=colors[idx % len(colors)]),
                    name=f"Event: {ev}"
                ))

    fig.update_layout(
        title=f"Subject {subject} — Series {series} (Frames {start_idx} to {end_idx})",
        xaxis_title="Time (seconds @ 500Hz)",
        yaxis_title="Stacked Channels (µV) & Event Signals",
        height=650,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'graph_json': graph_json})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>WAY-GAL EEG Data Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background-color: #121212; color: #e0e0e0; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 320px; background-color: #1e1e1e; padding: 20px; box-sizing: border-box; overflow-y: auto; border-right: 1px solid #333; }
        .main { flex: 1; padding: 20px; display: flex; flex-direction: column; }
        h2, h4 { color: #40c4ff; margin-top: 0; }
        select, button { width: 100%; padding: 8px; margin-bottom: 12px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px; }
        .channel-list { max-height: 180px; overflow-y: auto; background: #252525; padding: 8px; border-radius: 4px; margin-bottom: 12px; }
        .channel-item { display: block; margin-bottom: 4px; font-size: 13px; }
        .trial-btn { display: inline-block; width: 45%; margin: 2%; padding: 6px; background: #333; text-align: center; cursor: pointer; border-radius: 3px; font-size: 12px; }
        .trial-btn:hover { background: #00838f; }
        #plot-area { flex: 1; min-height: 500px; }
        .breadcrumb { font-size: 14px; background: #252525; padding: 10px; border-radius: 4px; margin-bottom: 10px; color: #b0bec5; }
    </style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <h2>WAY-GAL Explorer</h2>
        
        <label>Subject:</label>
        <select id="subject-select" onchange="onSubjectChange()">
            {% for sub in hierarchy.keys() %}
                <option value="{{ sub }}">Subject {{ sub }}</option>
            {% endfor %}
        </select>
        
        <label>Series:</label>
        <select id="series-select" onchange="loadSeriesMetadata()"></select>

        <div class="breadcrumb" id="series-connection-info">
            Select a series to view sequential continuum.
        </div>

        <h4>EEG Channels</h4>
        <div class="channel-list" id="channel-selector"></div>
        <button onclick="updatePlot()">Update Signals</button>

        <h4>Trials in Series</h4>
        <div id="trials-container"></div>
    </div>
    
    <div class="main">
        <div id="plot-area"></div>
    </div>
</div>

<script>
    const hierarchy = {{ hierarchy | tojson }};

    function onSubjectChange() {
        const sub = document.getElementById('subject-select').value;
        const seriesSelect = document.getElementById('series-select');
        seriesSelect.innerHTML = '';
        
        if (hierarchy[sub]) {
            hierarchy[sub].forEach(ser => {
                const opt = document.createElement('option');
                opt.value = ser;
                opt.textContent = `Series ${ser}`;
                seriesSelect.appendChild(opt);
            });
        }
        loadSeriesMetadata();
    }

    async function loadSeriesMetadata() {
        const sub = document.getElementById('subject-select').value;
        const ser = document.getElementById('series-select').value;
        if (!sub || !ser) return;

        const allSeries = hierarchy[sub];
        const currentIdx = allSeries.indexOf(parseInt(ser));
        const prevSer = currentIdx > 0 ? allSeries[currentIdx - 1] : null;
        const nextSer = currentIdx < allSeries.length - 1 ? allSeries[currentIdx + 1] : null;

        let connText = `<b>Subject ${sub}</b> | Series ${ser}<br>`;
        connText += prevSer ? `← Preceded by Series ${prevSer}<br>` : `• (First Series)<br>`;
        connText += nextSer ? `→ Followed by Series ${nextSer}` : `• (Last Series)`;
        document.getElementById('series-connection-info').innerHTML = connText;

        const res = await fetch(`/api/load_series?subject=${sub}&series=${ser}`);
        const data = await res.json();

        const chanContainer = document.getElementById('channel-selector');
        chanContainer.innerHTML = '';
        data.eeg_channels.forEach((ch, idx) => {
            const label = document.createElement('label');
            label.className = 'channel-item';
            label.innerHTML = `<input type="checkbox" value="${ch}" ${idx < 8 ? 'checked' : ''}> ${ch}`;
            chanContainer.appendChild(label);
        });

        const trialContainer = document.getElementById('trials-container');
        trialContainer.innerHTML = '';
        data.trials.forEach(t => {
            const btn = document.createElement('div');
            btn.className = 'trial-btn';
            btn.textContent = `Trial ${t.trial_num}`;
            btn.onclick = () => renderRange(t.start, t.end);
            trialContainer.appendChild(btn);
        });

        if (data.trials.length > 0) {
            renderRange(data.trials[0].start, data.trials[0].end);
        } else {
            renderRange(0, Math.min(2000, data.total_frames));
        }
    }

    function getSelectedChannels() {
        const checkboxes = document.querySelectorAll('#channel-selector input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    let currentStart = 0;
    let currentEnd = 2000;

    function updatePlot() {
        renderRange(currentStart, currentEnd);
    }

    async function renderRange(start, end) {
        currentStart = start;
        currentEnd = end;
        const sub = document.getElementById('subject-select').value;
        const ser = document.getElementById('series-select').value;
        const channels = getSelectedChannels();

        let query = `/api/plot_data?subject=${sub}&series=${ser}&start=${start}&end=${end}`;
        channels.forEach(ch => query += `&channels=${encodeURIComponent(ch)}`);

        const res = await fetch(query);
        const data = await res.json();
        const fig = JSON.parse(data.graph_json);
        Plotly.newPlot('plot-area', fig.data, fig.layout, {responsive: true});
    }

    window.onload = () => { onSubjectChange(); };
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)