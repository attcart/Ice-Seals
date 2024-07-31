# Import necessary libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

# Load the data w/ file path here
file_path = r'C:\Users\Attca\Downloads\results (1).csv'
data = pd.read_csv(file_path)

# Strip spaces from column names
data.columns = data.columns.str.strip()

# Print the column names to verify
print("Column names:", data.columns)

# Define thresholds
very_good_thresholds = {
    'train/box_loss': 1.0,
    'train/cls_loss': 1.0,
    'train/dfl_loss': 0.5,
    'metrics/precision': 0.9,
    'metrics/recall': 0.9,
    'metrics/mAP_0.5': 0.9,
    'metrics/mAP_0.5:0.95': 0.7,
    'val/box_loss': 1.0,
    'val/cls_loss': 1.0,
    'val/dfl_loss': 0.5
}

# Create subplots
fig = sp.make_subplots(rows=5, cols=2, subplot_titles=(
    'Train Box Loss', 'Train Class Loss',
    'Train DFL Loss', 'Metrics Precision',
    'Metrics Recall', 'Metrics mAP_0.5',
    'Metrics mAP_0.5:0.95', 'Val Box Loss',
    'Val Class Loss', 'Val DFL Loss'
))

# Add traces for each metric
metrics = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
    'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/cls_loss',
    'val/dfl_loss'
]

for i, metric in enumerate(metrics):
    row = i // 2 + 1
    col = i % 2 + 1
    fig.add_trace(go.Scatter(x=data['epoch'], y=data[metric], mode='lines+markers', name=metric), row=row, col=col)
    fig.add_trace(go.Scatter(x=data['epoch'], y=[very_good_thresholds[metric]]*len(data), mode='lines', name=f'Very Good ({metric})', line=dict(dash='dash')), row=row, col=col)

# Update layout
fig.update_layout(height=1500, width=1000, title_text="DataDashboard workflow")
fig.show()
