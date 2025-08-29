import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Read Excel file (specify the uploaded file)
file_path = ""  # Enter the file name here
df = pd.read_excel(file_path)

# Check the data (display the first few rows)
print(df.head())

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract the correlation coefficients with Depression_Score
depression_corr = correlation_matrix["Depression_Score"].sort_values(ascending=False)

print(depression_corr)

# Automatically extract nodes for the Sankey diagram
all_nodes = list(df.columns)
all_nodes.remove("Depression_Score")  # Exclude Depression_Score

# Group nodes (example: user can modify as needed)
left_nodes = ["Spine_Total", "Spine_Mushroom", "Spine_Thin", "Spine_Stubby", "Cor_IL_6", "HPC_IL_6", "Cor_TNF_a", "HPC_TNF_a", "Mature_BDNF", "Corticosterone", "Serotonin"]
right_nodes = [
    "HPC_PFC_delta", "HPC_PFC_theta", "HPC_PFC_alpha", "HPC_PFC_beta", "HPC_PFC_gamma",
    "HPC_HPC_delta", "HPC_HPC_theta", "HPC_HPC_alpha", "HPC_HPC_beta", "HPC_HPC_gamma",
    "Theta_coherence", "Gamma_coherence"
]
center_node = ["Depression_Score"]
nodes = left_nodes + center_node + right_nodes

# Define links
links = []
for source in left_nodes:
    if source in depression_corr:
        corr_value = depression_corr[source]
        links.append({
            "source": nodes.index(source),
            "target": nodes.index("Depression_Score"),
            "value": abs(corr_value),  # Use the absolute value of the correlation coefficient
            'color': "grey"
        })

for source in right_nodes:
    if source in depression_corr:
        corr_value = depression_corr[source]
        links.append({
            "source": nodes.index("Depression_Score"),
            "target": nodes.index(source),
            "value": abs(corr_value),  # Use the absolute value of the correlation coefficient
            'color': "grey"
        })

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
    ),
    link=dict(
        source=[link["source"] for link in links],
        target=[link["target"] for link in links],
        value=[link["value"] for link in links],
        color=[link["color"] for link in links]
    )
))

# Update layout
fig.update_layout(
    title_text="Sankey Diagram",
    font_size=10,
    height=800  # Increase the height
)
fig.show()
