# tsne_plot.py

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Prepare the dataset
data = {
    'Group': ['sham'] * 10 + ['dep'] * 10 + ['TES'] * 10,
    'OFT_Total_Distance': [],
    'OFT_Center_Zone': [],
    'TST_Immobile': [],
    'FST_Immobile': [],
    'Social_Preference': [],
    'Social_Novelty': [],
    'Spine_Total': [],
    'Spine_Mushroom': [],
    'Spine_Thin': [],
    'Spine_Stubby': [],
    'Cor_IL_6': [],
    'HPC_IL_6': [],
    'Cor_TNF_a': [],
    'HPC_TNF_a': [],
    'Mature_BDNF': [],
    'Corticosterone': [],
    'Serotonin': [],
    'HPC_PFC_delta': [],
    'HPC_PFC_theta': [],
    'HPC_PFC_alpha': [],
    'HPC_PFC_beta': [],
    'HPC_PFC_gamma': [],
    'HPC_HPC_delta': [],
    'HPC_HPC_theta': [],
    'HPC_HPC_alpha': [],
    'HPC_HPC_beta': [],
    'HPC_HPC_gamma': [],
    'Theta_coherence': [],
    'Gamma_coherence': []
    }
df = pd.DataFrame(data)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X = df.drop(columns=['Group'])  # Exclude 'Group' column
X_embedded = tsne.fit_transform(X)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(X_embedded, columns=['Dimension 1', 'Dimension 2'])
tsne_df['Group'] = df['Group']

# Calculate group means and covariances
group_means = tsne_df.groupby('Group')[['Dimension 1', 'Dimension 2']].mean()
group_covs = tsne_df.groupby('Group')[['Dimension 1', 'Dimension 2']].cov()

factor = 9  # Scaling factor for ellipse size

# Plot group ellipses
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'sham': 'green', 'dep': 'hotpink', 'TES': 'orange'}

for group in ['sham', 'dep', 'TES']:
    mean = group_means.loc[group].values
    cov = group_covs.loc[group].values.reshape(2, 2)

    cov_scaled = cov * factor
    eigenvalues, eigenvectors = np.linalg.eigh(cov_scaled)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        mean, width, height, angle=angle,
        edgecolor=colors[group], facecolor=colors[group], alpha=0.3
    )
    ax.add_patch(ellipse)

# Scatter plot of t-SNE results
ax.scatter(
    tsne_df['Dimension 1'], tsne_df['Dimension 2'],
    c=tsne_df['Group'].map(colors), alpha=0.6
)

ax.set_aspect(0.5, adjustable='datalim')
ax.axis('off')
plt.tight_layout()
plt.show()
