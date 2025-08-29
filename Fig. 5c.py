# chord_diagram.py

import numpy as np
import pandas as pd
import openchord as ocd

# Prepare the dataset (insert data)
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

# Remove 'Group' column to retain only numeric features
numeric_df = df.drop(columns=['Group'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Calculate the absolute value of the correlation matrix
absolute_correlation_matrix = correlation_matrix.abs()

# Filter correlations less than 0.5
filtered_correlation_matrix = absolute_correlation_matrix.where(absolute_correlation_matrix > 0.5, 0)

# Square the filtered correlation values
squared_correlation_matrix = filtered_correlation_matrix ** 2

# Retain diagonal values as 1
np.fill_diagonal(squared_correlation_matrix.values, 1)

# Prepare matrix for Chord diagram
ff_matrix = squared_correlation_matrix.where(squared_correlation_matrix > 0.5, 0)
np.fill_diagonal(ff_matrix.values, 0)

# Ensure isolated nodes are visible
for i in range(ff_matrix.shape[0]):
    if np.all(ff_matrix.iloc[i, :] == 0):
        ff_matrix.iloc[i, i] = 0.3

# Define labels for the Chord diagram
labels = [
    "OFT (Dis)", "OFT (Cen)", "TST(Imm)", "FST(Imm)",
    "Pre", "Nov", "Tot", "Mush",
    "Thin", "Stub", "C_IL6", "H_IL6", "C_TNFa", "H_TNFa", "mBDNF",
    "Cort", "5-HT", "HP(D)", "HP(T)",
    "HP(A)", "HP(B)", "HP(G)", "HH(D)",
    "HH(T)", "HH(A)", "HH(B)", "HH(G)", "Coh_T", "Coh_G"
]

# Save correlation matrices to an Excel file
output_path = r""
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    correlation_matrix.to_excel(writer, sheet_name='Correlation')
    squared_correlation_matrix.to_excel(writer, sheet_name='Squared_Correlation')
    ff_matrix.to_excel(writer, sheet_name='Final_For_Visualization')

print(f"Correlation matrices saved to '{output_path}'")

# Create and show Chord diagram
fig = ocd.Chord(ff_matrix, labels)
fig.show()
fig.save_svg("figure_diagonal_as_circle.svg")
