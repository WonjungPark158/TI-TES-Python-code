import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Read the Excel file
file_path = ""  # Insert the file path here
df = pd.read_excel(file_path)

# Rename the first unnamed column to 'label'
df = df.rename(columns={'Unnamed: 0': 'label'})

# Filter the data to include only 'sham' and 'dep' groups
filtered_df = df[df['label'].isin(['sham', 'dep'])].copy()

# Encode the labels into numeric values
filtered_df['label_numeric'] = filtered_df['label'].astype('category').cat.codes

# Select feature columns
features = ['Behavioral', 'Biological', 'Neural']

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(filtered_df[features])
y = filtered_df['label_numeric'].values

# Train an SVM model with a linear kernel
svm_model = SVC(kernel='linear', C=1000)
svm_model.fit(X_std, y)

# Extract the weights and bias of the SVM hyperplane
w = svm_model.coef_[0]
b = svm_model.intercept_[0]

# Generate a grid for plotting the hyperplane
grid_vals = np.linspace(0, 1, 50)
xx, yy = np.meshgrid(grid_vals, grid_vals)

# Create 2D grid points and standardize them
flat_xy = np.c_[xx.ravel(), yy.ravel()]
flat_xy_std = scaler.transform(np.c_[flat_xy, np.zeros_like(xx.ravel())])[:, :2]

# Calculate corresponding z values for the hyperplane in standardized space
zz_std = (-w[0] * flat_xy_std[:, 0] - w[1] * flat_xy_std[:, 1] - b) / w[2]

# Combine x, y, z in standardized space
flat_xyz_std = np.c_[flat_xy_std[:, 0], flat_xy_std[:, 1], zz_std]

# Inverse transform to original scale
flat_xyz_orig = scaler.inverse_transform(flat_xyz_std)

# Keep only points within [0, 1] range
mask = np.all((flat_xyz_orig >= 0) & (flat_xyz_orig <= 1), axis=1)
filtered_points = flat_xyz_orig[mask]

# Extract coordinates for plotting
xx_plot = filtered_points[:, 0]
yy_plot = filtered_points[:, 1]
zz_plot = filtered_points[:, 2]

# Inverse transform the original data for plotting
X_orig = scaler.inverse_transform(X_std)

# Plot the 3D scatter plot and hyperplane
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot data points for each label
for label, group in zip(['sham', 'dep'], [0, 1]):
    idx = y == group
    ax.scatter(X_orig[idx, 0], X_orig[idx, 1], X_orig[idx, 2], label=label)

# Plot the hyperplane
ax.scatter(xx_plot, yy_plot, zz_plot, alpha=0.3, color='black', label='SVM Plane', s=1)

# Set axis labels and title
ax.set_xlabel('Behavioral')
ax.set_ylabel('Biological')
ax.set_zlabel('Neural')
ax.set_title('SVM Hyperplane (clipped to [0, 1])')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Save the clipped hyperplane points to an Excel file
df_plane_clipped = pd.DataFrame({
    'Behavioral (X)': xx_plot,
    'Biological (Y)': yy_plot,
    'Neural (Z)': zz_plot
})

output_path = ""  # Specify output file path here
df_plane_clipped.to_excel(output_path, index=False)
print("Success:", output_path)
