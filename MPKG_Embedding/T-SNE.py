#-*-coding -utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
import json
import os
import yaml
from yaml.loader import Loader

from matplotlib.colors import ListedColormap, BoundaryNorm

# --model_folder TransE --entities_path data/entities.dict --fig_folder Fig

# Example: Load your pre-trained entity embeddings
# Replace this with your actual embeddings
# Assume 'entity_embeddings' is a NumPy array of shape (num_entities, embedding_dim)

def read_json(dir):
    # 打开JSON文件并读取内容
    with open(dir, 'r', encoding='utf-8') as json_file:
        file_content = json_file.read()

    # 解析JSON字符串为字典
    data_dict = json.loads(file_content)
    return data_dict

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default=None,help="The model folder when training finish")
parser.add_argument("--entities_path", default=None)
parser.add_argument("--fig_folder", default=None)
parser.add_argument("-lr", "--learning_rate", default=10, type=float, help="The learning rate controls the step size of the t-SNE algorithm during the optimization process.")
parser.add_argument("--dim", default=2, type=int, help="Specify the target dimension to which t-SNE reduces the data.")
parser.add_argument("--perplexity", default=30, type=float, help="A complexity measure of t-SNE when considering the local neighbors of each data point.")
arg = parser.parse_args()

if not os.path.exists(arg.model_folder):
    raise ValueError("The model training folder is necessary.")
elif not os.path.exists(os.path.join(arg.model_folder,"entity_embedding.npy")):
    raise ValueError("The entity embedding results are lacking.")
if os.path.exists(arg.entities_path):
    entities = read_json(arg.entities_path)
else:
    raise ValueError("Entity files need to be provided.")

entity_embeddings_path = os.path.join(arg.model_folder,"entity_embedding.npy")
entity_embeddings = np.load(entity_embeddings_path)

config_path = os.path.join(arg.model_folder,"config.yml")
yaml_file = open(config_path).read()
model_config = yaml.load(yaml_file,Loader=Loader)
entity_embedding_dim = model_config["EMBEDDING_SETTINGS"]["hidden_dim"]
model_name = model_config["GLOBAL_CONFIG"]["MODEL"]

# Apply t-SNE for 2D projection
tsne = TSNE(n_components=arg.dim, perplexity=arg.perplexity, learning_rate=arg.learning_rate, random_state=42)
entity_embeddings_tsne = tsne.fit_transform(entity_embeddings)

labels = []
entities_label = []
for key, item in entities.items():
    item_label = item["labels"]
    if item_label not in labels:
        labels.append(item_label)
    entities_label.append(labels.index(item_label))

# Plot the t-SNE graph
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 8))
cmap = get_cmap("viridis")  # Use 'viridis' colormap

# Scatter plot with labels as color
scatter = plt.scatter(
    entity_embeddings_tsne[:, 0],
    entity_embeddings_tsne[:, 1],
    c=entities_label,
    cmap="rainbow",
    # cmap=cmap,
    s=50,
    alpha=0.7
)
# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Entities Labels', fontsize=12)
cbar.set_ticks(np.arange(len(labels)))
cbar.set_ticklabels(labels)
cbar.ax.tick_params(axis='y', labelrotation=-45, labelsize=10)

# Title and axis labels
fig_name = '{} {}D Visualization - {} lr'.format(model_name, arg.dim, arg.learning_rate)
# plt.title(fig_name, fontsize=16)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True)

# Show plot
png_name = '{}_{}D_Visualization_{}_lr'.format(model_name, arg.dim, arg.learning_rate)
plt.tight_layout()
plt.savefig(os.path.join(arg.fig_folder,png_name + ".png"),dpi=300)
# plt.show()



# Define a discrete colormap
'''

colors = ['red', 'green', 'blue', 'yellow', 'black', 'white', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray']
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries=np.arange(-0.5, len(colors)+0.5, 1), ncolors=len(colors))

# Plot the t-SNE graph
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    entity_embeddings_tsne[:, 0],
    entity_embeddings_tsne[:, 1],
    c=entities_label,
    cmap=cmap,
    s=50,
    alpha=0.7
)

# Add a discrete color bar
cbar = plt.colorbar(scatter, ticks=np.arange(len(colors)))
cbar.set_label('Cluster Labels', fontsize=12)
cbar.ax.set_yticklabels(labels)

# Title and axis labels
plt.title('GraphSAGE Visualization with Discrete Clusters', fontsize=16)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
'''
# Create a 3D scatter plot
"""
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with labels as color
scatter = ax.scatter(
    entity_embeddings_tsne[:, 0],
    entity_embeddings_tsne[:, 1],
    entity_embeddings_tsne[:, 2],
    c=entities_label,
    cmap='viridis',
    s=50,
    alpha=0.7
)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Entity Labels', fontsize=12)

# Add labels and title
ax.set_title('3D t-SNE Visualization of Knowledge Graph Entities', fontsize=16)
ax.set_xlabel('Dimension 1', fontsize=12)
ax.set_ylabel('Dimension 2', fontsize=12)
ax.set_zlabel('Dimension 3', fontsize=12)

# Show plot
plt.tight_layout()
plt.show()"""
# Create a 3D scatter plot grouped by clusters
"""
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# Assign a color to each cluster
unique_labels = np.unique(entities_label)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Use 'tab10' colormap

for label, color in zip(entities_label, colors):
    cluster_points =                                                                                                                   [entities_label == label]
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        label=f"Cluster {label}",
        color=color,
        s=50,
        alpha=0.7
    )

# Add legend
ax.legend(title="Clusters", fontsize=10, loc="best")

# Add labels and title
ax.set_title('3D t-SNE Visualization of Clusters', fontsize=16)
ax.set_xlabel('Dimension 1', fontsize=12)
ax.set_ylabel('Dimension 2', fontsize=12)
ax.set_zlabel('Dimension 3', fontsize=12)

# Show plot
plt.tight_layout()
plt.show()

"""