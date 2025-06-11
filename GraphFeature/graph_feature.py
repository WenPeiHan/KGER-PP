#-*-coding -utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GAE, GCNConv, global_mean_pool, GATConv
from torch_sparse import SparseTensor
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import json
from tqdm import tqdm
import random
import pickle

# Define the Graph Attention Network (GAT) model
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_dim=256, heads=1):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # List to store the layers
        self.gat_layers = nn.ModuleList()

        # Input layer
        self.gat_layers.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=0.6))

        # Hidden layers (num_layers - 2, as one layer is for input, one is for output)
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6))

        # Output layer
        self.gat_layers.append(GATConv(hidden_dim * heads, out_channels, heads=1, dropout=0.6))

        self.reset_parameters()

    def forward(self, x, edge_index):
        # Pass through each layer
        for i in range(self.num_layers - 1):
            x = F.elu(self.gat_layers[i](x, edge_index))

        # Last layer (no activation function)
        x = self.gat_layers[-1](x, edge_index)
        return x

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()


# Define the contrastive loss with negative sampling
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, z_neg, cross):
        # Cosine similarity between positive pairs (z_i, z_j) and negative pair (z_neg)
        batch_size = 1
        query = z_i.unsqueeze(0)
        positive_key = z_j.unsqueeze(0)
        negative_keys = z_neg.unsqueeze(0)

        if cross:
            # Compute positive similarity
            positive_similarity = torch.sum(query * positive_key, dim=1)  # (batch_size,)

            # Compute negative similarities
            negative_similarity = torch.einsum('bd,bnd->bn', query, negative_keys)  # (batch_size, num_negatives)

            # Combine positive and negative similarities
            logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)  # (batch_size, 1 + num_negatives)

            # Scale by temperature
            logits /= self.temperature

            # Create labels: positive sample is the first in each row
            labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        else:
            # 计算正样本与锚点的相似度，这里使用余弦相似度（也可以使用点积等其他相似度度量方式）
            positive_similarity = F.cosine_similarity(query, positive_key, dim=1)
            positive_similarity = positive_similarity.unsqueeze(1)  # 扩展维度为 (batch_size, 1)

            # 计算负样本与锚点的相似度
            negative_similarities = F.cosine_similarity(
                query.unsqueeze(1).expand_as(negative_keys),
                negative_keys,
                dim=2
            )

            # 拼接正样本和负样本的相似度，形状变为 (batch_size, 1 + num_negatives)
            all_similarities = torch.cat([positive_similarity, negative_similarities], dim=1)

            # 对相似度进行温度缩放，这里假设温度参数为0.07，你可以根据实际情况调整
            all_similarities = all_similarities / self.temperature

            # 计算InfoNCE损失
            exp_similarities = torch.exp(all_similarities)
            numerator = exp_similarities[:, 0:1]  # 正样本的相似度得分
            denominator = torch.sum(exp_similarities, dim=1, keepdim=True)  # 所有样本相似度得分总和
            loss = -torch.log(numerator / denominator)

        return loss

# Negative Sampling function
def negative_sampling(graphs, current_graph_idx, num_negatives=1):
    """ Randomly sample negative graphs that are not similar to the current graph """
    negative_pairs = []
    for _ in range(num_negatives):
        # Randomly select a graph that is not the current one
        neg_idx = random.choice([i for i in range(len(graphs)) if i != current_graph_idx])
        negative_pairs.append(graphs[neg_idx])
    return negative_pairs

# Function for aggregating updated node embeddings to obtain global graph representation
def aggregate_graph_features(updated_node_embeddings):
    return updated_node_embeddings.mean(dim=0)  # Mean pooling of node embeddings


# Training function with augmentations and checkpoints

def train_gat_model(graphs, args):
    epochs = args.epochs
    lr = args.learning_rate
    temperature = args.temperature
    embedding_dim = args.node_dim
    checkpoint_path = args.checkpoint_path

    # 检查GPU是否可用，如果可用则获取GPU设备，否则使用CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the GAT model and optimizer
    model = GATModel(in_channels=embedding_dim, out_channels=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(temperature=temperature).to(device)

    # Check if there's a checkpoint to load from
    start_epoch = 0
    try:
        model_state, optimizer_state, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        for i, graph in enumerate(graphs):
            node_features = graph.x.to(device)  # 将节点特征数据移动到GPU设备上
            edge_index = graph.edge_index.to(device)  # 将边索引数据移动到GPU设备上

            # Apply GAT model to get updated node embeddings
            updated_node_embeddings = model(node_features, edge_index)

            # Aggregate node embeddings to form global graph feature
            global_feature = aggregate_graph_features(updated_node_embeddings)

            # Positive pair: Use the next graph as the positive pair for simplicity
            if i < len(graphs) - 1:
                next_graph = graphs[i + 1]
                next_node_features = next_graph.x.to(device)
                next_edge_index = next_graph.edge_index.to(device)
                next_updated_node_embeddings = model(next_node_features, next_edge_index)
                next_global_feature = aggregate_graph_features(next_updated_node_embeddings)

                # Negative pairs: Sample a negative graph
                negative_graphs = negative_sampling(graphs, i, args.num_negatives)
                negative_updated_node_embeddings = []
                for neg_graph in negative_graphs:
                    negative_node_features = neg_graph.x.to(device)
                    negative_edge_index = neg_graph.edge_index.to(device)
                    neg_update_emb = model(negative_node_features, negative_edge_index)
                    neg_feature = aggregate_graph_features(neg_update_emb)
                    negative_updated_node_embeddings.append(neg_feature)

                negative_updated_node_embeddings = torch.stack(negative_updated_node_embeddings)
                # Compute contrastive loss between positive and negative pairs
                loss = criterion(global_feature, next_global_feature, negative_updated_node_embeddings,cross = False)

                # total_loss += loss.item()
                # Ensure that loss is a tensor
                total_loss += loss  # `total_loss` is now a tensor, not a float

        optimizer.zero_grad()  # Before loss.backward()
        # Backpropagation and optimization step
        total_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss / len(graphs)}')

    save_checkpoint(epoch, model, optimizer, loss, checkpoint_path)

    return model

# Define a Graph Autoencoder with a GCN Encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Efficient Graph Batch Generation (for large graph datasets)
def generate_graphs(node_embedding_file_path,data_json_file_path,args):
    graphs = []

    loaded_data = np.load(node_embedding_file_path)
    node_embedding_list = loaded_data

    args.node_dim = node_embedding_list[0].shape[0]

    with open(data_json_file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)


    for items in tqdm(data_list,desc="Prepare Graph Data",total=len(data_list)):
        kopl_program = items["kopl_program"]
        kopl_edges = items["kopl_edges"]
        node_feature = []
        for index in range(len(kopl_program)):
            try:
                kopl_node = next(item for item in kopl_program if item['index'] == index)
                node_index = kopl_node["node_index"]
                node_embedding = node_embedding_list[node_index]
                node_embedding = torch.from_numpy(node_embedding)
                node_feature.append(node_embedding)
            except StopIteration:
                print(f"没有找到index为{index}的数据")
        edge_index = torch.tensor(kopl_edges, dtype=torch.long)

        node_feature = torch.stack(node_feature)

        edge_index_sparse = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(len(kopl_program), len(kopl_program)))
        graphs.append(Data(x=node_feature, edge_index=edge_index_sparse))

    return graphs

# Save checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")

# Load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}. Loss: {loss}")
    return model, optimizer, epoch, loss

# Save embeddings to a file
def save_embeddings(embeddings, filepath):
    np.save(filepath, embeddings)

# Main training code
def main(args):
    # Generate dummy graph dataset
    # node_embedding_file = os.path.join(args.save_dir,args.node_emb_path.format(node_dim))
    node_embedding_file = os.path.join(args.save_dir,args.node_emb_path.format(args.model_name))
    data_json_file = os.path.join(args.data_folder,args.processed_data)
    graphs  = generate_graphs(node_embedding_file,data_json_file,args)

    # GAT
    # Train the GAT model with contrastive loss and augmentations
    trained_model = train_gat_model(graphs, args)

    # After training, obtain global graph representations (no labels involved)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_graph_features = []

    for graph in graphs:
        node_features = graph.x.to(device)  # 将节点特征数据移动到GPU设备上
        edge_index = graph.edge_index.to(device)  # 将边索引数据移动到GPU设备上

        # Apply trained GAT model
        trained_model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            updated_node_embeddings = trained_model(node_features, edge_index)

        # Aggregate node embeddings to obtain global graph feature
        global_feature = aggregate_graph_features(updated_node_embeddings)
        global_feature = global_feature.cpu().detach().numpy()  # 将结果从GPU移回CPU，并转换为numpy数组
        global_graph_features.append(global_feature)


    all_embeddings = np.array(global_graph_features)
    save_dir = args.save_dir
    filepath = os.path.join(save_dir, "graph_feature_{}.npy".format(args.model_name))
    save_embeddings(all_embeddings, filepath)



if __name__ == "__main__":
    main()
