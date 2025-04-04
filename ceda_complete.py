# Sample code for CEDA: Causal Echo Diffusion Attenuator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

NUM_USERS = 100
EMBED_DIM = 16
NUM_CATEGORIES = 5
SEQUENCE_LEN = 10
NUM_CLUSTERS = 4
ATTR_DIM = 6

# -----------------------------
# Positional Encodings
# -----------------------------
def get_positional_encodings(seq_len, embed_dim):
    positions = torch.arange(seq_len).unsqueeze(1).repeat(1, embed_dim // 2)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pos_enc = torch.zeros(seq_len, embed_dim)
    pos_enc[:, 0::2] = torch.sin(positions * div_term)
    pos_enc[:, 1::2] = torch.cos(positions * div_term)
    return pos_enc

# -----------------------------
# Synthetic Data
# -----------------------------
def generate_synthetic_data():
    attrs = torch.randint(0, 2, (NUM_USERS, ATTR_DIM)).float()
    outcomes = torch.rand(NUM_USERS, 1)
    cascades = [torch.randperm(NUM_USERS)[:SEQUENCE_LEN].tolist() for _ in range(NUM_CLUSTERS)]
    pos_enc = get_positional_encodings(SEQUENCE_LEN, EMBED_DIM)[:NUM_USERS]
    return attrs, pos_enc, outcomes, cascades

# -----------------------------
# User Dual Embedding
# -----------------------------
class UserDualEmbedding(nn.Module):
    def __init__(self, attr_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(attr_dim + embed_dim, embed_dim)

    def forward(self, attrs, pos_enc):
        return self.linear(torch.cat([attrs, pos_enc], dim=1))

# -----------------------------
# Residual Estimator
# -----------------------------
class ResidualEstimator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wz = nn.Linear(embed_dim, embed_dim)
        self.Wg = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, e, o):
        z = F.relu(self.Wz(e))
        g = F.relu(self.Wg(o))
        return self.mlp(z - g), z, g

# -----------------------------
# Causal Attention Layer
# -----------------------------
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, residual):
        residual = residual.unsqueeze(1).repeat(1, x.shape[1], 1)
        x_adj = x - residual
        out, _ = self.mha(x_adj, x_adj, x_adj)
        return out

# -----------------------------
# Transformer
# -----------------------------
class CausalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = CausalMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, residual):
        return self.ffn(self.attn(x, residual) + x)

# -----------------------------
# Social Diffusion Predictor
# -----------------------------
class SocialDiffusionPredictor(nn.Module):
    def __init__(self, embed_dim, num_categories):
        super().__init__()
        self.M = nn.Linear(embed_dim * 2, 1)
        self.Wc = nn.Linear(embed_dim, num_categories)

    def predict_diffusion(self, u1, u2):
        x = torch.cat([u1, u2], dim=1)
        return torch.sigmoid(self.M(x)).squeeze(-1)

    def compute_MAE(self, pred, truth):
        return F.l1_loss(pred, truth)

    def compute_ILD(self, embeddings, cascade):
        emb = embeddings[cascade]
        sim = cosine_similarity(emb.detach().numpy())
        n = len(cascade)
        return (1 - sim).sum() / (n * (n - 1))

    def compute_CC(self, embeddings):
        probs = torch.sigmoid(self.Wc(embeddings))
        active = (probs > 0.3).float()
        return torch.sum(torch.any(active.bool(), dim=0)).item() / NUM_CATEGORIES

# -----------------------------
# Full Model
# -----------------------------
class CEDA(nn.Module):
    def __init__(self, attr_dim, embed_dim, num_heads, num_categories):
        super().__init__()
        self.embed = UserDualEmbedding(attr_dim, embed_dim)
        self.residual = ResidualEstimator(embed_dim)
        self.transformer = CausalTransformer(embed_dim, num_heads)
        self.predictor = SocialDiffusionPredictor(embed_dim, num_categories)

    def forward(self, attrs, pos_enc, outcomes):
        E = self.embed(attrs, pos_enc)
        residual, z, g = self.residual(E, outcomes)
        unbiased = self.transformer(E.unsqueeze(0), residual).squeeze(0)
        return unbiased, z, g, residual, E

# -----------------------------
# Gini Utility
# -----------------------------
def compute_gini(probs):
    sorted_probs, _ = torch.sort(probs)
    n = probs.numel()
    index = torch.arange(1, n + 1)
    return ((2 * index - n - 1) * sorted_probs).sum() / (n * sorted_probs.sum())

# -----------------------------
# Training Loop
# -----------------------------
def train_and_evaluate(model, attrs, pos_enc, outcomes, cascades):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        unbiased, z, g, _, _ = model(attrs, pos_enc, outcomes)
        u_idx, v_idx = torch.randint(0, NUM_USERS, (20,)), torch.randint(0, NUM_USERS, (20,))
        D_true = torch.rand(20)
        D_pred = model.predictor.predict_diffusion(unbiased[u_idx], unbiased[v_idx])
        Lr = F.mse_loss(z, g)
        Lm = F.l1_loss(D_pred, D_true)
        Ld = sum([model.predictor.compute_ILD(unbiased, c) for c in cascades]) / len(cascades)
        Lc = model.predictor.compute_CC(unbiased)
        loss = Lr + Lm + (1 - Ld) + (1 - Lc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, MAE={Lm.item():.4f}, ILD={Ld:.4f}, CC={Lc:.4f}")

# -----------------------------
# Final Evaluation
# -----------------------------
def final_outputs(model, attrs, pos_enc, outcomes):
    with torch.no_grad():
        unbiased, _, _, _, _ = model(attrs, pos_enc, outcomes)
        probs = torch.sigmoid(model.predictor.Wc(unbiased)).flatten()
        print("\nTop predicted categories per user (sample):")
        for i in range(5):
            top = torch.topk(probs[i * NUM_CATEGORIES:(i + 1) * NUM_CATEGORIES], 3).indices.tolist()
            print(f"User {i}: Top-3 categories → {top}")
        gini = compute_gini(probs)
        print(f"\nGini Coefficient: {gini:.4f}")

# -----------------------------
# Echo Chamber Detection + Visualization
# -----------------------------
def detect_echo_chambers(embeddings, n_clusters=NUM_CLUSTERS):
    clusters = KMeans(n_clusters=n_clusters).fit(embeddings.detach().numpy())
    echo_chambers = {}
    for label in range(n_clusters):
        cluster_indices = np.where(clusters.labels_ == label)[0]
        if len(cluster_indices) > 2:
            sim = cosine_similarity(embeddings[cluster_indices].detach().numpy())
            avg_sim = np.mean(sim)
            if avg_sim > 0.8:
                echo_chambers[label] = (cluster_indices, avg_sim)
    return echo_chambers

def plot_embeddings_2d(embeddings, title="User Embeddings", echo_labels=None):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings.detach().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c='gray', alpha=0.6, label="Users")
    if echo_labels:
        for label, (indices, _) in echo_labels.items():
            plt.scatter(emb_2d[indices, 0], emb_2d[indices, 1], label=f"Echo Cluster {label}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    attrs, pos_enc, outcomes, cascades = generate_synthetic_data()
    model = CEDA(attr_dim=ATTR_DIM, embed_dim=EMBED_DIM, num_heads=4, num_categories=NUM_CATEGORIES)
    train_and_evaluate(model, attrs, pos_enc, outcomes, cascades)
    final_outputs(model, attrs, pos_enc, outcomes)
    with torch.no_grad():
        unbiased, _, _, _, _ = model(attrs, pos_enc, outcomes)
        echo_clusters = detect_echo_chambers(unbiased)
        print("\n🔍 Detected Echo Chambers:")
        for cid, (indices, sim) in echo_clusters.items():
            print(f" - Cluster {cid}: {len(indices)} users | Avg Similarity: {sim:.2f}")
        plot_embeddings_2d(unbiased, title="CEDA User Embeddings", echo_labels=echo_clusters)
