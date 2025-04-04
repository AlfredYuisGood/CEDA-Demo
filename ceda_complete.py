
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict

# Set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Constants
NUM_USERS = 100
EMBED_DIM = 16
NUM_CATEGORIES = 5
SEQUENCE_LEN = 10
NUM_CLUSTERS = 4

# Data Simulation
user_attributes = torch.randint(0, 2, (NUM_USERS, 6)).float()
positions = torch.arange(SEQUENCE_LEN).unsqueeze(1).repeat(1, EMBED_DIM // 2)
div_term = torch.exp(torch.arange(0, EMBED_DIM, 2) * -(np.log(10000.0) / EMBED_DIM))
positional_encodings = torch.zeros(SEQUENCE_LEN, EMBED_DIM)
positional_encodings[:, 0::2] = torch.sin(positions * div_term)
positional_encodings[:, 1::2] = torch.cos(positions * div_term)
cascades = [torch.randperm(NUM_USERS)[:SEQUENCE_LEN].tolist() for _ in range(NUM_CLUSTERS)]
observed_outcomes = torch.rand(NUM_USERS, 1)

# Dual User Embedding
class UserDualEmbedding(nn.Module):
    def __init__(self, attr_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(attr_dim + embed_dim, embed_dim)

    def forward(self, attrs, pos_enc):
        combined = torch.cat([attrs, pos_enc], dim=1)
        return self.linear(combined)

# Residual Estimation
class ResidualEstimator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wz = nn.Linear(embed_dim, embed_dim)
        self.Wg = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, e, o):
        z = F.relu(self.Wz(e))
        g = F.relu(self.Wg(o))
        return self.mlp(z - g), z, g

# Attention
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, residual):
        residual = residual.unsqueeze(1).repeat(1, x.shape[1], 1)
        x_adj = x - residual
        attn_output, _ = self.mha(x_adj, x_adj, x_adj)
        return attn_output

# Causal Transformer
class CausalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = CausalMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
                                 nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, x, residual):
        out = self.attn(x, residual)
        return self.ffn(out + x)

# Diffusion Predictor
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
        unique_categories = torch.sum(torch.any(active.bool(), dim=0)).item()
        return unique_categories / NUM_CATEGORIES

# CEDA Model
class CEDA(nn.Module):
    def __init__(self, attr_dim, embed_dim, num_heads, num_categories):
        super().__init__()
        self.dual_embed = UserDualEmbedding(attr_dim, embed_dim)
        self.residual = ResidualEstimator(embed_dim)
        self.transformer = CausalTransformer(embed_dim, num_heads)
        self.predictor = SocialDiffusionPredictor(embed_dim, num_categories)

    def forward(self, attrs, pos_enc, outcomes):
        E = self.dual_embed(attrs, pos_enc)
        residual, z, g = self.residual(E, outcomes)
        unbiased = self.transformer(E.unsqueeze(0), residual).squeeze(0)
        return unbiased, z, g, residual, E

# Training
model = CEDA(attr_dim=6, embed_dim=EMBED_DIM, num_heads=4, num_categories=NUM_CATEGORIES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda1 = lambda2 = lambda3 = lambda4 = 1.0

for epoch in range(5):
    pos_sample = positional_encodings[0:NUM_USERS]
    unbiased, z, g, residual, E = model(user_attributes, pos_sample, observed_outcomes)
    u_idx, v_idx = torch.randint(0, NUM_USERS, (20,)), torch.randint(0, NUM_USERS, (20,))
    u_embed, v_embed = unbiased[u_idx], unbiased[v_idx]
    D_true, D_pred = torch.rand(20), model.predictor.predict_diffusion(u_embed, v_embed)

    Lm = model.predictor.compute_MAE(D_pred, D_true)
    Lr = F.mse_loss(z, g)
    Ld = sum([model.predictor.compute_ILD(unbiased, c) for c in cascades]) / len(cascades)
    Lc = model.predictor.compute_CC(unbiased)
    loss = lambda1 * Lr + lambda2 * Lm + lambda3 * (1 - Ld) + lambda4 * (1 - Lc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} | Loss: {loss.item():.4f} | MAE: {Lm.item():.4f} | ILD: {Ld:.4f} | CC: {Lc:.4f}")

# Intervention
def targeted_intervention(unbiased_embeddings):
    clusters = KMeans(n_clusters=NUM_CLUSTERS).fit(unbiased_embeddings.detach().numpy())
    ild_per_cluster = defaultdict(list)
    for i, c in enumerate(clusters.labels_):
        ild_per_cluster[c].append(i)
    for k, users in ild_per_cluster.items():
        if len(users) < 3: continue
        sim = cosine_similarity(unbiased_embeddings[users].detach().numpy())
        if np.mean(sim) > 0.8:
            print(f"Cluster {k} is echo-chamber like. Consider rewiring...")

# Eval helpers
def print_diffusion_predictions(embeddings):
    print("\nSample Predicted Diffusion Probabilities:")
    for _ in range(5):
        u1, u2 = random.sample(range(NUM_USERS), 2)
        prob = model.predictor.predict_diffusion(embeddings[u1].unsqueeze(0), embeddings[u2].unsqueeze(0)).item()
        print(f"User {u1} → User {u2} | Predicted Probability: {prob:.3f}")

def print_category_probabilities(embeddings):
    print("\nSample Category Probabilities:")
    probs = torch.sigmoid(model.predictor.Wc(embeddings))
    for i in range(5):
        top_categories = torch.topk(probs[i], 3).indices.tolist()
        print(f"User {i} | Top Categories: {top_categories}")

def compute_gini(probs):
    sorted_probs, _ = torch.sort(probs)
    n = probs.numel()
    index = torch.arange(1, n + 1)
    return ((2 * index - n - 1) * sorted_probs).sum() / (n * sorted_probs.sum())

# Final evaluation
print("\n--- Final Evaluation ---")
with torch.no_grad():
    unbiased_final, _, _, _, _ = model(user_attributes, positional_encodings[0:NUM_USERS], observed_outcomes)
    print_diffusion_predictions(unbiased_final)
    print_category_probabilities(unbiased_final)
    flat_probs = torch.sigmoid(model.predictor.Wc(unbiased_final)).flatten()
    print(f"\nGini Coefficient: {compute_gini(flat_probs):.4f}")
    targeted_intervention(unbiased_final)

print("\n✅ Complete CEDA Demo Finished.")
