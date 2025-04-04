
# -----------------------------
# README FILE (Markdown String)
# -----------------------------
README = """
# CEDA: Causal Echo Diffusion Attenuator

This is the official implementation of the **CEDA** model introduced in the TOIS 2025 paper:
**"Breaking the Loop: Causal Learning to Mitigate Echo Chambers in Social Networks"**

## Highlights
- **User Dual Modelling**: Merges user attributes and positional encodings.
- **Residual Estimation**: Adjusts embeddings using causal residual.
- **Causal Transformer**: Integrates residuals into multi-head attention.
- **Diffusion Predictor**: Predicts interactions and measures diversity.
- **Evaluation Metrics**: MAE, Intra-list Diversity (ILD), Category Coverage (CC), Gini Coefficient.

## How to Run
```bash
python ceda_echo_chamber.py
```

## Dependencies
- Python >= 3.7
- PyTorch
- scikit-learn
- numpy

## License
MIT License
"""

print("\n--- README ---")
print(README)
