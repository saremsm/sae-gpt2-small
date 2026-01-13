"""compute FVU for a trained SAE checkpoint. standard FVU: residual SS over SS about
the per-dimension mean."""

import torch
from transformer_lens import HookedTransformer
from sparse_autoencoder import SparseAutoencoder, SAEConfig
from training import InlineActivationSource

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2").to(device)

ckpt = torch.load("sae_gpt2_layer8.pt", map_location=device, weights_only=True)
cfg = SAEConfig(**ckpt["config"])
sae = SparseAutoencoder(cfg).to(device)
sae.load_state_dict(ckpt["sae_state_dict"])
sae.eval()

src = InlineActivationSource(
    model, layer=ckpt["layer"], batch_size=8, context_length=128,
    device=device, max_batches=10,
)

chunks = []
for chunk in src:
    chunks.append(chunk)
acts = torch.cat(chunks, dim=0).to(device)

with torch.no_grad():
    out = sae(acts)                # normalizes internally
    x = sae.preprocess(acts)       # the target lives in normalized space

    resid_ss = (x - out.reconstructed).pow(2).sum()
    total_ss = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    fvu = (resid_ss / total_ss).item()
    ve = 1.0 - fvu

print(f"Tokens evaluated     : {acts.shape[0]}")
print(f"Reconstruction MSE   : {out.reconstruction_loss.item():.4f}")
print(f"L0 (avg active feats): {out.l0.item():.1f}")
print(f"FVU                  : {fvu:.3f}")
print(f"Variance explained   : {ve:.3f}  ({ve*100:.1f}%)  [1 - FVU]")
