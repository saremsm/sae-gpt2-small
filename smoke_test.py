from sparse_autoencoder import SparseAutoencoder, SAEConfig
from training import InlineActivationSource, train_sae
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
cfg = SAEConfig(d_model=768, n_features=256, l1_coefficient=8e-4, warmup_steps=10)
sae = SparseAutoencoder(cfg)
src = InlineActivationSource(model, layer=8, batch_size=4, context_length=64, max_batches=2)
history = train_sae(sae, src, n_training_tokens=2000, log_interval=1, device="cpu")

if history["loss"]:
    print("smoke test passed, final loss:", history["loss"][-1])
else:
    print("Training ran but produced no log entries - increase n_training_tokens or lower log_interval")
