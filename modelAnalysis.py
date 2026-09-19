# Load saved models from path
import torch

from attentionCalcs import rope_encode, sinusoidal_encode
from attentionModels import Embedder, MultiHeadAttention, OutputHead, ToyTransformer
import matplotlib.pyplot as plt
import os

# Load the saved models
embed_dim = 8
num_heads = 2
output_dim = 8
seq_length = 8

embedder_rope = Embedder(input_dim=seq_length, embed_dim=embed_dim)
layer1_rope = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=rope_encode)
layer2_rope = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=rope_encode)
output_head_rope = OutputHead(embed_dim, output_dim)
rope_transformer = ToyTransformer(embedder_rope, layer1_rope, layer2_rope, output_head_rope, input_position_encoder=None)

embedder = Embedder(input_dim=seq_length, embed_dim=embed_dim)
layer1 = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=None)
layer2 = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=None)
output_head = OutputHead(embed_dim, output_dim)
sinusoidal_transformer = ToyTransformer(embedder, layer1, layer2, output_head, input_position_encoder=sinusoidal_encode)

rope_transformer.load_state_dict(torch.load("rope_transformer.pth"))
rope_transformer.eval()
sinusoidal_transformer.load_state_dict(torch.load("sinusoidal_transformer.pth"))
sinusoidal_transformer.eval()

# Move models to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rope_transformer.to(device)
sinusoidal_transformer.to(device)

input_data = [[7,6,5,4,3,2,1,0], [4,3,5,2,6,1,7,0]]
input_tensor = torch.tensor(input_data, dtype=torch.long).to(device)

for t in input_tensor:
    # Perform inference for each individual sequence
    t = t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        rope_output, rope_attn_weights, _, _, _, rope_residuals = rope_transformer(t)
        sinusoidal_output, sinusoidal_attn_weights, _, _, _, sinusoidal_residuals = sinusoidal_transformer(t)
    print("Input:", t)
    print("RoPE Output:", rope_output.argmax(dim=-1))
    print("Sinusoidal Output:", sinusoidal_output.argmax(dim=-1))
    print()

    # Create attention maps for each layer's visualization and save to png files    

    os.makedirs("attention_maps", exist_ok=True)

    for layer_idx, layer in enumerate(rope_attn_weights):
        plt.figure(figsize=(8, 8))
        plt.imshow(layer[0][0].cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"RoPE Attention Map - Layer {layer_idx}")
        plt.savefig(f"attention_maps/rope_attention_{t[0].tolist()}_layer_{layer_idx}_head_0.png")
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(layer[1][0].cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"RoPE Attention Map - Layer {layer_idx}")
        plt.savefig(f"attention_maps/rope_attention_{t[0].tolist()}_layer_{layer_idx}_head_1.png")
        plt.close()

    for layer_idx, layer in enumerate(sinusoidal_attn_weights):
        plt.figure(figsize=(8, 8))
        plt.imshow(layer[0][0].cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Sinusoidal Attention Map - Layer {layer_idx}")
        plt.savefig(f"attention_maps/sinusoidal_attention_{t[0].tolist()}_layer_{layer_idx}_head_0.png")
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(layer[1][0].cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Sinusoidal Attention Map - Layer {layer_idx}")
        plt.savefig(f"attention_maps/sinusoidal_attention_{t[0].tolist()}_layer_{layer_idx}_head_1.png")
        plt.close()

    # Run residuals through output head
    rope_lens_logits = [
        rope_transformer.output_head(residual)
        for residual in rope_residuals
    ]
    sinusoidal_lens_logits = [
        sinusoidal_transformer.output_head(residual)
        for residual in sinusoidal_residuals
    ]

    print("RoPE Lens Predictions:", [logits.argmax(dim=-1) for logits in rope_lens_logits])
    print("Sinusoidal Lens Predictions:", [logits.argmax(dim=-1) for logits in sinusoidal_lens_logits])
    print()