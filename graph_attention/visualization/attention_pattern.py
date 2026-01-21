import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


def visualize_head_attention(
    attention_pattern: torch.Tensor,
    image: torch.Tensor,
    patch_size: int,
    head_index: int = 0,
    token_index: int = None,
    title: str = "Attention Map",
):
    """
    Visualizes the attention pattern for a specific head.

    Args:
        attention_pattern: Tensor of shape [Batch, Heads, Tokens, Tokens]
                           (We assume Batch=1 for visualization)
        image: Input tensor of shape [Batch, Channels, Height, Width]
        patch_size: The patch size used in the model (int)
        head_index: Which attention head to visualize.
        token_index: Which query token to visualize attention *from*.
                     If None, visualizes the center patch of the image.
    """

    attn = attention_pattern[0, head_index]  # Shape: [Tokens, Tokens]
    img_tensor = image[0]  # Shape: [C, H, W]

    C, H, W = img_tensor.shape

    h_patches = H // patch_size
    w_patches = W // patch_size
    num_tokens = h_patches * w_patches

    assert (
        attn.shape[-1] == num_tokens
    ), f"Mismatch: Attention has {attn.shape[-1]} tokens, but image suggests {num_tokens} patches."

    if token_index is None:
        center_h = h_patches // 2
        center_w = w_patches // 2
        token_index = (center_h * w_patches) + center_w

    attn_map_flat = attn[token_index, :]

    attn_map_2d = attn_map_flat.view(h_patches, w_patches)

    attn_map_resized = F.interpolate(
        attn_map_2d.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="nearest",
    ).squeeze()

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    attn_np = attn_map_resized.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\nQuery Patch: {token_index}")

    py = (token_index // w_patches) * patch_size
    px = (token_index % w_patches) * patch_size
    rect = plt.Rectangle((px, py), patch_size, patch_size, linewidth=2, edgecolor="r", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(attn_np, cmap="jet", alpha=0.5)  # Alpha blends heatmap over image
    axes[1].set_title(f"Head {head_index} Attention\n(Red = High Attention)")
    axes[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_attention_matrix(attention_pattern, head_index=0):
    """
    Helper to see the raw N x N grid.
    """
    attn = attention_pattern[0, head_index].cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.title(f"Raw Attention Matrix (Head {head_index})\nShape: {attn.shape}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()
