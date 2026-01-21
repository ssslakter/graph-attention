from trainer_tools.imports import *
from torchvision.utils import make_grid


def show_img(x: t.Tensor, ax=None, title=""):
    """Displays a single image tensor of shape (C, H, W)."""
    if ax is None:
        ax = plt.subplot()
    ax.imshow(x.cpu().detach().permute(1, 2, 0))
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def show_imgs(xb: t.Tensor, ax=None, title=""):
    """Arranges and displays a batch of image tensors (B, C, H, W) in a grid."""
    nrow = int(xb.shape[0] ** 0.5)
    xb = make_grid(xb, nrow=nrow, pad_value=1)
    return show_img(xb, ax, title)
