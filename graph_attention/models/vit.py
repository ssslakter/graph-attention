import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .layers import Attention, ViTBlock


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=1000,
        dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attention_layer=Attention,
        channels=3,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        
        patch_dim = channels * patch_size * patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim, bias=True)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None
        total_tokens = num_patches + (1 if use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, dim) * .02)
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            ViTBlock(attention_layer, dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes, bias=True)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embedding
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
            
        return self.head(x)

    @classmethod
    def from_pretrained(cls, model_name, attention_layer=Attention, **kwargs):
        """
        Loads a timm model, extracts config, creates TimmCompatibleViT, and loads weights.
        """
        import timm
        print(f"Loading weights from timm: {model_name}")
        t_model = timm.create_model(model_name, pretrained=True)
        t_model.eval()

        cfg = {
            'img_size': t_model.patch_embed.img_size[0],
            'patch_size': t_model.patch_embed.patch_size[0],
            'num_classes': t_model.head.out_features,
            'dim': t_model.embed_dim,
            'depth': len(t_model.blocks),
            'num_heads': t_model.blocks[0].attn.num_heads,
            'use_cls_token': True
        }
        cfg.update(kwargs)
        
        model = cls(attention_layer=attention_layer, **cfg)

        t_sd = t_model.state_dict()
        sd = model.state_dict()
        
        mapping = {
            'cls_token': 'cls_token',
            'pos_embedding': 'pos_embed',
            'norm.weight': 'norm.weight', 
            'norm.bias': 'norm.bias',
            'head.weight': 'head.weight', 
            'head.bias': 'head.bias',
            'to_patch_embedding.1.bias': 'patch_embed.proj.bias'
        }

        for k_our, k_timm in mapping.items():
            if k_timm in t_sd:
                sd[k_our].copy_(t_sd[k_timm])

        conv_w = t_sd['patch_embed.proj.weight']
        sd['to_patch_embedding.1.weight'].copy_(conv_w.flatten(1))

        for i in range(cfg['depth']):
            p_our = f'blocks.{i}.'
            p_timm = f'blocks.{i}.'
            
            block_map = {
                f'{p_our}norm1': f'{p_timm}norm1',
                f'{p_our}norm2': f'{p_timm}norm2',
                f'{p_our}attn.to_qkv': f'{p_timm}attn.qkv',
                f'{p_our}attn.to_out': f'{p_timm}attn.proj',
                f'{p_our}mlp.0': f'{p_timm}mlp.fc1',
                f'{p_our}mlp.2': f'{p_timm}mlp.fc2',
            }
            
            for m_our, m_timm in block_map.items():
                sd[f'{m_our}.weight'].copy_(t_sd[f'{m_timm}.weight'])
                sd[f'{m_our}.bias'].copy_(t_sd[f'{m_timm}.bias'])

        missing, unexpected = model.load_state_dict(sd, strict=False)
        
        real_missing = [k for k in missing if 'alphas' not in k]
        if real_missing:
            print(f"Warning: Missing keys: {real_missing}")
        
        return model