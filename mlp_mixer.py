import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from functools import partial


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def check_sizes(image_size, patch_size):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    return num_patches

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., dense = nn.Linear):
        super().__init__()
        self.net = nn.Sequential(
            dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def execute(self, x):
        return self.net(x)


class MLPMixer(nn.Module):
    def __init__(self, num_patches, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, FeedForward(num_patches, num_patches * expansion_factor, dropout, chan_first)),
                PreNormResidual(d_model, FeedForward(d_model, d_model * expansion_factor, dropout, chan_last))
            ) for _ in range(depth)]
        )

    def execute(self, x):
        return self.model(x)

class MLPMixerForImageClassification(MLPMixer):
    def __init__(
        self, 
        in_channels = 3, 
        d_model = 512, 
        num_classes = 1000, 
        patch_size = 16, 
        image_size = 224, 
        depth = 12, 
        expansion_factor = 4):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(num_patches, d_model, expansion_factor, depth)

        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
        )

        self.active = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def execute(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)     # feature vector!!!!
        embedding = self.active(embedding)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out

def MLPMixer_S_16(num_classes: int = 1000, **kwargs):
    model = MLPMixerForImageClassification(patch_size = 16, d_model = 512, depth = 12, 
                    num_classes=num_classes,
                    **kwargs)
    return model

def MLPMixer_128_4(num_classes: int = 1000, **kwargs):
    model = MLPMixerForImageClassification(patch_size = 16, d_model = 128, depth = 4, 
                    num_classes=num_classes,
                    **kwargs)
    return model
def MLPMixer_64_8(num_classes: int = 1000, **kwargs):
    model = MLPMixerForImageClassification(patch_size = 16, d_model = 64, depth = 8, 
                    num_classes=num_classes,
                    **kwargs)
    return model