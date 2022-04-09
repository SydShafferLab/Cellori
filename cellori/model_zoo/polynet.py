from flax import linen as nn
from typing import Any, Callable, Tuple

ModuleDef = Any

class PolyHead(nn.Module):

    conv: ModuleDef = nn.Conv
    act: Callable = nn.softmax
    num_classes: int = 3
    name: str = None

    @nn.compact
    def __call__(self, x):

        x = self.conv(
            features=self.num_classes,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            name=self.name + 'conv'
        )(x)
        x = self.act(x)

        return x


class PolyNet(nn.Module):

    fpn: ModuleDef
    semantic_heads: Tuple[int] = (1, 3)

    @nn.compact
    def __call__(self, x, train: bool = True):

        # Get aggregate feature maps from FPN
        agg_features = self.fpn()(x, train=train)

        # Create list to store feature maps
        poly_features = []

        # Process aggregate feature map through semantic heads
        for i, num_classes in enumerate(self.semantic_heads):
            f = PolyHead(
                act=nn.activation.softmax if num_classes > 1 else nn.activation.relu,
                num_classes=num_classes,
                name='semantic{}_'.format(i + 1)
            )(agg_features)
            poly_features.append(f)

        return poly_features
