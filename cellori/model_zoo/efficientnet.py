import copy
import math

from flax import linen as nn
from functools import partial
from typing import Any, Callable

from .capture import CaptureModule
from .conv_blocks import MBConvBlock
from .efficientnet_config import DEFAULT_BLOCKS_ARGS

CONV_KERNEL_INITIALIZER = nn.initializers.variance_scaling(scale=2.0, mode='fan_out', distribution='truncated_normal')

ModuleDef = Any


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class EfficientNet(CaptureModule):
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish
    blocks_args: list = None
    default_size: int = 224
    drop_connect_rate: float = 0.2
    bn_momentum: float = 0.9

    @nn.compact
    def __call__(
            self,
            x,
            train: bool = True,
            capture_list: list = []
    ):

        captures = {}

        conv = partial(
            self.conv,
            kernel_size=(1, 1),
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_init=CONV_KERNEL_INITIALIZER,
        )
        norm = partial(
            self.norm,
            use_running_average=not train,
            axis=-1,
            momentum=self.bn_momentum,
        )

        # Build stem
        x = conv(
            features=self.blocks_args[0][0]['input_filters'],
            kernel_size=(3, 3),
            strides=2,
            name="stem_conv",
        )(x)
        x = norm(
            name="stem_bn"
        )(x)
        x = self.act(x)

        # Build blocks
        blocks_args = copy.deepcopy(self.blocks_args)
        b = 0
        blocks = float(sum(len(block_args) for block_args in blocks_args))

        for i, block_args in enumerate(blocks_args):

            for args in block_args:

                x = MBConvBlock(
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    dropout_rate=self.drop_connect_rate * b / blocks,
                    **args,
                )(x)
                b += 1

            captures = self.capture(capture_list, captures, x, "C{}".format(i + 1))

        return self.output(capture_list, captures, x)


def build_efficientnet(
        model_name: str,
        width_coefficient: float,
        depth_coefficient: float,
        default_size: int,
        drop_connect_rate: float,
        depth_divisor: int = 8,
):
    default_blocks_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)

    blocks_args = []
    b = 0

    for i, args in enumerate(default_blocks_args):

        block_args = []

        assert args['repeats'] > 0

        # Update block input and output filters based on depth multiplier.
        args['input_filters'] = round_filters(args['input_filters'], width_coefficient, depth_divisor)
        args['output_filters'] = round_filters(args['output_filters'], width_coefficient, depth_divisor)

        for j in range(round_repeats(args.pop('repeats'), depth_coefficient)):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['input_filters'] = args['output_filters']

            args["name"] = "block{}{}_".format(i + 1, chr(j + 97))

            block_args.append(args.copy())

            b += 1

        blocks_args.append(block_args)

        model = partial(
            EfficientNet,
            blocks_args=blocks_args,
            default_size=default_size,
            drop_connect_rate=drop_connect_rate,
            name=model_name
        )

    return model


EfficientNetB0 = build_efficientnet(
    model_name="efficientnetb0",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=224,
    drop_connect_rate=0.2
)

EfficientNetB1 = build_efficientnet(
    model_name="efficientnetb1",
    width_coefficient=1.0,
    depth_coefficient=1.1,
    default_size=240,
    drop_connect_rate=0.2
)

EfficientNetB2 = build_efficientnet(
    model_name="efficientnetb2",
    width_coefficient=1.1,
    depth_coefficient=1.2,
    default_size=260,
    drop_connect_rate=0.3
)

EfficientNetB3 = build_efficientnet(
    model_name="efficientnetb3",
    width_coefficient=1.2,
    depth_coefficient=1.4,
    default_size=300,
    drop_connect_rate=0.3
)

EfficientNetB4 = build_efficientnet(
    model_name="efficientnetb4",
    width_coefficient=1.4,
    depth_coefficient=1.8,
    default_size=380,
    drop_connect_rate=0.4
)

EfficientNetB5 = build_efficientnet(
    model_name="efficientnetb5",
    width_coefficient=1.6,
    depth_coefficient=2.2,
    default_size=456,
    drop_connect_rate=0.4
)

EfficientNetB6 = build_efficientnet(
    model_name="efficientnetb2",
    width_coefficient=1.8,
    depth_coefficient=2.6,
    default_size=528,
    drop_connect_rate=0.5
)

EfficientNetB7 = build_efficientnet(
    model_name="efficientnetb3",
    width_coefficient=2.0,
    depth_coefficient=3.1,
    default_size=600,
    drop_connect_rate=0.5
)
