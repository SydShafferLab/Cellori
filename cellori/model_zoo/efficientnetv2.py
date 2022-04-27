import copy
import math

from flax import linen as nn
from functools import partial
from typing import Any, Callable

from .capture import CaptureModule
from .conv_blocks import MBConvBlock, FusedMBConvBlock
from .efficientnetv2_config import DEFAULT_BLOCKS_ARGS

CONV_KERNEL_INITIALIZER = nn.initializers.variance_scaling(scale=2.0, mode='fan_out', distribution='truncated_normal')

ModuleDef = Any


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class EfficientNetV2(CaptureModule):
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish
    stem_strides: int = 2
    blocks_args: list = None
    default_size: int = None
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
            strides=self.stem_strides,
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
                # Determine which conv type to use:
                args = args.unfreeze()
                block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]

                x = block(
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    dropout_rate=self.drop_connect_rate * b / blocks,
                    **args,
                )(x)
                b += 1

            captures = self.capture(capture_list, captures, x, "C{}".format(i + 1))

        return self.output(capture_list, captures, x)


def build_efficientnetv2(
        model_name: str,
        width_coefficient: float,
        depth_coefficient: float,
        default_size: int,
        depth_divisor: int = 8,
        min_depth: int = 8
):
    default_blocks_args = DEFAULT_BLOCKS_ARGS[model_name]
    default_blocks_args = copy.deepcopy(default_blocks_args)

    blocks_args = []
    b = 0
    for i, args in enumerate(default_blocks_args):

        block_args = []

        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor
        )

        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient)
        for j in range(repeats):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            args["name"] = "block{}{}_".format(i + 1, chr(j + 97))

            block_args.append(args.copy())

            b += 1

        blocks_args.append(block_args)

        model = partial(
            EfficientNetV2,
            blocks_args=blocks_args,
            default_size=default_size,
            name=model_name
        )

    return model


EfficientNetV2B0 = build_efficientnetv2(
    model_name="efficientnetv2-b0",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=224
)

EfficientNetV2B1 = build_efficientnetv2(
    model_name="efficientnetv2-b1",
    width_coefficient=1.0,
    depth_coefficient=1.1,
    default_size=240
)

EfficientNetV2B2 = build_efficientnetv2(
    model_name="efficientnetv2-b2",
    width_coefficient=1.1,
    depth_coefficient=1.2,
    default_size=260
)

EfficientNetV2B3 = build_efficientnetv2(
    model_name="efficientnetv2-b3",
    width_coefficient=1.2,
    depth_coefficient=1.4,
    default_size=300
)

EfficientNetV2S = build_efficientnetv2(
    model_name="efficientnetv2-s",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=384
)

EfficientNetV2M = build_efficientnetv2(
    model_name="efficientnetv2-m",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=480
)

EfficientNetV2L = build_efficientnetv2(
    model_name="efficientnetv2-l",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=480
)
