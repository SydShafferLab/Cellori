import copy

from functools import partial

from .efficientnetv2 import EfficientNetV2, round_filters, round_repeats
from .originnet_config import DEFAULT_BLOCKS_ARGS

OriginNet = partial(
    EfficientNetV2,
    stem_strides=1,
    drop_connect_rate=0
)


def build_originnet(
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
            OriginNet,
            blocks_args=blocks_args,
            default_size=default_size,
            name=model_name
        )

    return model


OriginNetS = build_originnet(
    model_name="originnet-s",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=256
)

OriginNetM = build_originnet(
    model_name="originnet-m",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=256
)

OriginNetL = build_originnet(
    model_name="originnet-l",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=256
)
