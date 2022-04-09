DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': (3, 3),
    'repeats': 1,
    'input_filters': 32,
    'output_filters': 16,
    'expand_ratio': 1,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': (3, 3),
    'repeats': 2,
    'input_filters': 16,
    'output_filters': 24,
    'expand_ratio': 6,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': (5, 5),
    'repeats': 2,
    'input_filters': 24,
    'output_filters': 40,
    'expand_ratio': 6,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': (3, 3),
    'repeats': 3,
    'input_filters': 40,
    'output_filters': 80,
    'expand_ratio': 6,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': (5, 5),
    'repeats': 3,
    'input_filters': 80,
    'output_filters': 112,
    'expand_ratio': 6,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': (5, 5),
    'repeats': 4,
    'input_filters': 112,
    'output_filters': 192,
    'expand_ratio': 6,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': (3, 3),
    'repeats': 1,
    'input_filters': 192,
    'output_filters': 320,
    'expand_ratio': 6,
    'strides': 1,
    'se_ratio': 0.25
}]
