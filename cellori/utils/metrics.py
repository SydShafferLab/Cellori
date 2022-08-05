import numpy as np

from functools import cached_property
from skimage import measure
from scipy import ndimage, stats

from cellori.utils import dynamics


class SpotMetrics:

    def __init__(self, y_true, y_single_pred, id_pred, **kwargs):

        self.y_true = y_true
        self.y_single_pred = y_single_pred
        self.id_pred = id_pred

        self.smooth_l1_beta = 1

        allowed_keys = list(self.__dict__.keys())
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError(f"Invalid arguments in constructor: {rejected_keys}")

    @cached_property
    def l1(self):

        _l1s = np.linalg.norm(self.y_true - self.y_single_pred, ord=1, axis=-1)
        _best_l1s_index = np.argmin(_l1s)
        _best_l1 = {
            'value': _l1s[_best_l1s_index],
            'match': _best_l1s_index
        }

        return _best_l1

    @cached_property
    def l2(self):

        _l2s = np.linalg.norm(self.y_true - self.y_single_pred, ord=2, axis=-1)
        _best_l2s_index = np.argmin(_l2s)
        _best_l2 = {
            'value': _l2s[_best_l2s_index],
            'match': _best_l2s_index
        }

        return _best_l2

    @cached_property
    def smooth_l1(self):

        diff = self.y_true - self.y_single_pred
        _l1s = np.linalg.norm(diff, ord=1, axis=-1)
        _l2s = np.linalg.norm(diff, ord=2, axis=-1)
        _criteria = _l1s < self.smooth_l1_beta

        _smooth_l1s = 0
        _smooth_l1s = _smooth_l1s + _criteria * 0.5 * _l2s / self.smooth_l1_beta
        _smooth_l1s = _smooth_l1s + (~_criteria) * (_l1s - 0.5 * self.smooth_l1_beta)

        _best_smooth_l1s_index = np.argmin(_smooth_l1s)
        _best_smooth_l1 = {
            'value': _smooth_l1s[_best_smooth_l1s_index],
            'match': _best_smooth_l1s_index
        }

        return _best_smooth_l1


class SpotsMetrics:

    def __init__(self, y_true, y_pred, **kwargs):

        self.y_true = y_true
        self.y_pred = y_pred

        self._spot_metrics = [
            SpotMetrics(y_true, y_single_pred, i, **kwargs) for i, y_single_pred in enumerate(y_pred)
        ]

    def calculate(self, agg_metric, match_metric, threshold):

        matches = [
            spot_metrics for spot_metrics in self._spot_metrics
            if getattr(spot_metrics, match_metric)['value'] < threshold
        ]
        match_ids = np.unique([getattr(match, match_metric)['match'] for match in matches])
        match_ids = match_ids[match_ids > 0]

        tp = len(match_ids)
        fp = len(self.y_pred) - tp
        fn = len(self.y_true) - len(match_ids)

        if agg_metric == 'f1':

            precision = tp / (tp + fp + 1e-07)
            recall = tp / (tp + fn + 1e-07)
            f1 = 2 * precision * recall / (precision + recall + 1e-07)

            return f1

        elif agg_metric == 'AP':

            ap = tp / (tp + fp + fn)

            return ap


class PixelMetrics:
    """Calculates pixel-based statistics.
    (Dice, Jaccard, Precision, Recall, F-measure)
    Takes in raw prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.
    Args:
        y_true (numpy.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (numpy.array): Binary predictions for a single feature,
            (batch,x,y)
    Raises:
        ValueError: Shapes of y_true and y_pred do not match.
    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for y_true and y_pred
    """

    def __init__(self, y_true, y_pred, id_true, id_pred):

        self.y_true = (y_true > 0).astype(int)
        self.y_pred = (y_pred > 0).astype(int)
        self.id_true = id_true
        self.id_pred = id_pred

        self._tp = np.count_nonzero(self.y_true & self.y_pred)
        self._fp = np.count_nonzero(~self.y_true & self.y_pred)
        self._tn = np.count_nonzero(~self.y_true & ~self.y_pred)
        self._fn = np.count_nonzero(self.y_true & ~self.y_pred)

        self._p = self._tp + self._fn
        self._pp = self._tp + self._fp

    @cached_property
    def recall(self):

        if self._p > 0:
            _recall = self._tp / self._p
        else:
            if self._fp == 0:
                _recall = 1.0
            else:
                _recall = 0.0

        return _recall

    @cached_property
    def precision(self):

        if self._pp > 0:
            _precision = self._tp / self._pp
        else:
            if self._fn == 0:
                _precision = 1.0
            else:
                _precision = 0.0

        return _precision

    @cached_property
    def f1(self):

        _recall = self.recall
        _precision = self.precision

        if (_recall == 0) & (_precision == 0):
            _f1 = 0.0
        else:
            _f1 = stats.hmean([_recall, _precision])

        return _f1

    @cached_property
    def jaccard(self):

        _union = self._tp + self._fp + self._fn

        if _union > 0:
            _jaccard = self._tp / _union
        else:
            _jaccard = 1.0

        return _jaccard

    def to_dict(self):
        return {
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'jaccard': self.jaccard
        }


class RegionMetrics:

    def __init__(self, y_true, y_pred, id_pred):

        self.y_true = y_true
        self.y_pred = y_pred
        self.id_pred = id_pred

        self._pixel_metrics = [
            PixelMetrics(y_true == region.label, y_pred, region.label, id_pred) for region in measure.regionprops(y_true)
        ]

    @cached_property
    def recall(self):

        _recalls = [pixel_metrics.recall for pixel_metrics in self._pixel_metrics]
        if len(_recalls) > 0:
            _best_recall_index = np.argmax(_recalls)
            _best_recall = {
                'value': _recalls[_best_recall_index],
                'match': self._pixel_metrics[_best_recall_index].id_true
            }
        else:
            _best_recall = {
                'value': 0,
                'match': 0
            }

        return _best_recall

    @cached_property
    def precision(self):

        _precisions = [pixel_metrics.precision for pixel_metrics in self._pixel_metrics]
        if len(_precisions) > 0:
            _best_precision_index = np.argmax(_precisions)
            _best_precision = {
                'value': _precisions[_best_precision_index],
                'match': self._pixel_metrics[_best_precision_index].id_true
            }
        else:
            _best_precision = {
                'value': 0,
                'match': 0
            }

        return _best_precision

    @cached_property
    def f1(self):

        _f1s = [pixel_metrics.f1 for pixel_metrics in self._pixel_metrics]
        if len(_f1s) > 0:
            _best_f1_index = np.argmax(_f1s)
            _best_f1 = {
                'value': _f1s[_best_f1_index],
                'match': self._pixel_metrics[_best_f1_index].id_true
            }
        else:
            _best_f1 = {
                'value': 0,
                'match': 0
            }

        return _best_f1

    @cached_property
    def jaccard(self):

        _jaccards = [pixel_metrics.jaccard for pixel_metrics in self._pixel_metrics]
        if len(_jaccards) > 0:
            _best_jaccard_index = np.argmax(_jaccards)
            _best_jaccard = {
                'value': _jaccards[_best_jaccard_index],
                'match': self._pixel_metrics[_best_jaccard_index].id_true
            }
        else:
            _best_jaccard = {
                'value': 0,
                'match': 0
            }

        return _best_jaccard


class MaskMetrics:

    def __init__(self, y_true, y_pred):

        self.y_true = measure.label(y_true)
        self.y_pred = measure.label(y_pred)

        self._region_metrics = [
            RegionMetrics(y_true[region.slice], region.image, region.label) for region in measure.regionprops(y_pred)
        ]

    def calculate(self, agg_metric, match_metric, threshold):

        if agg_metric == 'AP':

            matches = [
                region_metrics for region_metrics in self._region_metrics
                if getattr(region_metrics, match_metric)['value'] > threshold
            ]
            match_ids = np.unique([getattr(match, match_metric)['match'] for match in matches])
            match_ids = match_ids[match_ids > 0]

            tp = len(match_ids)
            fp = np.max(self.y_pred) - tp
            fn = np.max(self.y_true) - len(match_ids)

            ap = tp / (tp + fp + fn)

            return ap


def flow_error(maski, dP_net, use_gpu=False):
    """ error in flows from predicted masks vs flows predicted by network run on image
    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted
    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.
    Parameters
    ------------

    maski: ND-array (int)
        masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float)
        ND flows where dP_net.shape[1:] = maski.shape
    Returns
    ------------
    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks

    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return

    # flows predicted from estimated masks
    dP_masks = dynamics.masks_to_flows(maski, use_gpu=use_gpu)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += ndimage.mean((dP_masks[i] - dP_net[i] / 5.) ** 2, maski, index=np.arange(1, maski.max() + 1))

    return flow_errors, dP_masks
