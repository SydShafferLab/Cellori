import cv2 as cv
import numpy as np
import SimpleITK as sitk

from cellori.netmap import get_touch_map
from pathlib import Path
from scipy import special
from skimage import feature, filters, measure, morphology, segmentation


class Cellori:
    """Cellori class object that takes the path to an image file or an image array.

    Parameters
    ----------
        image : str or numpy.ndarray
            The path to an ND2 or TIFF file or a ``numpy.ndarray`` of an image that has already been loaded.
        nd2_overlap : float, optional, default 0.1
            The overlap percentage used by StitchWell for ND2 stitching. If ``None``, the value will be determined by
            automatic overlap calculation. This value is ignored if ``image`` is not the path to an ND2 file.
        nuclei_channel : int, optional, default 0
            The index of the channel containing the nuclei for segmentation. This value is ignored if ``image`` has a
            single channel.

    Raises
    ------
        ValueError
            If ``image`` is an invalid image path or array.
        ValueError
            If ``nuclei_channel`` is not specified for an ``image`` with multiple channels.
        ValueError
            If ``image`` has invalid dimensions.
    """

    def __init__(self, image, **kwargs):

        self.all_coords = None
        self.default_nuclei_diameter = None
        self.default_sigma = None
        self.masks = None

        if isinstance(image, np.ndarray):

            self.image = image

        elif Path(image).is_file():

            if image.endswith('.nd2'):

                from stitchwell import StitchWell
                nd2_overlap = kwargs.get('nd2_overlap', 0.1)
                self.image = StitchWell(image).stitch(0, nd2_overlap)

            elif image.endswith(('.tif', '.tiff')):

                from tifffile import imread
                self.image = imread(image)

        else:

            raise ValueError("Invalid image.")

        if self.image.ndim != 2:
            if self.image.ndim == 3:
                nuclei_channel = kwargs.get('nuclei_channel')
                if nuclei_channel is not None:
                    self.image = self.image[nuclei_channel]
                else:
                    raise ValueError("Nuclei channel not specified.")
            else:
                raise ValueError("Invalid image dimensions.")

        global_thresh = filters.threshold_otsu(self.image[self.image > 0])
        self.global_binary = self.image > global_thresh
        background = np.ma.masked_array(self.image, self.global_binary)
        self.background_std = np.std(background)

        markers = measure.label(self.global_binary)
        watershed = sitk.MorphologicalWatershedFromMarkers(sitk.GetImageFromArray(np.zeros(self.image.shape)),
                                                           sitk.GetImageFromArray(markers), markWatershedLine=False)
        watershed = sitk.GetArrayFromImage(watershed)
        self.watershed_labeled = measure.label(watershed)
        self.watershed_regions = measure.regionprops(self.watershed_labeled, cache=False)

    def gui(self, estimate_parameters=True):

        """Initiates the Cellori GUI.

        Parameters
        ----------
            estimate_parameters : bool, optional, default True
                Whether or not to run automatic parameter detection.
            
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3,
                ..., N. The most recent segmentation result is returned, if any.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei. The most recent segmentation result is
                returned, if any.
            image : numpy.ndarray
                Array of the image for use in post-processing. The most recent segmentation result is returned, if any.
        """

        from cellori.run_gui import run_gui

        if estimate_parameters:
            self._estimate_parameters()
        else:
            self.default_sigma = 1
            self.default_nuclei_diameter = 6

        self.masks = None

        run_gui(self)

        if self.masks is not None:
            masks = self.masks
            coords = self.all_coords
            image = self.image

            return masks, coords, image

    def segment(self, segmentation_mode='combined', threshold_locality=0.5, sigma=None, nuclei_diameter=None,
                coordinate_format='indices'):

        """Segments the image using the Cellori algorithm.

        Parameters
        ----------
            segmentation_mode : {'combined', 'intensity', 'morphology'}, optional, default 'combined'
                * ‘combined’: Use a combined maxima metric that incorporates both intensity and morphology.
                * ‘intensity’: Use an intensity-only maxima metric.
                * ‘morphology’: Use a morphology-only maxima metric.
            threshold_locality : float, optional, default 0.5
                Fractional weight on local intensity used in thresholding. The value must be between 0 (global
                thresholding) and 1 (local thresholding).
            sigma : float, optional, default 1.5
                Gaussian sigma used for denoising. If ``None``, the value will be determined by automatic parameter
                detection.
            nuclei_diameter : int, optional, default None
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be
                considered for segmentation. If ``None``, the value will be determined by automatic parameter detection.
            coordinate_format : {'xy', 'indices'}, optional, default 'indices'
                * ‘xy’: Format coordinates for plotting on standard XY axes.
                * ‘indices’: Format coordinates as indices of the original image array.
        
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3,
                ..., N.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
            image : numpy.ndarray
                Array of the image for use in post-processing.

        Raises
        ------
            ValueError
                If ``segmentation_mode`` is an invalid segmentation mode.
            ValueError
                If ``coordinate_format`` is an invalid coordinate format.
        """

        if segmentation_mode not in ['combined', 'intensity', 'morphology']:
            raise ValueError("Invalid segmentation mode.")

        if (sigma is None) or (nuclei_diameter is None):

            self._estimate_parameters()

            if sigma is None:
                sigma = self.default_sigma
            if nuclei_diameter is None:
                nuclei_diameter = self.default_nuclei_diameter

        masks, coords = self._segment(self.image, self.watershed_labeled, segmentation_mode, threshold_locality, sigma,
                                      nuclei_diameter)

        if coordinate_format == 'xy':
            coords = self._indices_to_xy(coords)
        elif coordinate_format != 'indices':
            raise ValueError("Invalid coordinate format.")

        image = self.image

        return masks, coords, image

    def _segment(self, image, watershed_labeled, segmentation_mode, threshold_locality, sigma, nuclei_diameter,
                 origin=None):

        """(For internal use) Get masks and nuclei coordinates using the Cellori algorithm.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be segmented.
            watershed_labeled : numpy.ndarray
                Array of labeled watershed regions.
            segmentation_mode : {'combined', 'intensity', 'morphology'}
                * ‘combined’: Use a combined maxima metric that incorporates both intensity and morphology.
                * ‘intensity’: Use an intensity-only maxima metric.
                * ‘morphology’: Use a morphology-only maxima metric.
            threshold_locality : float
                Fractional weight on local intensity used in thresholding. The value must be between 0 (global
                thresholding) and 1 (local thresholding).
            sigma : float
                Gaussian sigma used for denoising. If ``None``, the value will be determined by automatic parameter
                detection.
            nuclei_diameter : int
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be
                considered for segmentation. If ``None``, the value will be determined by automatic parameter detection.
            origin : tuple, optional, default None
                Origin coordinates of the GUI preview region.
        
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3,
                ..., N.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
        """

        self.min_area = np.pi * (nuclei_diameter / 2) ** 2

        coords, binary = self._find_nuclei(image, watershed_labeled, segmentation_mode, sigma, threshold_locality,
                                           nuclei_diameter, origin)
        masks = self._get_masks(binary, coords)
        masks, coords = self._merge_correct(masks)

        return masks, coords

    def _find_nuclei(self, image, watershed_labeled, segmentation_mode, sigma, threshold_locality, nuclei_diameter,
                     origin):

        """(For internal use) Find nuclei using the Cellori algorithm.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be segmented.
            watershed_labeled : numpy.ndarray
                Array of labeled watershed regions.
            segmentation_mode : {'combined', 'intensity', 'morphology'}
                * ‘combined’: Use a combined maxima metric that incorporates both intensity and morphology.
                * ‘intensity’: Use an intensity-only maxima metric.
                * ‘morphology’: Use a morphology-only maxima metric.
            threshold_locality : float
                Fractional weight on local intensity used in thresholding. The value must be between 0 (global
                thresholding) and 1 (local thresholding).
            sigma : float
                Gaussian sigma used for denoising. If ``None``, the value will be determined by automatic parameter
                detection.
            nuclei_diameter : int
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be
                considered for segmentation. If ``None``, the value will be determined by automatic parameter detection.
            origin : tuple
                Origin coordinates of the GUI preview region.
        
        Returns
        -------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
            binary : numpy.ndarray
                Binarized array of the same size as the original image.
        """

        block_size = 2 * round(nuclei_diameter) + 1

        binary = np.zeros(image.shape, dtype=bool)
        if origin is None:
            watershed_regions = self.watershed_regions
            origin = (0, 0)
        else:
            watershed_regions = measure.regionprops(watershed_labeled, cache=False)

        for region in watershed_regions:

            indices = [region.bbox[0], region.bbox[2], region.bbox[1], region.bbox[3]]
            image_crop = image[indices[0]:indices[1], indices[2]:indices[3]]
            global_binary_crop = self.global_binary[indices[0] + origin[0]:indices[1] + origin[0],
                                                    indices[2] + origin[1]:indices[3] + origin[1]]

            binary_crop = self._conditional_local_threshold(image_crop, region.image, global_binary_crop, block_size,
                                                            threshold_locality, self.background_std)
            binary_crop = np.where(region.image, binary_crop, 0)
            binary_current = binary[indices[0]:indices[1], indices[2]:indices[3]]
            binary[indices[0]:indices[1], indices[2]:indices[3]] = np.where(binary_crop, True, binary_current)

        binary = morphology.remove_small_objects(binary, self.min_area)
        binary = morphology.remove_small_holes(binary, self.min_area)
        binary_labeled = morphology.label(binary)
        regions = measure.regionprops(binary_labeled, cache=False)

        coords = np.empty(0)

        for region in regions:

            indices = [region.bbox[0], region.bbox[2], region.bbox[1], region.bbox[3]]

            if segmentation_mode == 'combined' or segmentation_mode == 'intensity':
                image_crop = self.image[indices[0] + origin[0]:indices[1] + origin[0],
                                        indices[2] + origin[1]:indices[3] + origin[1]]
                image_crop = np.where(region.image, image_crop, 0)
            if segmentation_mode == 'combined' or segmentation_mode == 'morphology':
                binary_crop = binary[indices[0]:indices[1], indices[2]:indices[3]]
                binary_distance = cv.distanceTransform(binary_crop.astype(np.uint8), cv.DIST_L2, 0)

            if segmentation_mode == 'combined':

                binary_distance = self._normalize(binary_distance)
                image_crop = self._normalize(image_crop)
                metric = image_crop + binary_distance

            elif segmentation_mode == 'intensity':
                metric = image_crop
            elif segmentation_mode == 'morphology':
                metric = binary_distance

            metric = filters.gaussian(metric, sigma, preserve_range=True)
            maxima = feature.peak_local_max(metric, min_distance=round(nuclei_diameter / 4), threshold_rel=0.5,
                                            exclude_border=False)

            coords = np.append(coords, maxima + region.bbox[:2])

        coords = np.reshape(coords, (-1, 2))

        return coords, binary

    @staticmethod
    def _get_masks(binary, coords):

        """(For internal use) Get masks using the watershed algorithm.

        Parameters
        ----------
            binary : numpy.ndarray
                Binarized array of the same size as the original image.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.

        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3,
                ..., N.
        """

        markers = np.zeros(binary.shape, dtype=bool)
        markers[tuple(np.rint(coords).astype(np.uint).T)] = True
        markers = measure.label(markers)
        masks = segmentation.watershed(binary, markers, mask=binary)

        return masks

    def _merge_correct(self, masks):

        """(For internal use) Correct for oversegmentation via rule-based region merging.

        Parameters
        ----------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3,
                ..., N.
        
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the image after region merging corrections.
        """

        masks = measure.label(masks)
        regions = measure.regionprops(masks, cache=False)

        idx = get_touch_map(masks)
        cells = np.unique(list(idx.keys()))

        if len(cells) > 0:

            corrected_masks = masks.copy()
            merged_cells = list()
            cell_map = {cell: cell for cell in cells}

            for cell in cells:

                merge_candidates = [merge_candidate for merge_candidate in idx[cell] if
                                    merge_candidate not in merged_cells]
                merged_cells.append(cell)

                if len(merge_candidates) == 0:
                    continue

                cell_region = regions[cell - 1]
                cell_indices = np.array(
                    [cell_region.bbox[0], cell_region.bbox[2], cell_region.bbox[1], cell_region.bbox[3]])

                for merge_candidate in merge_candidates:

                    merge_candidate_region = regions[merge_candidate - 1]
                    merge_candidate_indices = np.array(
                        [merge_candidate_region.bbox[0], merge_candidate_region.bbox[2], merge_candidate_region.bbox[1],
                         merge_candidate_region.bbox[3]])

                    merge_indices = np.array([min(cell_indices[0], merge_candidate_indices[0]),
                                              max(cell_indices[1], merge_candidate_indices[1]),
                                              min(cell_indices[2], merge_candidate_indices[2]),
                                              max(cell_indices[3], merge_candidate_indices[3])])
                    merge_crop = masks[merge_indices[0]:merge_indices[1], merge_indices[2]:merge_indices[3]]
                    merge_test = np.where((merge_crop == cell) | (merge_crop == merge_candidate), 1, 0)

                    merge_region = measure.regionprops(merge_test, cache=False)[0]
                    average_solidity = (cell_region.solidity + merge_candidate_region.solidity) / 2
                    if (4 * cell_region.area < self.min_area) | (4 * merge_candidate_region.area < self.min_area) | (
                            (merge_region.area < 4 * self.min_area) & (
                            merge_region.solidity > 0.975 * average_solidity)):
                        new_cell = min(cell_map[cell], cell_map[merge_candidate])
                        corrected_masks[merge_indices[0]:merge_indices[1], merge_indices[2]:merge_indices[3]][
                            merge_test > 0] = new_cell
                        cell_map[merge_candidate] = new_cell

            masks = measure.label(corrected_masks)
            regions = measure.regionprops(masks)

        coords = np.array([region.centroid for region in regions])

        return masks, coords

    @staticmethod
    def _conditional_local_threshold(image, mask, global_binary, block_size, k_max, c):

        """(For internal use) Calculate conditional local threshold.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be thresholded.
            mask : numpy.ndarray
                Array marking the region to be thresholding.
            global_binary : numpy.ndarray
                Binarized array of the image using global thresholding.
            block_size : int
                Odd size of pixel neighborhood which is used for local thresholding (e.g., 3, 5, 7, ..., 21).
            k_max : float
                Maximum fractional weight on local intensity used in thresholding. The value must be between 0 (global
                thresholding) and 1 (local thresholding).
            c : float
                Global constant subtracted from local threshold array.

        Returns
        -------
            threshold : numpy.ndarray
                Array of local threshold values. All pixels in the input image higher than the corresponding pixel in
                the threshold array are considered foreground.
        """

        image_masked = np.float64(image) * mask
        background = image_masked * np.invert(global_binary)
        mask = np.float64(mask)

        image_blurred = cv.blur(image_masked, (block_size, block_size))
        mask = cv.blur(mask, (block_size, block_size))

        smooth = image_blurred / (mask + 1e-15)

        threshold = image_masked > smooth + c

        if (k_max > 0) & (np.sum(background) > 0):
            k = k_max * (1 - np.sqrt(np.sum(threshold) / np.sum(mask)))
            bg_std = np.std(background[background > 0])
            offset = k * bg_std + (1 - k) * c
            threshold = image_masked > smooth + offset

        threshold = threshold * mask

        return threshold

    def _estimate_parameters(self):

        """(For internal use) Estimate parameters for segmentation.
        """

        global_binary = cv.morphologyEx(self.global_binary.astype(np.uint8), cv.MORPH_ERODE, np.ones((3, 3)))
        global_binary = segmentation.clear_border(global_binary)
        foreground_labeled = measure.label(global_binary)
        regions = measure.regionprops(foreground_labeled, cache=False)

        equivalent_diameters = np.array([region.equivalent_diameter for region in regions])
        self.default_nuclei_diameter = np.around(np.median(equivalent_diameters), 2)

        n = round(self.default_nuclei_diameter / 4)
        self.default_sigma = np.around(2 ** (2 * n) / (special.comb(2 * n, n) * np.sqrt(2 * np.pi)), 2)

    def _indices_to_xy(self, coords):

        """(For internal use) Convert array indices to XY coordinates.

        Parameters
        ----------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei formatted as indices of the original image
                array.
        
        Returns
        -------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei formatted for plotting on standard XY axes.
        """

        coords[:, 0] = self.image.shape[0] - coords[:, 0]
        coords = np.fliplr(coords)

        return coords

    @staticmethod
    def _calculate_edge_indices(indices, image):

        """(For internal use) Calculate indices for slicing near the edges of an array.

        Parameters
        ----------
            indices : list
                Indices for array slicing.
        
        Returns
        -------
            outlines : list
                Adjusted indices for array slicing near the edges.
        """

        if indices[0] < 0:
            indices[0] = 0
        if indices[1] > image.shape[0]:
            indices[1] = image.shape[0]
        if indices[2] < 0:
            indices[2] = 0
        if indices[3] > image.shape[1]:
            indices[3] = image.shape[1]

        return indices

    @staticmethod
    def _normalize(array):

        """(For internal use) Normalize an array.

        Parameters
        ----------
            array : numpy.ndarray
                Array to be normalized.
        
        Returns
        -------
            array_normalized : numpy.ndarray
                Normalized array.
        """

        array_min = np.min(array)
        array_max = np.max(array)
        array_normalized = (array - array_min) / (array_max - array_min)

        return array_normalized
