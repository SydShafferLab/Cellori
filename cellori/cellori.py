import cv2 as cv
import numpy as np
import os

from skimage import feature,filters,measure,morphology,segmentation

class Cellori:

    """Cellori class object that takes the path to an image file or an image array.

    Parameters
    ----------
        image : str or numpy.ndarray 
            The path to an ND2 or TIFF file or a ``numpy.ndarray`` of an image that has already been loaded.
        nd2_overlap : float, optional, default 0.1
            The overlap percentage used by StitchWell for ND2 stitching. If ``None``, the value will be determined by automatic overlap calculation. This value is ignored if ``image`` is not the path to an ND2 file.
        nd2_stitch_channel : float, optional, default 0
            The index of the channel used by StitchWell for automatic overlap calculation during ND2 stitching. This value is ignored if automatic overlap calculation is not applicable.
        nuclei_channel : int, optional, default 0
            The index of the channel containing the nuclei for segmentation. This value is ignored if ``image`` has a single channel.

    Raises
    ------
        ValueError
            If ``image`` is an invalid image path or array.
        ValueError
            If ``nuclei_channel`` is not specified for an ``image`` with multiple channels.
        ValueError
            If ``image`` has invalid dimensions.
    """

    def __init__(self,image,**kwargs):
        
        if os.path.isfile(image):

            if image.endswith('.nd2'):

                from stitchwell import StitchWell
                nd2_overlap = kwargs.get('nd2_overlap',0.1)
                nd2_stitch_channel = kwargs.get('nd2_stitch_channel',0)
                self.image = StitchWell(image).stitch(0,nd2_overlap,nd2_stitch_channel)

            elif image.endswith(('.tif','.tiff')):

                from tifffile import imread
                self.image = imread(image)

        elif isinstance(image,np.ndarray):
            
            self.image = image

        else:

            raise ValueError("Invalid image.")

        if self.image.ndim != 2:
            if self.image.ndim == 3:
                    nuclei_channel = kwargs.get('nuclei_channel')
                    if nuclei_channel != None:
                        self.image = self.image[nuclei_channel]
                    else:
                        raise ValueError("Nuclei channel not specified.")
            else:
                raise ValueError("Invalid image dimensions.")

        self.image = self.image.astype(np.uint16)

        self.nan_mask = np.where(self.image == 0,True,False)
        self.exists_nan = np.any(self.nan_mask)
        if self.exists_nan:
            self.image = np.ma.masked_array(self.image,self.nan_mask)

        global_thresh = filters.threshold_otsu(self.image)
        if global_thresh > 0:
            self.foreground_mask = self.image > global_thresh
            background = np.ma.masked_array(self.image,self.foreground_mask)
            self.threshold_offset = np.std(background)
        else:
            self.threshold_offset = 0
            
    def gui(self,estimate_parameters=True):

        """Initiates the Cellori GUI.

        Parameters
        ----------
            estimate_parameters : bool, optional, default True
                Whether or not to run automatic parameter detection.   
        """

        from cellori.run_gui import run_gui

        if estimate_parameters:
            self._estimate_parameters()
        else:
            self.default_block_size = 15
            self.default_nuclei_diameter = 6
        
        run_gui(self)

    def segment(self,sigma=1.5,block_size=None,nuclei_diameter=None,segmentation_mode='masks',coordinate_format='indices'):

        """Segments the image using the Cellori algorithm.

        Parameters
        ----------
            sigma : float, optional, default 1.5
                Gaussian sigma used for background denoising.
            block_size : int, optional, default None
                Odd size of pixel neighborhood which is used for local thresholding (e.g., 3, 5, 7, ..., 21). If ``None``, the value will be determined by automatic parameter detection.
            nuclei_diameter : int, optional, default None
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be considered for segmentation. If ``None``, the value will be determined by automatic parameter detection.
            segmentation_mode : {'masks', 'coordinates'}, optional, default 'masks'
                * 'masks': Get masks using the watershed algorithm in addition to finding nuclei coordinates.
                * 'coordinates': Find the coordinates of nuclei centers only.
            coordinate_format : {'xy', 'indices'}, optional, default indices
                * 'xy': Format coordinates for plotting on standard XY axes.
                * 'indices': Format coordinates as indices of the original image array.
        
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ..., N. This array is only returned if the ``segmentation_mode`` is ``masks``.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.

        Raises
        ------
            ValueError
                If ``segmentation_mode`` is an invalid segmentation mode.
            ValueError
                If ``coordinate_format`` is an invalid coordinate format.
        """

        if segmentation_mode == 'masks':
            masks,coords = self._segment(self.image,sigma,block_size,nuclei_diameter)
        elif segmentation_mode == 'coordinates':
            coords,_ = self._find_nuclei(self.image,sigma,block_size,nuclei_diameter)
        else:
            raise ValueError("Invalid segmentation mode.")

        if coordinate_format =='xy':
            coords = self._indices_to_xy(coords)
        elif coordinate_format !='indices':
            raise ValueError("Invalid coordinate format.")

        output = (masks,coords) if segmentation_mode == 'masks' else coords

        return output

    def _segment(self,image,sigma,block_size,nuclei_diameter):

        """(For internal use) Get masks and nuclei coordinates using the Cellori algorithm.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be segmented.
            sigma : float
                Gaussian sigma used for background denoising.
            block_size : int
                Odd size of pixel neighborhood which is used for local thresholding (e.g., 3, 5, 7, ..., 21).
            nuclei_diameter : int
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be considered for segmentation.
            segmentation_mode : {'masks', 'coordinates'}
                * 'masks': Get masks using the watershed algorithm in addition to finding nuclei coordinates.
                * 'coordinates': Find the coordinates of nuclei centers only.
            coordinate_format : {'xy', 'indices'}
                * 'xy': Format coordinates for plotting on standard XY axes.
                * 'indices': Format coordinates as indices of the original image array.
        
        Returns
        -------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ..., N.
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
        """

        coords,binary = self._find_nuclei(image,sigma,block_size,nuclei_diameter)
        masks = self._get_masks(binary,coords)

        return masks,coords

    def _find_nuclei(self,image,sigma,block_size,nuclei_diameter,origin=None):

        """(For internal use) Find nuclei using the Cellori algorithm.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be segmented.
            sigma : float
                Gaussian sigma used for background denoising.
            block_size : int
                Odd size of pixel neighborhood which is used for local thresholding (e.g., 3, 5, 7, ..., 21).
            nuclei_diameter : int
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be considered for segmentation.
            coordinate_format : {'xy', 'indices'}
                * 'xy': Format coordinates for plotting on standard XY axes.
                * 'indices': Format coordinates as indices of the original image array.
            origin : tuple, optional, default None
                Origin coordinates of the GUI preview region.
        
        Returns
        -------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
            binary : numpy.ndarray
                Binarized array of the same size as the original image.
        """

        if block_size == None or nuclei_diameter == None:

            self._estimate_parameters()
            
            if block_size == None:
                block_size = self.default_block_size
            if nuclei_diameter == None:
                nuclei_diameter = self.default_nuclei_diameter

        image_blurry = filters.gaussian(image,sigma,preserve_range=True)
        image_blurrier = filters.gaussian(image,sigma * 2,preserve_range=True)
        adaptive_thresh = filters.threshold_local(image_blurrier,block_size,method='mean')
        binary = image_blurrier > adaptive_thresh + self.threshold_offset

        min_area = np.pi * (nuclei_diameter / 2) ** 2
        binary = morphology.remove_small_objects(binary,min_area)
        binary = morphology.remove_small_holes(binary)
        binary_labeled = morphology.label(binary)
        regions = measure.regionprops(binary_labeled,cache=False)

        coords = list()

        for region in regions:
            
            indices = [region.bbox[0],region.bbox[2],region.bbox[1],region.bbox[3]]

            if self.exists_nan:

                offset = int(((block_size) - 1) / 2)
                neighborhood_indices = [indices[0] - offset,indices[1] + offset,indices[2] - offset,indices[3] + offset]
                if origin != None:
                    neighborhood_indices = [neighborhood_indices[0] + origin[0],neighborhood_indices[1] + origin[0],neighborhood_indices[2] + origin[1],neighborhood_indices[3] + origin[1]]
                neighborhood_indices = self._calculate_edge_indices(neighborhood_indices)
                neighborhood_nan_mask = self.nan_mask[neighborhood_indices[0]:neighborhood_indices[1],neighborhood_indices[2]:neighborhood_indices[3]]

                if np.any(neighborhood_nan_mask):
                    continue

            image_crop = image_blurry[indices[0]:indices[1],indices[2]:indices[3]]
            image_crop = np.where(region.image,image_crop,0)

            maxima = feature.peak_local_max(image_crop,min_distance=round(nuclei_diameter / 3),exclude_border=False)
            
            for coord in maxima:
                coords.append((region.bbox[0] + coord[0],region.bbox[1] + coord[1]))
        
        coords = np.array(coords)

        return coords,binary

    def _get_masks(self,binary,coords):

        """(For internal use) Find nuclei using the Cellori algorithm.

        Parameters
        ----------
            image : numpy.ndarray
                Array of the image to be segmented.
            sigma : float
                Gaussian sigma used for background denoising.
            block_size : int
                Odd size of pixel neighborhood which is used for local thresholding (e.g., 3, 5, 7, ..., 21).
            nuclei_diameter : int
                Estimated lower bound of nuclei diameters. Any objects smaller than this threshold will not be considered for segmentation.
            coordinate_format : {'xy', 'indices'}
                * 'xy': Format coordinates for plotting on standard XY axes.
                * 'indices': Format coordinates as indices of the original image array.
            origin : tuple, optional, default None
                Origin coordinates of the GUI preview region.
        
        Returns
        -------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei.
            binary : numpy.ndarray
                Binarized array of the same size as the original image.
        """

        markers = np.zeros(binary.shape,dtype=bool)
        markers[tuple(np.rint(coords).astype(np.uint).T)] = True
        markers = morphology.label(markers)
        masks = segmentation.watershed(binary,markers,mask=binary)

        return masks

    def _masks_to_outlines(self,masks):

        """(For internal use) Convert masks to outlines for displaying GUI segmentation results.

        Parameters
        ----------
            masks : numpy.ndarray
                Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ..., N.
        
        Returns
        -------
            outlines : numpy.ndarray
                Array of the same size as the original image with outlines around each cell.
        """

        regions = measure.regionprops(masks,cache=False)

        outlines = np.zeros(masks.shape,dtype=bool)

        for region in regions:
            mask = region.image.astype(np.uint8)
            contours = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
            contours = np.concatenate(contours[0],axis=0).squeeze().T
            outlines[contours[1] + region.slice[0].start,contours[0] + region.slice[1].start] = 1
                
        return outlines

    def _estimate_parameters(self):

        """(For internal use) Estimate parameters for segmentation.
        """

        foreground_labeled = morphology.label(self.foreground_mask)
        regions = measure.regionprops(foreground_labeled,cache=False)
        
        d = np.array([region.equivalent_diameter for region in regions])
        self.default_nuclei_diameter = round(np.mean(d))
        self.default_block_size = 2 * self.default_nuclei_diameter + 1

    def _calculate_edge_indices(self,indices):

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
        if indices[1] > self.image.shape[0]:
            indices[1] = self.image.shape[0]
        if indices[2] < 0:
            indices[2] = 0
        if indices[3] > self.image.shape[1]:
            indices[3] = self.image.shape[1]

        return indices

    def _indices_to_xy(self,coords):

        """(For internal use) Convert array indices to XY coordinates.

        Parameters
        ----------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei formatted as indices of the original image array.
        
        Returns
        -------
            coords : numpy.ndarray
                Array of size (N, 2) with the coordinates of cell nuclei formatted for plotting on standard XY axes.
        """
        
        coords[:,0] = self.image.shape[0] - coords[:,0]
        coords = np.fliplr(coords)

        return coords