import cv2 as cv
import numpy as np
import os

from skimage import feature,filters,measure,morphology,segmentation

class Cellori:

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
            
            if self.image.ndim == 3:
                
                nuclei_channel = kwargs.get('nuclei_channel')
                self.image = self.image[nuclei_channel]

        elif isinstance(image,np.ndarray):
            
            self.image = image

        self.image = self.image.astype(np.uint16)

        self.nan_mask = np.where(self.image == 0,True,False)
        self.exists_nan = np.any(self.nan_mask)
        if self.exists_nan:
            self.image = np.ma.masked_array(self.image,self.nan_mask)

        global_thresh = filters.threshold_otsu(self.image)
        if global_thresh > 0:
            foreground_mask = self.image > global_thresh
            background = np.ma.masked_array(self.image,foreground_mask)
            self.threshold_offset = np.std(background)
        else:
            self.threshold_offset = 0

    def gui(self):

        from cellori.run_gui import run_gui
        
        run_gui(self)

    def segment(self,sigma=2,block_size=7,nuclei_diameter=6,segmentation_mode='masks',coordinate_format='indices'):

        if segmentation_mode == 'masks':
            masks,coords = self._segment(self.image,sigma,block_size,nuclei_diameter)
        elif segmentation_mode == 'coordinates':
            coords,_ = self._find_nuclei(self.image,sigma,block_size,nuclei_diameter)
        else:
            print("Invalid segmentation mode.")
            exit()

        if coordinate_format =='xy':
            coords = self._indices_to_xy(coords)
        elif coordinate_format !='indices':
            print("Invalid coordinate format.")
            exit()

        output = (masks,coords) if segmentation_mode == 'masks' else coords

        return output

    def _segment(self,image,sigma,block_size,nuclei_diameter):

        coords,binary = self._find_nuclei(image,sigma,block_size,nuclei_diameter)
        masks = self._get_masks(binary,coords)

        return masks,coords

    def _find_nuclei(self,image,sigma,block_size,nuclei_diameter,origin=None):

        image_blurred = filters.gaussian(image,sigma,preserve_range=True)
        adaptive_thresh = filters.threshold_local(image_blurred,block_size,method='mean')
        binary = image_blurred > adaptive_thresh + self.threshold_offset

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

            image_crop = image_blurred[indices[0]:indices[1],indices[2]:indices[3]]
            image_crop = np.where(region.image,image_crop,0)
            
            maxima = feature.peak_local_max(image_crop,min_distance=round(nuclei_diameter / 3),exclude_border=False)
            
            for coord in maxima:
                coords.append((region.bbox[0] + coord[0],region.bbox[1] + coord[1]))
        
        coords = np.array(coords)

        return coords,binary

    def _get_masks(self,binary,coords):

        markers = np.zeros(binary.shape,dtype=bool)
        markers[tuple(np.rint(coords).astype(np.uint).T)] = True
        markers = morphology.label(markers)
        masks = segmentation.watershed(binary,markers,mask=binary)

        return masks

    def _masks_to_outlines(self,masks):

        regions = measure.regionprops(masks,cache=False)

        outlines = np.zeros(masks.shape,dtype=bool)

        for region in regions:
            sr,sc = region.slice
            mask = region.image.astype(np.uint8)
            contours = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
            pvc,pvr = np.concatenate(contours[0],axis=0).squeeze().T            
            vr,vc = pvr + sr.start,pvc + sc.start 
            outlines[vr,vc] = 1
                
        return outlines

    def _calculate_edge_indices(self,indices):

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
        
        coords[:,0] = self.image.shape[0] - coords[:,0]
        coords = np.fliplr(coords)

        return coords