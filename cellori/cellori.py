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
        
        self.image_std = np.std(self.image)

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

        output = masks,coords if segmentation_mode == 'masks' else coords

        return output

    def _segment(self,image,sigma,block_size,nuclei_diameter):

        coords,binary = self._find_nuclei(image,sigma,block_size,nuclei_diameter)
        masks = self._get_masks(binary,coords)

        return masks,coords

    def _find_nuclei(self,image,sigma,block_size,nuclei_diameter):

        image_blurred = filters.gaussian(image,sigma,preserve_range=True)
        adaptive_thresh = filters.threshold_local(image_blurred,block_size,method='mean',offset=-self.image_std / 10)
        binary = image_blurred > adaptive_thresh

        min_area = np.pi * (nuclei_diameter / 2) ** 2
        binary = morphology.remove_small_objects(binary,min_area)
        binary = morphology.remove_small_holes(binary)
        binary_labeled = morphology.label(binary)
        regions = measure.regionprops(binary_labeled,cache=False)

        coords = list()

        for region in regions:
            image_crop = image_blurred[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]]
            image_crop = np.where(region.image,image_crop,0)
            
            maxima = feature.peak_local_max(image_crop,min_distance=round(nuclei_diameter / 3))
            
            if len(maxima) == 0:
                if region.bbox[0] < region.centroid[0] < region.bbox[1] & region.bbox[2] < region.centroid[1] < region.bbox[3]:
                    coords.append(region.centroid)
            else:
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

    def _indices_to_xy(self,coords):
        
        coords[:,0] = self.image.shape[0] - coords[:,0]
        coords = np.fliplr(coords)

        return coords