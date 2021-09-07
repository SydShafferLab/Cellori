import numpy as np
import os

from skimage import exposure,feature,filters,measure,morphology

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

    def gui(self):

        def count(event):
            
            def crop_counts(ax):
                
                cropped_coords = [coord for coord in self.all_coords if ax.viewLim.x0 < coord[1] < ax.viewLim.x1 and ax.viewLim.y1 < coord[0] < ax.viewLim.y0]
                ax.set_title(str(len(cropped_coords)) + " Cells")

            def save(event):
                
                save_path = QtWidgets.QFileDialog.getSaveFileName(None,"Save Coordinates",os.getcwd(),"CSV (*.csv);; Text File (*.txt)")[0]
                np.savetxt(save_path,self.all_coords,delimiter=',')

            self.all_coords = self._count(self.image,float(self.sigma.text),int(self.block_size.text),float(self.nuclei_diameter.text))

            if self.count_fig == None:

                self.count_fig = plt.figure(figsize=(12,6))
                self.count_fig.canvas.manager.set_window_title('Count Results')
                self.count_ax1 = plt.subplot(1,2,1)
                self.count_ax1.set_title("Original Image")
                self.count_ax1.xaxis.set_visible(False)
                self.count_ax1.yaxis.set_visible(False)
                self.count_ax2 = plt.subplot(1,2,2,sharex=self.count_ax1,sharey=self.count_ax1)
                self.count_ax2.xaxis.set_visible(False)
                self.count_ax2.yaxis.set_visible(False)
                self.count_ax2.callbacks.connect('xlim_changed',crop_counts)
                self.count_ax2.callbacks.connect('ylim_changed',crop_counts)
                self.count_ax1_image = imshow(self.count_ax1,self.image_adjusted,vmin=0,vmax=255,cmap="gray")
                self.count_ax2_image = imshow(self.count_ax2,self.image_adjusted,vmin=0,vmax=255,cmap="gray")
                self.count_viewlim = np.rot90(self.count_ax2.viewLim.get_points().copy(),2)

                plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.1)

                self.ax_save = plt.axes([0.4,0.025,0.2,0.05])
                self.save_button = Button(self.ax_save,'Save Coordinates')
                self.save_button.on_clicked(save)

            else:

                self.count_ax1_image.set_data(self.image_adjusted)
                self.count_ax2_image.set_data(self.image_adjusted)

                if not np.array_equal(self.count_ax2.viewLim.get_points(),self.count_viewlim):
                    
                    self.count_ax2.set_xlim(self.count_viewlim[0])
                    self.count_ax2.set_ylim(self.count_viewlim[1])

            if len(self.count_ax2.collections) > 0:
                    self.count_ax2.collections[-1].remove()
            if len(self.all_coords) > 0:
                y,x = zip(*self.all_coords)
                self.count_ax2.scatter(x,y,s=3,c='r')

            self.count_fig.canvas.draw_idle()
            self.count_fig.show()

        def update_count():

            self.ax2.set_xlim(self.origin[0] - self.preview_size / 2,self.origin[0] + self.preview_size / 2)
            self.ax2.set_ylim(self.origin[1] + self.preview_size / 2,self.origin[1] - self.preview_size / 2)

            offset = int((int(self.block_size.text) - 1) / 2)
            offsets = [offset] * 4

            x_low,x_high,y_low,y_high = round(self.origin[1] - self.preview_size / 2) - offset,round(self.origin[1] + self.preview_size / 2) + offset,round(self.origin[0] - self.preview_size / 2) - offset,round(self.origin[0] + self.preview_size / 2) + offset

            if x_low < 0:
                x_low += offset
                offsets[0] = 0
            if x_high > self.image.shape[0]:
                x_high -= offset
                offsets[1] = 0
            if y_low < 0:
                y_low += offset
                offsets[2] = 0
            if y_high > self.image.shape[1]:
                y_high -= offset
                offsets[3] = 0

            image_crop = self.image[x_low:x_high,y_low:y_high].T
            coords = self._count(image_crop,float(self.sigma.text),int(self.block_size.text),float(self.nuclei_diameter.text))
            coords = [coord for coord in coords if offsets[0] < coord[1] < x_high - x_low - offsets[1] and offsets[2] < coord[0] < y_high - y_low - offsets[3]]

            self.ax2.set_title(str(len(coords)) + " Cells")
            if len(self.ax2.collections) > 0:
                    self.ax2.collections[-1].remove()
            if len(coords) > 0:
                y,x = zip(*coords)
                x = np.add(x,self.origin[1] - self.preview_size / 2 - offsets[0])
                y = np.add(y,self.origin[0] - self.preview_size / 2 - offsets[2])
                self.ax2.scatter(y,x,s=3,c='r')

        def update_parameters(parameter):

            update_count()
            self.fig.canvas.draw_idle()
        
        def update_contrast(n):
            
            self.global_thresh = self.image_mean + n * self.image_std
            self.image_adjusted = exposure.rescale_intensity(self.image,(0,self.global_thresh),(0,255))
            self.ax1_image.set_data(self.image_adjusted)
            self.ax2_image.set_data(self.image_adjusted)

        def update_preview(n):

            self.preview_size = n
            check_origin()
            self.rect.set_bounds(self.origin[0] - n / 2,self.origin[1] - n / 2,n,n)
            update_count()

        def update_viewlims():

            check_origin()
            self.rect.set_bounds(self.origin[0] - self.preview_size / 2,self.origin[1] - self.preview_size / 2,self.preview_size,self.preview_size)
            self.fig.canvas.draw_idle()
            update_count()

        def on_click(event):

            if event.inaxes == self.ax1:
                self.origin = [event.xdata,event.ydata]
                update_viewlims()

        def on_press(event):

            if event.key in ['up','right','down','left']:

                if event.key == 'up':
                    self.origin[1] -= 0.25 * self.preview_size
                elif event.key == 'right':
                    self.origin[0] += 0.25 * self.preview_size
                elif event.key == 'down':
                    self.origin[1] += 0.25 * self.preview_size
                elif event.key == 'left':
                    self.origin[0] -= 0.25 * self.preview_size

                update_viewlims()

        def check_origin():

            if self.origin[0] - self.preview_size / 2 < 1:
                self.origin[0] = self.preview_size / 2 + 1
            if self.origin[1] - self.preview_size / 2 < 1:
                self.origin[1] = self.preview_size / 2 + 1
            if self.origin[0] + self.preview_size / 2 > self.image.shape[1] - 1:
                self.origin[0] = self.image.shape[1] - self.preview_size / 2 - 1
            if self.origin[1] + self.preview_size / 2 > self.image.shape[0] - 1:
                self.origin[1] = self.image.shape[0] - self.preview_size / 2 - 1

        import matplotlib
        import matplotlib.pyplot as plt

        from cellori.imshowfast import imshow
        from matplotlib.widgets import Button,Slider,TextBox
        from matplotlib.patches import Rectangle
        from PyQt5 import QtWidgets

        matplotlib.use('Qt5Agg')

        self.image_mean = np.mean(self.image)
        self.image_std = np.std(self.image)
        self.global_thresh = self.image_mean + 3 * self.image_std
        self.image_adjusted = exposure.rescale_intensity(self.image,(0,self.global_thresh),(0,255))

        self.fig = plt.figure(figsize=(12,6.5))
        self.fig.canvas.mpl_connect('button_press_event',on_click)
        self.fig.canvas.mpl_connect('key_press_event',on_press)
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.manager.set_window_title('Cellori')
        self.ax1 = plt.subplot(1,2,1)
        self.ax1.xaxis.set_visible(False)
        self.ax1.yaxis.set_visible(False)
        self.ax1.set_title("Preview Region")
        self.ax2 = plt.subplot(1,2,2)
        self.ax2.xaxis.set_visible(False)
        self.ax2.yaxis.set_visible(False)

        self.ax1_image = imshow(self.ax1,self.image_adjusted,vmin=0,vmax=255,cmap="gray")
        self.ax2_image = imshow(self.ax2,self.image_adjusted,vmin=0,vmax=255,cmap="gray")

        plt.subplots_adjust(left=0.025,right=0.975,top=1,bottom=0.15)

        ax_sigma = plt.axes([0.10,0.1,0.15,0.05])
        self.sigma = TextBox(ax_sigma,'Sigma',initial='2')
        self.sigma.on_submit(update_parameters)
        ax_block_size = plt.axes([0.425,0.1,0.15,0.05])
        self.block_size = TextBox(ax_block_size,'Block Size',initial='7')
        self.block_size.on_submit(update_parameters)
        ax_nuclei_diameter = plt.axes([0.75,0.1,0.15,0.05])
        self.nuclei_diameter = TextBox(ax_nuclei_diameter,'Nuclei Diameter',initial='6')
        self.nuclei_diameter.on_submit(update_parameters)

        ax_contrast = plt.axes([0.1,0.0375,0.2,0.025])
        self.constrast_slider = Slider(
            ax=ax_contrast,
            label='Contrast',
            valmin=-20,
            valmax=20,
            valinit=3,
        )
        self.constrast_slider.on_changed(update_contrast)

        self.preview_size = min(round(0.25 * min(self.image.shape)),500)
        ax_preview = plt.axes([0.45,0.0375,0.2,0.025])
        self.preview_slider = Slider(
            ax=ax_preview,
            label='Preview Size',
            valmin=1,
            valmax=2 * self.preview_size - 1,
            valinit=self.preview_size,
        )
        self.preview_slider.on_changed(update_preview)

        ax_count = plt.axes([0.8,0.025,0.1,0.05])
        self.count_button = Button(ax_count,'Count')
        self.count_button.on_clicked(count)
        self.count_fig = None

        self.origin = [self.image.shape[1] / 2,self.image.shape[0] / 2]
        self.rect = Rectangle((self.origin[0] - self.preview_size / 2,self.origin[1] - self.preview_size / 2),self.preview_size,self.preview_size,facecolor='none',edgecolor='r',linewidth=1)
        self.ax1.add_patch(self.rect)
        self.ax2.set_xlim((self.image.shape[0] - self.preview_size) / 2,(self.image.shape[0] + self.preview_size) / 2)
        self.ax2.set_ylim((self.image.shape[1] - self.preview_size) / 2,(self.image.shape[1] + self.preview_size) / 2)

        update_count()
        
        toolbar = self.fig.canvas.window().findChild(QtWidgets.QToolBar)
        toolbar.setVisible(False)

        plt.show()

    def get_coordinates(self,sigma,block_size,nuclei_diameter):
        
        coords = self._count(self.image,sigma,block_size,nuclei_diameter)
        coords = np.array(coords)

        return coords

    def save_coordinates(self,path,sigma,block_size,nuclei_diameter):
        
        coords = self.get_coordinates(self,sigma,block_size,nuclei_diameter)
        
        np.savetxt(path,coords,delimiter=',')

    def _count(self,image,sigma,block_size,nuclei_diameter):

        image_blurred = filters.gaussian(image,sigma,preserve_range=True)
        adaptive_thresh = filters.threshold_local(image_blurred,block_size,offset=-10)
        binary = image_blurred > adaptive_thresh

        min_area = np.pi * (nuclei_diameter / 2) ** 2
        binary = morphology.remove_small_objects(binary,min_area)
        binary_labeled = morphology.label(binary)
        regions = measure.regionprops(binary_labeled,cache=False)

        coords = list()

        for region in regions:
            image_crop = image_blurred[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]]
            image_crop = np.where(region.image,image_crop,0)
            
            maxima = feature.peak_local_max(image_crop,min_distance=round(nuclei_diameter / 2))
            
            if len(maxima) == 0:
                coords.append(region.centroid)
            else:
                for coord in maxima:
                    coords.append((region.bbox[0] + coord[0],region.bbox[1] + coord[1]))
        
        return coords
