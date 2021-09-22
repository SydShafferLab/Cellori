import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from cellori.imshowfast import imshow
from matplotlib.widgets import Button,Slider,TextBox
from matplotlib.patches import Rectangle
from PyQt5 import QtWidgets
from skimage import exposure

def run_gui(Cellori):

    def segment(event):

        def crop_coords(ax):

            viewlim = np.array([[ax.viewLim.y1,ax.viewLim.x0],[ax.viewLim.y0,ax.viewLim.x1]])
            cropped_coords = Cellori.all_coords[np.all((viewlim[0] <= Cellori.all_coords) & (Cellori.all_coords <= viewlim[1]),axis=1)]
            Cellori.segmentation_ax2.set_title(str(len(cropped_coords)) + " Cells")
            Cellori.segmentation_fig.canvas.draw_idle()

        def save(event):
            
            save_path = QtWidgets.QFileDialog.getSaveFileName(None,"Save Coordinates",os.getcwd(),"CSV (*.csv);; Text File (*.txt)")[0]

            if event.inaxes == Cellori.ax_save_masks:
                save_data = Cellori.masks
            elif event.inaxes == Cellori.ax_save_xy:
                save_data = Cellori._indices_to_xy(Cellori.all_coords.copy)
            elif event.inaxes == Cellori.ax_save_indices:
                save_data = Cellori.all_coords
            
            np.savetxt(save_path,save_data,delimiter=',')

        Cellori.masks,Cellori.all_coords = Cellori._segment(Cellori.image,float(Cellori.sigma.text),int(Cellori.block_size.text),float(Cellori.nuclei_diameter.text))

        Cellori.outlines = Cellori._masks_to_outlines(Cellori.masks)
        Cellori.outlines = np.where(Cellori.outlines,Cellori.outlines,np.nan)

        if Cellori.segmentation_fig == None:

            Cellori.segmentation_fig = plt.figure(figsize=(12,6))
            Cellori.segmentation_fig.canvas.manager.set_window_title('Count Results')
            Cellori.segmentation_ax1 = plt.subplot(1,2,1)
            Cellori.segmentation_ax1.set_title("Original Image")
            Cellori.segmentation_ax1.xaxis.set_visible(False)
            Cellori.segmentation_ax1.yaxis.set_visible(False)
            Cellori.segmentation_ax2 = plt.subplot(1,2,2,sharex=Cellori.segmentation_ax1,sharey=Cellori.segmentation_ax1)
            Cellori.segmentation_ax2.xaxis.set_visible(False)
            Cellori.segmentation_ax2.yaxis.set_visible(False)
            Cellori.segmentation_ax1.callbacks.connect('xlim_changed',crop_coords)
            Cellori.segmentation_ax1.callbacks.connect('ylim_changed',crop_coords)
            Cellori.segmentation_ax2.callbacks.connect('xlim_changed',crop_coords)
            Cellori.segmentation_ax2.callbacks.connect('ylim_changed',crop_coords)
            Cellori.segmentation_ax1_image = imshow(Cellori.segmentation_ax1,Cellori.image_adjusted,vmin=0,vmax=255,cmap="gray")
            Cellori.segmentation_ax2_image = imshow(Cellori.segmentation_ax2,Cellori.image_adjusted,vmin=0,vmax=255,cmap="gray")
            Cellori.segmentation_ax2_outlines = imshow(Cellori.segmentation_ax2,Cellori.outlines,cmap='winter')
            Cellori.segmentation_viewlim = np.rot90(Cellori.segmentation_ax2.viewLim.get_points().copy(),2)
            Cellori.segmentation_ax2.set_xlim(Cellori.segmentation_viewlim[0])
            Cellori.segmentation_ax2.set_ylim(Cellori.segmentation_viewlim[1])

            plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.1)

            Cellori.ax_save_masks = plt.axes([0.1,0.025,0.2,0.05])
            Cellori.save_masks_button = Button(Cellori.ax_save_masks,'Save Masks')
            Cellori.save_masks_button.on_clicked(save)
            Cellori.ax_save_xy = plt.axes([0.4,0.025,0.2,0.05])
            Cellori.save_xy_button = Button(Cellori.ax_save_xy,'Save XY Coordinates')
            Cellori.save_xy_button.on_clicked(save)
            Cellori.ax_save_indices = plt.axes([0.7,0.025,0.2,0.05])
            Cellori.save_indices_button = Button(Cellori.ax_save_indices,'Save Array Indices')
            Cellori.save_indices_button.on_clicked(save)

        else:

            Cellori.segmentation_ax1_image.set_data(Cellori.image_adjusted)
            Cellori.segmentation_ax2_image.set_data(Cellori.image_adjusted)
            Cellori.segmentation_ax2_outlines.set_data(Cellori.outlines)

            if not np.array_equal(Cellori.segmentation_ax2.viewLim.get_points(),Cellori.segmentation_viewlim):
                
                Cellori.segmentation_ax2.set_xlim(Cellori.segmentation_viewlim[0])
                Cellori.segmentation_ax2.set_ylim(Cellori.segmentation_viewlim[1])
                
            crop_coords(Cellori.segmentation_ax2)

        if len(Cellori.segmentation_ax2.collections) > 0:
                Cellori.segmentation_ax2.collections[-1].remove()
        if len(Cellori.all_coords) > 0:
            Cellori.segmentation_ax2.scatter(Cellori.all_coords[:,1],Cellori.all_coords[:,0],s=3,c='r')

        Cellori.segmentation_fig.show()

    def update_segmentation():

        Cellori.ax2.set_xlim(Cellori.origin[0] - Cellori.preview_size / 2,Cellori.origin[0] + Cellori.preview_size / 2)
        Cellori.ax2.set_ylim(Cellori.origin[1] + Cellori.preview_size / 2,Cellori.origin[1] - Cellori.preview_size / 2)

        offset = int((int(Cellori.block_size.text) - 1) / 2)
        indices = np.array([round(Cellori.origin[1] - Cellori.preview_size / 2) - offset,round(Cellori.origin[1] + Cellori.preview_size / 2) + offset,round(Cellori.origin[0] - Cellori.preview_size / 2) - offset,round(Cellori.origin[0] + Cellori.preview_size / 2) + offset])
        adjusted_indices = Cellori._calculate_edge_indices(indices.copy())
        offsets = np.array([offset] * 4) - np.abs(indices - adjusted_indices)

        image_crop = Cellori.image[adjusted_indices[0]:adjusted_indices[1],adjusted_indices[2]:adjusted_indices[3]]
        coords,_ = Cellori._find_nuclei(image_crop,float(Cellori.sigma.text),int(Cellori.block_size.text),float(Cellori.nuclei_diameter.text),(adjusted_indices[0],adjusted_indices[2]))

        viewlim = np.array([[offsets[0],offsets[2]],[adjusted_indices[1] - adjusted_indices[0] - offsets[1],adjusted_indices[3] - adjusted_indices[2] - offsets[3]]])

        if len(coords) > 0:
            coords = coords[np.all((viewlim[0] <= coords),axis=1) & np.all((coords <= viewlim[1]),axis=1)]

        Cellori.ax2.set_title(str(len(coords)) + " Cells")

        if len(Cellori.ax2.collections) > 0:
                Cellori.ax2.collections[-1].remove()
        if len(coords) > 0:
            y,x = zip(*coords)
            x = np.add(x,Cellori.origin[0] - Cellori.preview_size / 2 - offsets[2])
            y = np.add(y,Cellori.origin[1] - Cellori.preview_size / 2 - offsets[0])
            Cellori.ax2.scatter(x,y,s=3,c='r')

    def update_parameters(parameter):

        update_segmentation()
        Cellori.fig.canvas.draw_idle()

    def update_contrast(n):
        
        Cellori.global_thresh = Cellori.image_mean + n * Cellori.image_std
        Cellori.image_adjusted = exposure.rescale_intensity(Cellori.image,(0,Cellori.global_thresh),(0,255))
        Cellori.ax1_image.set_data(Cellori.image_adjusted)
        Cellori.ax2_image.set_data(Cellori.image_adjusted)

    def update_preview(n):

        Cellori.preview_size = n
        check_origin()
        Cellori.rect.set_bounds(Cellori.origin[0] - n / 2,Cellori.origin[1] - n / 2,n,n)
        update_segmentation()

    def update_viewlims():

        check_origin()
        Cellori.rect.set_bounds(Cellori.origin[0] - Cellori.preview_size / 2,Cellori.origin[1] - Cellori.preview_size / 2,Cellori.preview_size,Cellori.preview_size)
        Cellori.fig.canvas.draw_idle()
        update_segmentation()

    def on_click(event):

        if event.inaxes == Cellori.ax1:
            Cellori.origin = [event.xdata,event.ydata]
            update_viewlims()

    def on_press(event):

        if event.key in ['up','right','down','left']:

            if event.key == 'up':
                Cellori.origin[1] -= 0.25 * Cellori.preview_size
            elif event.key == 'right':
                Cellori.origin[0] += 0.25 * Cellori.preview_size
            elif event.key == 'down':
                Cellori.origin[1] += 0.25 * Cellori.preview_size
            elif event.key == 'left':
                Cellori.origin[0] -= 0.25 * Cellori.preview_size

            update_viewlims()

    def check_origin():

        if Cellori.origin[0] - Cellori.preview_size / 2 <= 0:
            Cellori.origin[0] = Cellori.preview_size / 2 - 1
        if Cellori.origin[1] - Cellori.preview_size / 2 <= 0:
            Cellori.origin[1] = Cellori.preview_size / 2 - 1
        if Cellori.origin[0] + Cellori.preview_size / 2 >= Cellori.image.shape[1]:
            Cellori.origin[0] = Cellori.image.shape[1] - Cellori.preview_size / 2 - 0.5
        if Cellori.origin[1] + Cellori.preview_size / 2 >= Cellori.image.shape[0]:
            Cellori.origin[1] = Cellori.image.shape[0] - Cellori.preview_size / 2 - 0.5

    matplotlib.use('Qt5Agg')

    Cellori.image_mean = np.mean(Cellori.image)
    Cellori.image_std = np.std(Cellori.image)
    Cellori.global_thresh = Cellori.image_mean + 3 * Cellori.image_std
    Cellori.image_adjusted = exposure.rescale_intensity(Cellori.image,(0,Cellori.global_thresh),(0,255))

    Cellori.fig = plt.figure(figsize=(12,6.5))
    Cellori.fig.canvas.mpl_connect('button_press_event',on_click)
    Cellori.fig.canvas.mpl_connect('key_press_event',on_press)
    Cellori.fig.canvas.mpl_disconnect(Cellori.fig.canvas.manager.key_press_handler_id)
    Cellori.fig.canvas.manager.set_window_title('Cellori')
    Cellori.ax1 = plt.subplot(1,2,1)
    Cellori.ax1.xaxis.set_visible(False)
    Cellori.ax1.yaxis.set_visible(False)
    Cellori.ax1.set_title("Preview Region")
    Cellori.ax2 = plt.subplot(1,2,2)
    Cellori.ax2.xaxis.set_visible(False)
    Cellori.ax2.yaxis.set_visible(False)

    Cellori.ax1_image = imshow(Cellori.ax1,Cellori.image_adjusted,vmin=0,vmax=255,cmap="gray")
    Cellori.ax2_image = imshow(Cellori.ax2,Cellori.image_adjusted,vmin=0,vmax=255,cmap="gray")

    plt.subplots_adjust(left=0.025,right=0.975,top=1,bottom=0.15)

    ax_sigma = plt.axes([0.10,0.1,0.15,0.05])
    Cellori.sigma = TextBox(ax_sigma,'Sigma',initial='2')
    Cellori.sigma.on_submit(update_parameters)
    ax_block_size = plt.axes([0.425,0.1,0.15,0.05])
    Cellori.block_size = TextBox(ax_block_size,'Block Size',initial='7')
    Cellori.block_size.on_submit(update_parameters)
    ax_nuclei_diameter = plt.axes([0.75,0.1,0.15,0.05])
    Cellori.nuclei_diameter = TextBox(ax_nuclei_diameter,'Nuclei Diameter',initial='6')
    Cellori.nuclei_diameter.on_submit(update_parameters)

    ax_contrast = plt.axes([0.1,0.0375,0.2,0.025])
    Cellori.constrast_slider = Slider(
        ax=ax_contrast,
        label='Contrast',
        valmin=-20,
        valmax=20,
        valinit=3,
    )
    Cellori.constrast_slider.on_changed(update_contrast)

    Cellori.preview_size = min(round(0.25 * min(Cellori.image.shape)),500)
    ax_preview = plt.axes([0.45,0.0375,0.2,0.025])
    Cellori.preview_slider = Slider(
        ax=ax_preview,
        label='Preview Size',
        valmin=1,
        valmax=2 * Cellori.preview_size - 1,
        valinit=Cellori.preview_size,
    )
    Cellori.preview_slider.on_changed(update_preview)

    ax_segment = plt.axes([0.8,0.025,0.1,0.05])
    Cellori.segment_button = Button(ax_segment,'Count')
    Cellori.segment_button.on_clicked(segment)
    Cellori.segmentation_fig = None

    Cellori.origin = [Cellori.image.shape[1] / 2,Cellori.image.shape[0] / 2]
    Cellori.rect = Rectangle((Cellori.origin[0] - Cellori.preview_size / 2,Cellori.origin[1] - Cellori.preview_size / 2),Cellori.preview_size,Cellori.preview_size,facecolor='none',edgecolor='r',linewidth=1)
    Cellori.ax1.add_patch(Cellori.rect)
    Cellori.ax2.set_xlim((Cellori.image.shape[0] - Cellori.preview_size) / 2,(Cellori.image.shape[0] + Cellori.preview_size) / 2)
    Cellori.ax2.set_ylim((Cellori.image.shape[1] - Cellori.preview_size) / 2,(Cellori.image.shape[1] + Cellori.preview_size) / 2)

    update_segmentation()

    toolbar = Cellori.fig.canvas.window().findChild(QtWidgets.QToolBar)
    toolbar.setVisible(False)

    plt.show()