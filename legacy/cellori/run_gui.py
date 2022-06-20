import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from cellori.imshowfast import imshow
from cellori.utils import _masks_to_outlines
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
from matplotlib.patches import Rectangle
from pathlib import Path
from PySide6 import QtWidgets
from skimage import exposure


def run_gui(cellori):
    def segment(event):

        def crop_coords(ax):

            viewlim = np.array([[ax.viewLim.y1, ax.viewLim.x0], [ax.viewLim.y0, ax.viewLim.x1]])
            cropped_coords = cellori.all_coords[
                np.all((viewlim[0] <= cellori.all_coords) & (cellori.all_coords <= viewlim[1]), axis=1)]
            cellori.segmentation_ax2.set_title(str(len(cropped_coords)) + " Cells")
            cellori.segmentation_fig.canvas.draw_idle()

        def save(event):

            save_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Coordinates", str(Path.cwd()),
                                                              "CSV (*.csv);; Text File (*.txt)")[0]

            if event.inaxes == cellori.ax_save_masks:
                save_data = cellori.masks
            elif event.inaxes == cellori.ax_save_xy:
                save_data = cellori._indices_to_xy(cellori.all_coords.copy())
            elif event.inaxes == cellori.ax_save_indices:
                save_data = cellori.all_coords

            np.savetxt(save_path, save_data, delimiter=',')

        cellori.masks, cellori.all_coords = cellori._segment(cellori.image, cellori.watershed_labeled,
                                                             parse_segmentation_mode(),
                                                             float(cellori.threshold_locality.text),
                                                             float(cellori.sigma.text),
                                                             float(cellori.nuclei_diameter.text))

        cellori.all_outlines = _masks_to_outlines(cellori.masks)
        cellori.all_outlines = np.where(cellori.all_outlines, cellori.all_outlines, np.nan)

        if cellori.segmentation_fig is None:

            cellori.segmentation_fig = plt.figure(figsize=(12, 6))
            cellori.segmentation_fig.canvas.manager.set_window_title('Segmentation Results')
            cellori.segmentation_ax1 = plt.subplot(1, 2, 1)
            cellori.segmentation_ax1.set_title("Original Image")
            cellori.segmentation_ax1.xaxis.set_visible(False)
            cellori.segmentation_ax1.yaxis.set_visible(False)
            cellori.segmentation_ax2 = plt.subplot(1, 2, 2, sharex=cellori.segmentation_ax1,
                                                   sharey=cellori.segmentation_ax1)
            cellori.segmentation_ax2.xaxis.set_visible(False)
            cellori.segmentation_ax2.yaxis.set_visible(False)
            cellori.segmentation_ax1.callbacks.connect('xlim_changed', crop_coords)
            cellori.segmentation_ax1.callbacks.connect('ylim_changed', crop_coords)
            cellori.segmentation_ax2.callbacks.connect('xlim_changed', crop_coords)
            cellori.segmentation_ax2.callbacks.connect('ylim_changed', crop_coords)
            cellori.segmentation_ax1_image = imshow(cellori.segmentation_ax1, cellori.image_adjusted, vmin=0, vmax=255,
                                                    cmap="gray")
            cellori.segmentation_ax2_image = imshow(cellori.segmentation_ax2, cellori.image_adjusted, vmin=0, vmax=255,
                                                    cmap="gray")
            cellori.segmentation_ax2_outlines = imshow(cellori.segmentation_ax2, cellori.all_outlines, cmap='winter')
            cellori.segmentation_viewlim = np.rot90(cellori.segmentation_ax2.viewLim.get_points().copy(), 2)
            cellori.segmentation_ax2.set_xlim(cellori.segmentation_viewlim[0])
            cellori.segmentation_ax2.set_ylim(cellori.segmentation_viewlim[1])

            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

            cellori.ax_save_masks = plt.axes([0.1, 0.025, 0.2, 0.05])
            cellori.save_masks_button = Button(cellori.ax_save_masks, 'Save Masks')
            cellori.save_masks_button.on_clicked(save)
            cellori.ax_save_xy = plt.axes([0.4, 0.025, 0.2, 0.05])
            cellori.save_xy_button = Button(cellori.ax_save_xy, 'Save XY Coordinates')
            cellori.save_xy_button.on_clicked(save)
            cellori.ax_save_indices = plt.axes([0.7, 0.025, 0.2, 0.05])
            cellori.save_indices_button = Button(cellori.ax_save_indices, 'Save Array Indices')
            cellori.save_indices_button.on_clicked(save)

        else:

            cellori.segmentation_ax1_image.set_data(cellori.image_adjusted)
            cellori.segmentation_ax2_image.set_data(cellori.image_adjusted)
            cellori.segmentation_ax2_outlines.set_data(cellori.all_outlines)

            if not np.array_equal(cellori.segmentation_ax2.viewLim.get_points(), cellori.segmentation_viewlim):
                cellori.segmentation_ax2.set_xlim(cellori.segmentation_viewlim[0])
                cellori.segmentation_ax2.set_ylim(cellori.segmentation_viewlim[1])

            crop_coords(cellori.segmentation_ax2)

        if len(cellori.segmentation_ax2.collections) > 0:
            cellori.segmentation_ax2.collections[-1].remove()
        if len(cellori.all_coords) > 0:
            cellori.segmentation_ax2.scatter(cellori.all_coords[:, 1], cellori.all_coords[:, 0], s=2, c='r')

        cellori.segmentation_fig.show()

    def update_segmentation():

        cellori.ax2.set_xlim(cellori.origin[0] - cellori.preview_size / 2, cellori.origin[0] + cellori.preview_size / 2)
        cellori.ax2.set_ylim(cellori.origin[1] + cellori.preview_size / 2, cellori.origin[1] - cellori.preview_size / 2)

        indices = np.array(
            [round(cellori.origin[1] - cellori.preview_size / 2), round(cellori.origin[1] + cellori.preview_size / 2),
             round(cellori.origin[0] - cellori.preview_size / 2), round(cellori.origin[0] + cellori.preview_size / 2)])
        watershed_labeled_crop = cellori.watershed_labeled[indices[0]:indices[1], indices[2]:indices[3]]
        crop_regions_ids = np.unique(watershed_labeled_crop)
        crop_regions_indices = crop_regions_ids[crop_regions_ids > 0] - 1
        crop_regions = [cellori.watershed_regions[i] for i in crop_regions_indices]
        crop_regions_bbox = np.array(
            [[region.bbox[0], region.bbox[2], region.bbox[1], region.bbox[3]] for region in crop_regions]).T
        adjusted_indices = np.array(
            [np.min(crop_regions_bbox[0]), np.max(crop_regions_bbox[1]), np.min(crop_regions_bbox[2]),
             np.max(crop_regions_bbox[3])])
        adjusted_indices = np.array([max(adjusted_indices[0], indices[0] - cellori.preview_size / 4),
                                     min(adjusted_indices[1], indices[1] + cellori.preview_size / 4),
                                     max(adjusted_indices[2], indices[2] - cellori.preview_size / 4),
                                     min(adjusted_indices[3], indices[3] + cellori.preview_size / 4)])
        adjusted_indices = cellori._calculate_edge_indices(adjusted_indices, cellori.image)
        adjusted_indices = np.rint(adjusted_indices).astype(int)
        offsets = np.abs(indices - adjusted_indices)

        image_crop = cellori.image[adjusted_indices[0]:adjusted_indices[1], adjusted_indices[2]:adjusted_indices[3]]
        watershed_labeled_crop = cellori.watershed_labeled[adjusted_indices[0]:adjusted_indices[1],
                                                           adjusted_indices[2]:adjusted_indices[3]]
        masks, coords = cellori._segment(image_crop, watershed_labeled_crop, parse_segmentation_mode(),
                                         float(cellori.threshold_locality.text), float(cellori.sigma.text),
                                         float(cellori.nuclei_diameter.text),
                                         (adjusted_indices[0], adjusted_indices[2]))
        outlines = _masks_to_outlines(
            masks[offsets[0]:masks.shape[0] - offsets[1], offsets[2]:masks.shape[1] - offsets[3]])
        outlines = np.where(outlines, outlines, np.nan)

        viewlim = np.array(
            [[offsets[0], offsets[2]], [offsets[0] + indices[1] - indices[0], offsets[2] + indices[3] - indices[2]]])
        if len(coords) > 0:
            coords = coords[np.all((viewlim[0] <= coords), axis=1) & np.all((coords <= viewlim[1]), axis=1)]

        cellori.ax2.set_title(str(len(coords)) + " Cells")

        extents = cellori.ax2.viewLim.extents
        if len(cellori.ax2.collections) > 0:
            cellori.ax2.collections[-1].remove()
        if len(coords) > 0:
            y, x = zip(*coords)
            x = np.add(x, extents[0] - offsets[2])
            y = np.add(y, extents[3] - offsets[0])
            cellori.ax2.scatter(x, y, s=1000 / cellori.preview_size, c='r')

        cellori.ax2_outlines.set(data=outlines, extent=(extents[0], extents[2], extents[1], extents[3]))

    def update_parameters(parameter):

        update_segmentation()
        cellori.fig.canvas.draw_idle()

    def update_contrast(n):

        cellori.global_thresh = cellori.image_mean + n * cellori.image_std
        cellori.image_adjusted = exposure.rescale_intensity(cellori.image, (0, cellori.global_thresh), (0, 255))
        cellori.ax1_image.set_data(cellori.image_adjusted)
        cellori.ax2_image.set_data(cellori.image_adjusted)

    def update_preview(n):

        cellori.preview_size = n
        check_origin()
        cellori.rect.set_bounds(cellori.origin[0] - n / 2, cellori.origin[1] - n / 2, n, n)
        update_segmentation()

    def update_viewlims():

        check_origin()
        cellori.rect.set_bounds(cellori.origin[0] - cellori.preview_size / 2,
                                cellori.origin[1] - cellori.preview_size / 2, cellori.preview_size,
                                cellori.preview_size)
        cellori.fig.canvas.draw_idle()
        update_segmentation()

    def on_click(event):

        if event.inaxes == cellori.ax1:
            cellori.origin = [event.xdata, event.ydata]
            update_viewlims()

    def on_press(event):

        if event.key in ['up', 'right', 'down', 'left']:

            if event.key == 'up':
                cellori.origin[1] -= 0.25 * cellori.preview_size
            elif event.key == 'right':
                cellori.origin[0] += 0.25 * cellori.preview_size
            elif event.key == 'down':
                cellori.origin[1] += 0.25 * cellori.preview_size
            elif event.key == 'left':
                cellori.origin[0] -= 0.25 * cellori.preview_size

            update_viewlims()

    def check_origin():

        if cellori.origin[0] - cellori.preview_size / 2 <= 0:
            cellori.origin[0] = cellori.preview_size / 2
        if cellori.origin[1] - cellori.preview_size / 2 <= 0:
            cellori.origin[1] = cellori.preview_size / 2
        if cellori.origin[0] + cellori.preview_size / 2 >= cellori.image.shape[1]:
            cellori.origin[0] = cellori.image.shape[1] - cellori.preview_size / 2
        if cellori.origin[1] + cellori.preview_size / 2 >= cellori.image.shape[0]:
            cellori.origin[1] = cellori.image.shape[0] - cellori.preview_size / 2

    def parse_segmentation_mode():

        if cellori.segmentation_mode.value_selected == 'Combined':
            segmentation_mode = 'combined'
        elif cellori.segmentation_mode.value_selected == 'Intensity':
            segmentation_mode = 'intensity'
        elif cellori.segmentation_mode.value_selected == 'Morphology':
            segmentation_mode = 'morphology'

        return segmentation_mode

    matplotlib.use('QtAgg')

    cellori.image_mean = np.mean(cellori.image)
    cellori.image_std = np.std(cellori.image)
    cellori.global_thresh = cellori.image_mean + 3 * cellori.image_std
    cellori.image_adjusted = exposure.rescale_intensity(cellori.image, (0, cellori.global_thresh), (0, 255))

    cellori.fig = plt.figure(figsize=(12, 6.75))
    cellori.fig.canvas.mpl_connect('button_press_event', on_click)
    cellori.fig.canvas.mpl_connect('key_press_event', on_press)
    cellori.fig.canvas.mpl_disconnect(cellori.fig.canvas.manager.key_press_handler_id)
    cellori.fig.canvas.manager.set_window_title('Cellori')
    cellori.ax1 = plt.subplot(1, 2, 1)
    cellori.ax1.xaxis.set_visible(False)
    cellori.ax1.yaxis.set_visible(False)
    cellori.ax1.set_title("Preview Region")
    cellori.ax2 = plt.subplot(1, 2, 2)
    cellori.ax2.xaxis.set_visible(False)
    cellori.ax2.yaxis.set_visible(False)

    cellori.ax1_image = imshow(cellori.ax1, cellori.image_adjusted, vmin=0, vmax=255, cmap="gray")
    cellori.ax2_image = imshow(cellori.ax2, cellori.image_adjusted, vmin=0, vmax=255, cmap="gray")
    cellori.ax2_outlines = cellori.ax2.imshow(np.zeros((1, 1)), cmap='winter', interpolation='none')

    plt.subplots_adjust(left=0.025, right=0.975, top=1, bottom=0.175)

    ax_segmentation_mode = plt.axes([0.075, 0.1, 0.1, 0.1])
    cellori.segmentation_mode = RadioButtons(ax_segmentation_mode, ('Combined', 'Intensity', 'Morphology'))
    cellori.segmentation_mode.on_clicked(update_parameters)
    ax_threshold_locality = plt.axes([0.325, 0.1, 0.1, 0.05])
    cellori.threshold_locality = TextBox(ax_threshold_locality, 'Threshold Locality ', initial=0.5)
    cellori.threshold_locality.on_submit(update_parameters)
    ax_sigma = plt.axes([0.575, 0.1, 0.1, 0.05])
    cellori.sigma = TextBox(ax_sigma, 'Gaussian Sigma ', initial=cellori.default_sigma)
    cellori.sigma.on_submit(update_parameters)
    ax_nuclei_diameter = plt.axes([0.825, 0.1, 0.1, 0.05])
    cellori.nuclei_diameter = TextBox(ax_nuclei_diameter, 'Nuclei Diameter ', initial=cellori.default_nuclei_diameter)
    cellori.nuclei_diameter.on_submit(update_parameters)

    ax_contrast = plt.axes([0.1, 0.0375, 0.2, 0.025])
    cellori.constrast_slider = Slider(
        ax=ax_contrast,
        label='Contrast',
        valmin=-20,
        valmax=20,
        valinit=3,
    )
    cellori.constrast_slider.on_changed(update_contrast)

    cellori.preview_size = min(round(0.25 * min(cellori.image.shape)), 500)
    ax_preview = plt.axes([0.45, 0.0375, 0.2, 0.025])
    cellori.preview_slider = Slider(
        ax=ax_preview,
        label='Preview Size',
        valmin=1,
        valmax=2 * cellori.preview_size - 1,
        valinit=cellori.preview_size,
    )
    cellori.preview_slider.on_changed(update_preview)

    ax_segment = plt.axes([0.8, 0.025, 0.1, 0.05])
    cellori.segment_button = Button(ax_segment, 'Segment')
    cellori.segment_button.on_clicked(segment)
    cellori.segmentation_fig = None

    cellori.origin = [cellori.image.shape[1] / 2, cellori.image.shape[0] / 2]
    cellori.rect = Rectangle(
        (cellori.origin[0] - cellori.preview_size / 2, cellori.origin[1] - cellori.preview_size / 2),
        cellori.preview_size, cellori.preview_size, facecolor='none', edgecolor='r', linewidth=1)
    cellori.ax1.add_patch(cellori.rect)
    cellori.ax2.set_xlim((cellori.image.shape[0] - cellori.preview_size) / 2,
                         (cellori.image.shape[0] + cellori.preview_size) / 2)
    cellori.ax2.set_ylim((cellori.image.shape[1] - cellori.preview_size) / 2,
                         (cellori.image.shape[1] + cellori.preview_size) / 2)

    update_segmentation()

    toolbar = cellori.fig.canvas.window().findChild(QtWidgets.QToolBar)
    toolbar.setVisible(False)

    plt.show()
