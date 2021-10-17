Cellori
===================================

**Cellori** (Cell Origin) is a fast and robust intensity-based algorithm for clustered nuclei segmentation in fluorescence microscopy images. It segments nuclei by applying a Gaussian filter to smoothen out background noise, calculating local thresholds to isolate the foreground, and splitting clustered nuclei via local maxima analysis. Masks are obtained using the watershed algorithm.

For a comprehensive walk-through, follow our :ref:`tutorial`.

Examples
--------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Raw Image
     - Segmented Image
   * - .. image:: ../examples/str_raw.png
           :width: 500
           :alt: Raw Image (STR)
     - .. image:: ../examples/str_segmented.png
           :width: 500
           :alt: Segmented Image (STR)
   * - .. image:: ../examples/wm989_raw.png
           :width: 500
           :alt: Raw Image (WM989)
     - .. image:: ../examples/wm989_segmented.png
           :width: 500
           :alt: Segmented Image (WM989)

Contents
--------

.. toctree::

   usage
   api
