Cellori
===================================

**Cellori** (Cell Origin) is a fast and robust algorithm for clustered nuclei segmentation in fluorescence microscopy images. It segments nuclei by calculating local thresholds to isolate the foreground, splitting clustered nuclei via local maxima analysis, and then merge correcting regions to account for potential oversegmentation. Masks are obtained using the watershed algorithm.

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
