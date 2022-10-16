from setuptools import setup, find_packages

VERSION = "4.0"
DESCRIPTION = "Cellori"
LONG_DESCRIPTION = "A fast and robust algorithm for whole-cell segmentation."

setup(
    name="cellori",
    version=VERSION,
    author="William Niu",
    author_email="<wniu721@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['deeptile', 'einops', 'fastremap', 'flax', 'imageio', 'numba', 'numpy', 'opencv-python', 'pandas',
                      'scikit-image', 'scipy', 'tifffile', 'torch', 'tqdm'],
    include_package_data=True,
    keywords=["segmentation"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
