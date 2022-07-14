# Visual Inspection Demo

Example model for a visual inspection use case - detection of knots in images of wooden boards.

## Requirements

- The data set is available via [https://doi.org/10.5281/zenodo.4694694](https://doi.org/10.5281/zenodo.4694694)
- If using conda, make sure to run:
    - `conda install -c conda-forge cudatoolkit=11.3.1 cudnn=8.2.1`
    - `conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
- Install both `requirements.txt` and `requirements_dev.txt` of `pynavio`
