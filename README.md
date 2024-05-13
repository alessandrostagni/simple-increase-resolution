# Simple Increase Resolution

Simple Python script for converting all images contained in a folder using ESRGAN model.
Usually installing requirements is enough to have a minimal tensorflow setup but you might need to sort setting drivers and cuda by yourself.

## How to use

- Install requirements

`pip install -r requirements.txt`

- Run the script

`python [INPUT_FOLDER] [INCREASE_RES_FACTOR] [OUTPUT_FOLDER]`

The output folder will be created if it does not exist.

Example:

`python increase-resolution.py ~/Pictures/ 2 ~/Pictures-res-increased`

### Rescaling back

- If you need also the images with the new resolution, just add `--rescale` to the command:

`python increase-resolution.py ~/Pictures/ 2 ~/Pictures-res-increased --rescale`

A proper folder will be created if it doesn't exist for the rescaled images.