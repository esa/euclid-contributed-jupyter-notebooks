README
======
The program 'exptime_objects.py' computes object specific exposure time and the number of imaging layers for the Euclid filters of all OU-MER DR1 data. The necessary input data for the tile(s) given as input is located automatically on the ESA data space. The program creates a copy of the FINAL catalog adding the columns "EXPTIME_<band>" and "NLAYER_<band>" for each desired Euclid band. As an option a FITS exposure time maps is computed for each band as well.

All quantities are computed on the basis of the layer files that are part of DR1 and contain the corner coordinates of all VIS and NIR input quadrants/detectors that were used for the coadded images.

Note that:
* as an intermetidate step exposure time and nlayers are determined per healpix index, enhancing the healpix order parameter leads to a higher spatial resolution (but also needs more processing time). The default order=17 has a resolution of 1.6"
* the exposure time and nlayers are computed for the objects positions provided in the FINAL catalog;
* missing pixels due to persistence or cosmics or pixel defects are not taken into account;

Usage
-----
usage: python exptime_objects.py [-h] (--tiles TILE[,TILE,...] | --tile-file FILE) [--band BAND] [--output DIR] [--chunk-height N] [--write-exptime-maps] [--healpix-order N]

Create exposure time maps and annotate MER catalogs with objects specific exposure time and # of layers for Euclid DR1 tiles.

options:
  -h, --help            show this help message and exit
  --tiles TILE[,TILE,...]
                        Single tile ID or comma-separated list of tile IDs.
  --tile-file FILE      Text file with one tile ID per line.
  --band BAND           Process a single band only. Choices: VIS, NIR_Y, NIR_J, NIR_H. Default: all four bands.
  --output DIR          Output directory (default: exptime_output).
  --chunk-height N      Row chunk size for exposure map creation; tune for memory (default: 10000).
  --write-exptime-maps  Write per-pixel exposure time map FITS files (skipped by default).
  --healpix-order N     HEALPix order for exposure map interpolation. Defaults to the mosaic's MOCORDER.
                        Lower values are faster but coarser (default:17).

1.) Installing dependencies
First of all we need to install a couple of packages that are needed:

> pip install -r requirements.txt

2.) Running the code

To run the code simply type

> python exptime_objects.py <TILE_ID> 

and it will start creating the exposure time maps + adding the exposure time to the final catalogs for each band (VIS + NIR by default)

Example:

> python exptime_objects.py --tiles 102052063

You can also specify a list of tiles

> python exptime_objects.py --tiles 102052063,102052064

or specify a band 

> python exptime_objects.py --tiles 102052063,102052064 --band NIR_Y

There are a few other options in the python script, so feel free to check them out by typing

> python exptime_objects.py --h