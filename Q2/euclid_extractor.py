#!/usr/bin/env python
"""
Euclid VIS Cutout Extractor

Version 1 of the tool **EuclidExtractor** was written by Manon Gilles (Exoplanet Team, IAP, France), 
with inputs from Henry Joy McCracken, Clotilde Laigle, the help of the VIS team, and the Exoplanet Team of IAP.

Efficiently extract postage stamps from Euclid VIS data at given RA/DEC coordinates.
Creates a spatial index database for fast coordinate-to-extension mapping.

PNG output is guaranteed to follow the astronomical convention:
North is up, East is left.

This file is subject to the terms and conditions defined in file LICENSE.txt, which is part of this source code package. No part of the package, including this file, may be copied, modified, propagated, or distributed except according to the terms contained in the file LICENSE.txt.
"""

import sqlite3
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import re
import pandas as pd
import glob

def normalize_extname(extname_raw):
    """Return the form 'R-C.Q' (e.g., '1-1.E') from '1-1.E.SCI' or variants."""
    s = str(extname_raw).strip()
    # try a direct regex extraction (number-number.letter)
    m = re.search(r'^(\d+-\d+\.[A-Z])', s, flags=re.I)
    if m:
        return m.group(1).upper()
    # fallback : remove suffix .SCI (case insensitive) then clean
    s2 = re.sub(r'\.sci$', '', s, flags=re.I).strip()
    return s2.rstrip('. ').upper()

def extract_exp_from_filename(match):
    """Extract exposure number from filename in match dict."""
    exp = re.search(r'EUC_VIS_SWL-DET-067070-(\d+)-', match['filename'])
    if not exp:
        return None
    return int(exp.group(1))

class EuclidPSFExtractor:
    """Extract PSF models from Euclid .psf files."""
    def __init__(self, psf_file):
        self.psf_file = psf_file

    def read_psf_file(self, quad=None, ij=(0.5, 0.5),
                      p_radec=False, p_quad=True, p_all=False):
        """Get PSF model from .psf file."""

        f = fits.open(self.psf_file)
        tab = f[1].data

        n_pix = np.shape(tab[0][0][0])[0]

        psf_final_radec = None
        psf_final_quad  = None
        psf_final_all   = None

        if p_all is True or quad is None:
            quad_start = 1
            quad_end = len(f)
            psf_tab_all = np.zeros([len(f)-1, n_pix, n_pix])
            psf_all = np.zeros([n_pix, n_pix])

            for l in range(quad_start, quad_end):
                tab = f[l].data
                psf_ct = tab[0,][0][0, :, :]
                psf_x  = tab[0,][0][1, :, :]
                psf_x2 = tab[0,][0][2, :, :]
                psf_y  = tab[0,][0][3, :, :]
                psf_xy = tab[0,][0][4, :, :]
                psf_y2 = tab[0,][0][5, :, :]

                nsteps = 10
                for i in range(nsteps):
                    for j in range(nsteps):
                        x = ((i+0.5)*f[l].header['POLSCAL1']/nsteps - f[l].header['POLZERO1']) / f[l].header['POLSCAL1']
                        y = ((j+0.5)*f[l].header['POLSCAL2']/nsteps - f[l].header['POLZERO2']) / f[l].header['POLSCAL2']
                        psf_all = (
                            psf_ct + x*psf_x + x**2*psf_x2 +
                            y*psf_y + y**2*psf_y2 + x*y*psf_xy
                        )
                        psf_tab_all[l-quad_start, :, :] += psf_all/nsteps**2
            
            psf_final_all = np.nanmean(psf_tab_all, axis=0)

        if p_radec is True or p_quad is True:
            quad_start = quad
            quad_end = quad + 1
            psf_final_quad = np.zeros([n_pix, n_pix])
            psf_quad = np.zeros([n_pix, n_pix])
            psf_final_radec = np.zeros([n_pix, n_pix])

            tab = f[quad].data
            hdr = f[quad].header
            psf_ct = tab[0,][0][0, :, :]
            psf_x  = tab[0,][0][1, :, :]
            psf_x2 = tab[0,][0][2, :, :]
            psf_y  = tab[0,][0][3, :, :]
            psf_xy = tab[0,][0][4, :, :]
            psf_y2 = tab[0,][0][5, :, :]

            if p_radec is True:
                # x = (ij[0]*f[l].header['POLSCAL1'] - f[l].header['POLZERO1']) / f[l].header['POLSCAL1']
                # y = (ij[1]*f[l].header['POLSCAL2'] - f[l].header['POLZERO2']) / f[l].header['POLSCAL2']
                x = (ij[0]*hdr['POLSCAL1'] - hdr['POLZERO1']) / hdr['POLSCAL1']
                y = (ij[1]*hdr['POLSCAL2'] - hdr['POLZERO2']) / hdr['POLSCAL2']
                psf_final_radec = (
                    psf_ct + x*psf_x + x**2*psf_x2 +
                    y*psf_y + y**2*psf_y2 + x*y*psf_xy
                )

            if p_quad is True:
                nsteps = 10
                for i in range(nsteps):
                    for j in range(nsteps):
                        # x = ((i+0.5)*f[l].header['POLSCAL1']/nsteps - f[l].header['POLZERO1']) / f[l].header['POLSCAL1']
                        # y = ((j+0.5)*f[l].header['POLSCAL2']/nsteps - f[l].header['POLZERO2']) / f[l].header['POLSCAL2']
                        x = (ij[0]*hdr['POLSCAL1'] - hdr['POLZERO1']) / hdr['POLSCAL1']
                        y = (ij[1]*hdr['POLSCAL2'] - hdr['POLZERO2']) / hdr['POLSCAL2']
                        psf_quad = (
                            psf_ct + x*psf_x + x**2*psf_x2 +
                            y*psf_y + y**2*psf_y2 + x*y*psf_xy
                        )
                        psf_final_quad += psf_quad/nsteps**2


        return psf_final_radec, psf_final_quad, psf_final_all

    def psf_from_match(self, match, RA=None, DEC=None,
                       p_radec=False, p_quad=True, p_all=False):
        """Get PSF model for a given match and RA/DEC."""

        extname = normalize_extname(match['extension'])
        quad_idx = None

        if RA is not None or DEC is not None:
            with fits.open(self.psf_file) as pf:
                for idx in range(1, len(pf)):
                    hdr = pf[idx].header
                    if hdr.get('HIERARCH VISEXTNAME') == extname:
                        quad_idx = idx
                        break

        if quad_idx is None:
            return None, None, ij

        if RA is not None and DEC is not None and quad_idx is not None:
            with fits.open(match['filename']) as f:
                hdr = f[quad_idx].header
                wcs = WCS(hdr)
            x_pixel, y_pixel = wcs.world_to_pixel_values(RA, DEC)
            ij = (x_pixel % 1, y_pixel % 1)
        else:
            ij = (0.5, 0.5)

        psf_radec, psf_quad, psf_all = self.read_psf_file(
            quad=quad_idx,
            ij=ij,
            p_radec=p_radec,
            p_quad=p_quad,
            p_all=p_all
        )

        return psf_radec, psf_quad, psf_all, quad_idx, x_pixel, y_pixel

    def psfs_for_matches(self, matches, RA=None, DEC=None,
                         p_radec=False, p_quad=True, p_all=False,
                         prefix="cutout", output_dir="cutouts"):
        """Get PSF models for a list of extensions."""

        results = []
        rows = []

        for m in matches:
            exp = extract_exp_from_filename(m)
            try:
                psf_radec, psf_quad, psf_all, quad_idx, x_pixel, y_pixel = self.psf_from_match(
                    m, RA, DEC, p_radec, p_quad, p_all
                )
                if psf_radec is None and psf_quad is None and psf_all is None:
                    print(f"No matching .psf HDU for {normalize_extname(m['extension'])}")
                    continue

                ext = normalize_extname(m.get('extension'))
                ccdid = ext.split('.')[0]
                quad  = ext.split('.')[1]

                tag = f"{prefix}_ra{RA:.6f}_dec{DEC:.6f}_ccd{ccdid}_quad{quad}_{exp:02d}_sci.fits"

                rows.append({
                    'filename': tag,
                    'extension': ext,
                    'quad': quad_idx,
                    'psf ra/dec': psf_radec,
                    'psf quadrant': psf_quad,
                    'psf all quadrants': psf_all,
                    'x position': float(f"{x_pixel:.4f}"),
                    'y position': float(f"{y_pixel:.4f}"),
                })

            except Exception as e:
                print('Error for match', m, '->', e)

        df = pd.DataFrame(rows)
        return df


def process_file_with_progress(filename):
    """Wrapper for multiprocessing – prints progress for each file."""
    results = EuclidCutoutExtractor._process_file_static(filename)
    print(f"Processed: {Path(filename).name} ({len(results)} extensions)")
    return results


class EuclidCutoutExtractor:
    def __init__(self, data_dir, db_path="euclid_index.db", rebuild_index=False):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.rebuild_index = rebuild_index

        # if rebuild_index or not os.path.exists(db_path):
        #     print("Building spatial index…")
        #     self._build_spatial_index()
        # else:
        #     print(f"Using existing index: {db_path}")
        if rebuild_index:
            if os.path.exists(db_path):
                print(f"Removing existing index: {db_path}")
                os.remove(db_path)
            print("Building spatial index…")
            self._build_spatial_index()
        elif not os.path.exists(db_path):
            print("Building spatial index…")
            self._build_spatial_index()
        else:
            print(f"Using existing index: {db_path}")


    # ---------------------------------------------------------------------
    # Index-building helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _process_file_static(filename):
        """Return a list of SCI extensions with bounding boxes for one FITS file."""
        results = []
        try:
            with fits.open(filename) as hdul:
                for hdu in hdul[1:]:                       # skip primary
                    if hdu.name.endswith('.SCI'):
                        bounds = EuclidCutoutExtractor._get_wcs_bounds_static(
                            filename, hdu.name
                        )
                        if bounds:
                            results.append({
                                'filename': str(filename),
                                'extension': hdu.name,
                                **bounds
                            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        return results

    @staticmethod
    def _get_wcs_bounds_static(filename, ext_name):
        """Return RA/Dec bounds and image size for one extension."""
        try:
            with fits.open(filename) as hdul:
                ext_hdu = next((h for h in hdul if h.name == ext_name), None)
                if ext_hdu is None:
                    return None

                wcs = WCS(ext_hdu.header)
                if wcs.naxis < 2:
                    return None

                ny, nx = ext_hdu.data.shape
                corners_pix = np.array([[0, 0], [nx - 1, 0],
                                        [nx - 1, ny - 1], [0, ny - 1]])

                corners_world = wcs.pixel_to_world_values(
                    corners_pix[:, 0], corners_pix[:, 1]
                )

                # Astropy can return either:
                #  - tuple of arrays (ra, dec)
                #  - Nx2 array-like
                if isinstance(corners_world, tuple):
                    ra, dec = corners_world
                else:
                    ra = corners_world[:, 0]
                    dec = corners_world[:, 1]

                # Handle RA wrap-around at 0°
                if np.any(ra > 300) and np.any(ra < 60):
                    ra = np.where(ra > 180, ra - 360, ra)

                return {
                    'ra_min': float(np.min(ra)),
                    'ra_max': float(np.max(ra)),
                    'dec_min': float(np.min(dec)),
                    'dec_max': float(np.max(dec)),
                    'nx': int(nx),
                    'ny': int(ny)
                }
        except Exception as e:
            print(f"Error processing {filename}[{ext_name}]: {e}")
            return None

    # ---------------------------------------------------------------------
    # Spatial-index construction
    # ---------------------------------------------------------------------
    def _build_spatial_index(self):
        sci_files = list(self.data_dir.glob("*_sci.fits"))
        print(f"Found {len(sci_files)} SCI files")

        if not sci_files:
            raise ValueError(f"No *_sci.fits files in {self.data_dir}")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS extensions (
              id       INTEGER PRIMARY KEY,
              filename TEXT,
              extension TEXT,
              ra_min REAL, ra_max REAL,
              dec_min REAL, dec_max REAL,
              nx INTEGER, ny INTEGER
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_spatial "
            "ON extensions (ra_min, ra_max, dec_min, dec_max)"
        )

        print("Processing files in parallel…")
        n_proc = min(8, cpu_count())
        print(f"Using {n_proc} processes")
        with Pool(n_proc) as pool:
            results = pool.map(process_file_with_progress, sci_files)

        flat = [row for sub in results for row in sub]
        if flat:
            cur.executemany("""
                INSERT INTO extensions
                (filename, extension, ra_min, ra_max,
                 dec_min, dec_max, nx, ny)
                VALUES (:filename, :extension, :ra_min, :ra_max,
                        :dec_min, :dec_max, :nx, :ny)
            """, flat)
            conn.commit()
            print(f"✓ Indexed {len(flat)} extensions")
        else:
            print("✗ No valid extensions found")

        conn.close()

    # ---------------------------------------------------------------------
    # Extension lookup
    # ---------------------------------------------------------------------
    def find_extensions(self, ra, dec, margin=0.1):
        """Return extensions whose bounding boxes include (ra,dec)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT filename, extension, ra_min, ra_max, dec_min, dec_max, nx, ny
            FROM extensions
            WHERE dec_min <= ? + ? AND dec_max >= ? - ?
              AND (
                    (ra_min <= ra_max AND ra_min <= ? + ? AND ra_max >= ? - ?)
                 OR (ra_min >  ra_max AND (ra_min <= ? + ? OR ra_max >= ? - ?))
              )
            """,
            (dec, margin, dec, margin,
             ra, margin, ra, margin,
             ra, margin, ra, margin)
        )
        matches = cur.fetchall()
        conn.close()

        # precise WCS check
        verified = []
        for f, ext, r1, r2, d1, d2, nx, ny in matches:
            try:
                with fits.open(f) as hdul:
                    hdu = next(h for h in hdul if h.name == ext)
                    wcs = WCS(hdu.header)
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    if 0 <= x < nx and 0 <= y < ny:
                        verified.append({
                            'filename': f,
                            'extension': ext,
                            'x_center': x,
                            'y_center': y,
                            'nx': nx,
                            'ny': ny
                        })
            except Exception:
                pass

        return verified

    # ---------------------------------------------------------------------
    # PNG creation – orientation fixed
    # ---------------------------------------------------------------------
    def _create_png(self, data, cutout_wcs, output_file,
                    title="", source_center=None):
        """
        Save a PNG with North up, East left.

        If a flip along either axis is required, the data are flipped and a
        plain matplotlib Axes is used.  If no flip is required, the full WCS
        projection is preserved.
        """
        try:
            # Optimal stretch
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(data)

            ny, nx = data.shape
            cx, cy = nx // 2, ny // 2  # central pixel

            # Test orientation
            try:
                c_world = cutout_wcs.pixel_to_world(cx, cy)
                x_world = cutout_wcs.pixel_to_world(cx + 10, cy)
                y_world = cutout_wcs.pixel_to_world(cx, cy + 10)

                ra_increases_right = (x_world.ra.deg > c_world.ra.deg)
                dec_increases_up   = (y_world.dec.deg > c_world.dec.deg)
            except Exception:
                # conservative fall-back: assume typical orientation
                ra_increases_right = False
                dec_increases_up   = True

            flipped = False
            data_disp = data

            # East must be left (RA decreases → right)
            if ra_increases_right:
                data_disp = np.fliplr(data_disp)
                flipped = True
                if source_center is not None:
                    source_center = (nx - 1 - source_center[0], source_center[1])

            # North must be up (Dec increases ↑ up)
            if not dec_increases_up:
                data_disp = np.flipud(data_disp)
                flipped = True
                if source_center is not None:
                    source_center = (source_center[0], ny - 1 - source_center[1])

            # Plot
            fig = plt.figure(figsize=(10, 8))
            if flipped:
                ax = plt.subplot()
                im = ax.imshow(data_disp, origin='lower',
                               cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_xlabel('East  ←  Pixels  →  West')
                ax.set_ylabel('South ←  Pixels  →  North')
            else:
                ax = plt.subplot(projection=cutout_wcs)
                im = ax.imshow(data_disp, origin='lower',
                               cmap='gray', vmin=vmin, vmax=vmax,
                               transform=ax.get_transform('pixel'))
                ax.coords[0].set_axislabel('Right Ascension (J2000)')
                ax.coords[1].set_axislabel('Declination (J2000)')
                ax.coords[0].set_major_formatter('hh:mm:ss.s')
                ax.coords[1].set_major_formatter('dd:mm:ss')

            # Source marker
            if source_center is not None:
                sx, sy = source_center
                circ = plt.Circle((sx, sy), radius=5,
                                  edgecolor='cyan', facecolor='none',
                                  linewidth=2, alpha=0.8)
                ax.add_patch(circ)
                ax.plot([0, data_disp.shape[1]], [sy, sy],
                        color='cyan', ls='--', lw=1, alpha=0.6)
                ax.plot([sx, sx], [0, data_disp.shape[0]],
                        color='cyan', ls='--', lw=1, alpha=0.6)

            # Compass rose (always correct now)
            comp_x = data_disp.shape[1] * 0.85
            comp_y = data_disp.shape[0] * 0.85
            comp_s = min(data_disp.shape) * 0.06
            ax.annotate('N', (comp_x, comp_y + comp_s),
                        (comp_x, comp_y), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='red', arrowprops=dict(arrowstyle='->',
                        lw=3, color='red'))
            ax.annotate('E', (comp_x - comp_s, comp_y),
                        (comp_x, comp_y), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='red', arrowprops=dict(arrowstyle='->',
                        lw=3, color='red'))
            ax.text(0.02, 0.02, 'Orientation:  N↑  E←',
                    transform=ax.transAxes, fontsize=10,
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if title:
                ax.set_title(title, fontsize=12)

            plt.colorbar(im, ax=ax, label='Flux (counts)', shrink=0.8)
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            return True

        except Exception as e:
            print(f"Error creating PNG {output_file}: {e}")
            plt.close('all')
            return False

    # ---------------------------------------------------------------------
    # Cut-out extraction (unchanged except for calls to _create_png)
    # ---------------------------------------------------------------------

    def extract_cutout(self, ra, dec, size_arcsec=60, include_rms=False,
                   include_flg=False, output_dir="cutouts",
                   prefix="cutout", debug_wcs=False, 
                   create_png_sci=False, create_png_rms=False, create_png_flg=False, size_rmsflg=25):

        extensions = self.find_extensions(ra, dec)
        if not extensions:
            print(f"No extensions contain RA={ra:.6f}, DEC={dec:.6f}")
            return []

        print(f"Found {len(extensions)} extension(s)")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []

        for i, ext in enumerate(extensions, 1):
            sci_file = ext['filename']                # VIS_xxx_sci.fits
            sci_ext  = ext['extension']               # 1-1.E
            exp = extract_exp_from_filename(ext)

            # build filenames for rms/flg files
            rms_file = sci_file.replace("_sci", "_rms")
            flg_file = sci_file.replace("_sci", "_flg")

            xc, yc = ext['x_center'], ext['y_center']
            print(f"[{i}/{len(extensions)}] {Path(sci_file).name}[{sci_ext}]")

            # -------------------------------------------
            # Helper function to extract from any FITS file
            # -------------------------------------------
            def _extract_from_file(file_path, extname,label, size, create_png=False):

                try:
                    with fits.open(file_path) as hdul:
                        # find the right extension
                        try:
                            hdu = next(h for h in hdul if h.name == extname)
                        except StopIteration:
                            print(f"  ✗ Missing extension {extname} in {file_path}")
                            return None

                        wcs = WCS(hdu.header)

                        # pixel scale
                        try:
                            pixscale = np.abs(wcs.proj_plane_pixel_scales().mean()
                                            .to(u.arcsec).value)
                        except Exception:
                            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                                pixscale = (np.sqrt(np.abs(np.linalg.det(wcs.wcs.cd))) * 3600)
                            elif hasattr(wcs.wcs, 'cdelt'):
                                pixscale = np.abs(wcs.wcs.cdelt[0] * 3600)
                            else:
                                pixscale = 0.1  # fallback

                        sz_pix = max(10, int(round(size / pixscale)))
                        # sz_pix = max(10, int(size / pixscale))
                        
                        pos = (xc, yc)
                        size_pix = (sz_pix, sz_pix)

                        indices_min = [
                            int(np.ceil(p - s / 2))
                            for p, s in zip(pos, size_pix)
                        ]
                        indices_max = [
                            idx_min + s
                            for idx_min, s in zip(indices_min, size_pix)
                        ]

                        # Determine array shape and compute necessary shifts to keep
                        # the requested cutout fully inside the image if needed.
                        ny, nx = hdu.data.shape

                        xmin, ymin = indices_min[0], indices_min[1]
                        xmax, ymax = indices_max[0], indices_max[1]

                        # Compute pixel shifts required to bring the cutout in-bounds.
                        # Positive shift_x means move the cutout to the right (increase x),
                        # positive shift_y means move the cutout down (increase y).
                        shift_x = 0
                        if xmin < 0:
                            shift_x = -xmin
                        elif xmax > nx:
                            shift_x = nx - xmax

                        shift_y = 0
                        if ymin < 0:
                            shift_y = -ymin
                        elif ymax > ny:
                            shift_y = ny - ymax

                        # Distances from target pixel to image borders
                        dist_left = float(xc)
                        dist_right = float(nx - 1 - xc)
                        dist_top = float(yc)
                        dist_bottom = float(ny - 1 - yc)

                        # If strict behavior desired, raise on partial overlap
                        # (kept for compatibility; default behavior below will shift)
                        for e_min in indices_min:
                            if e_min < 0:
                                # continue to allow shifting unless strict mode required
                                pass
                        for e_max, large_shape in zip(indices_max, (nx, ny)):
                            if e_max > large_shape:
                                # continue to allow shifting unless strict mode required
                                pass

                        # Apply computed shifts to the requested center so the cutout
                        # is moved in-bounds when necessary.
                        pos_shifted = (float(xc + shift_x), float(yc + shift_y))

                        cut = Cutout2D(hdu.data, pos_shifted, size_pix, wcs=wcs,
                                    mode='partial', fill_value=np.nan)

                        # Position of the cutout region in the original image
                        xmin_shifted = indices_min[0] + shift_x
                        ymin_shifted = indices_min[1] + shift_y

                        # Position of the source (target) within the cutout
                        x_in_cut = float(xc - xmin_shifted)
                        y_in_cut = float(yc - ymin_shifted)

                        # Get cutout size
                        ny_cut, nx_cut = cut.data.shape

                        # Distances from source to cutout borders
                        dist_left_cut = float(x_in_cut)
                        dist_right_cut = float(nx_cut - 1 - x_in_cut)
                        dist_top_cut = float(y_in_cut)
                        dist_bottom_cut = float(ny_cut - 1 - y_in_cut)

                        # output file name
                        ccdid = extname.split('.')[0]
                        quad  = extname.split('.')[1]

                        tag = (f"{prefix}_ra{ra:.6f}_dec{dec:.6f}"
                            f"_ccd{ccdid}_quad{quad}_{exp:02d}_{label}")

                        out_fits = output_dir / f"{tag}.fits"

                        # header
                        hdr = cut.wcs.to_header()
                        for k in (
                            'DETID', 'CCDID', 'QUADID', 'ROEID', 'ROECTV',
                            'BUNIT', 'GAIN', 'RDNOISE', 'EXPTIME', 'MAGZEROP',
                            'SATLEVEL', 'SATURATE', 'GAINCORR',
                            'DATE-OBS', 'TIMESYS', 'MJD-OBS',
                            'PRESCANX', 'OVRSCANX', 'OVRSCANY',
                            'CMPRTSCI', 'COSMICPC', 'STDCRMS', 'NUMBRMS',
                            'AVGRESID', 'PHOT_ERR', 'PHOTIRMS', 'APERCORR',
                            'APCORRMS', 'FLXSCALE'
                        ):
                            if k in hdu.header:
                                hdr[k] = hdu.header[k]

                        hdr['ORIG_RA']  = (ra,  'Original RA')
                        hdr['ORIG_DEC'] = (dec, 'Original DEC')
                        hdr['CUTSIZE']  = (size, 'arcsec')
                        hdr['ORIGFILE'] = (os.path.basename(file_path)) # 'Source file' is too long to be written in FITS
                        hdr['ORIGEXT']  = (extname, 'Source extension')

                        # Position within cutout
                        hdr['X_CENTER'] = (x_in_cut, "Source X in cutout")
                        hdr['Y_CENTER'] = (y_in_cut, "Source Y in cutout")

                        # Original image position and shift info
                        hdr['ORIG_X']   = (float(xc), "Source X orig image")
                        hdr['ORIG_Y']   = (float(yc), "Source Y orig image")
                        hdr['XSHIFT'] = (float(shift_x), "X shift to bounds")
                        hdr['YSHIFT'] = (float(shift_y), "Y shift to bounds")
                    
                        hdr['EXTNAME'] = label.upper()
                        fits.PrimaryHDU(cut.data, hdr).writeto(out_fits, overwrite=True)

                        print(f"  ✓ {label.upper()} → {out_fits}")

                        # PNG
                        if create_png:

                            png = out_fits.with_suffix(".png")
                            title = (f"{label.upper()} | RA={ra:.6f}°, DEC={dec:.6f}° "
                                    f"| {ccdid}.{quad} | {size}″")

                            # Source position within the cutout (already computed above)
                            # x_in_cut and y_in_cut are the target coordinates in cutout pixels
                            src_c = (x_in_cut, y_in_cut)
                            self._create_png(cut.data, cut.wcs, png, title, src_c)
                            
                        return str(out_fits)

                except Exception as e:
                    print(f"  ✗ Error reading {file_path}: {e}")
                    return None

            # --- Extract SCI ---
            sci_cut = _extract_from_file(sci_file, sci_ext, "sci", size_arcsec, create_png=create_png_sci)
            if sci_cut:
                extracted.append(sci_cut)

            base_ext = sci_ext.replace(".SCI", "")  # ex: "1-1.E"

            # --- Extract RMS ---
            if include_rms:
                rms_ext = base_ext + ".RMS"         # ex: 1-1.E.RMS
                rms_cut = _extract_from_file(rms_file, rms_ext, "rms", size_rmsflg, create_png=create_png_rms)
                if rms_cut:
                    extracted.append(rms_cut)

            # --- Extract FLG ---
            if include_flg:
                flg_ext = base_ext + ".FLG"         # ex: 1-1.E.FLG
                flg_cut = _extract_from_file(flg_file, flg_ext, "flg", size_rmsflg, create_png=create_png_flg)
                if flg_cut:
                    extracted.append(flg_cut)
            
            # --- Extract BKG ---
            sci_basename = os.path.basename(sci_file)  # we don't want the full path
            parts = sci_basename.split('-')
            common_id = parts[2] + '-' + parts[3]      # 067070-11
            sci_dir = os.path.dirname(sci_file)

            pattern = os.path.join(sci_dir, f"EUC_VIS_SWL-BKG-{common_id}*")
            bkg_matches = glob.glob(pattern)

            if not bkg_matches:
                print(f"  ✗ No BKG file found for {sci_file}")
            else:
                bkg_file = bkg_matches[0]
                bkg_cut = _extract_from_file(bkg_file, base_ext, "bkg", size_arcsec, create_png=create_png_sci)
                if bkg_cut:
                    extracted.append(bkg_cut)

        return extracted

# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Extract postage stamps from Euclid VIS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
python euclid_extractor.py ./../../calibrated-frames/ --ra 17:51:55.89 --dec=-29:59:23.03 -o ./cutouts/event_name/ --psf-radec
python euclid_extractor.py ./../../calibrated-frames/ --ra 17:51:55.89 --dec=-29:59:23.03 -s 60 --rms --flg -o ./cutouts/event_name/ --rebuild-index --png-sci --png-rms --png-flg --size-rmsflg 60 --psf-radec --psf-quad --psf-all
"""

    )
    p.add_argument('data_dir', help='Directory with *_sci.fits files')
    p.add_argument('--ra',  required=True,
                   help='RA (deg or HH:MM:SS.S)')
    p.add_argument('--dec', required=True,
                   help='Dec (deg or ±DD:MM:SS.S)')
    p.add_argument('-s', '--size', type=float, default=60,
                   help='Cut-out size (arcsec)')
    p.add_argument('--rms', default=False, action='store_true',
                   help='Include RMS cut-outs')
    p.add_argument('--flg', default=False, action='store_true',
                   help='Include FLG cut-outs')
    p.add_argument('-o', '--output-dir', default='cutouts')
    p.add_argument('--prefix', default='cutout')
    p.add_argument('--rebuild-index', action='store_true')
    p.add_argument('--db-path', default='euclid_index.db')
    p.add_argument('--debug-wcs', action='store_true')
    p.add_argument('--png-sci', default=False, action='store_true')
    p.add_argument('--png-flg', default=False, action='store_true')
    p.add_argument('--png-rms', default=False, action='store_true')
    p.add_argument('--size-rmsflg', type=float, default=25,
                   help='Cut-out size for RMS and FLG (arcsec)')
    p.add_argument('--psf-radec', default=False, action='store_true',
                   help='Extract PSF at RA/Dec position')
    p.add_argument('--psf-quad', default=True, action='store_true',
                   help='Extract PSF for quadrant')
    p.add_argument('--psf-all', default=False, action='store_true',
                   help='Extract PSF averaged over all quadrants')
    args = p.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} not found")
        sys.exit(1)

    # Parse coordinates
    try:
        if ':' in args.ra or ':' in args.dec:
            c = SkyCoord(f"{args.ra} {args.dec}",
                         unit=(u.hourangle, u.deg))
            ra_deg, dec_deg = c.ra.deg, c.dec.deg
        else:
            ra_deg, dec_deg = float(args.ra), float(args.dec)
    except Exception as e:
        print(f"Coordinate parse error: {e}")
        sys.exit(1)

    extractor = EuclidCutoutExtractor(
        args.data_dir, db_path=args.db_path,
        rebuild_index=args.rebuild_index
    )

    t0 = time.time()
    outs = extractor.extract_cutout(
        ra_deg, dec_deg, size_arcsec=args.size,
        include_rms=args.rms, include_flg=args.flg,
        output_dir=args.output_dir, prefix=args.prefix,
        debug_wcs=args.debug_wcs, create_png_sci=args.png_sci,
        create_png_rms=args.png_rms, create_png_flg=args.png_flg,
        size_rmsflg=25
    )
    dt = time.time() - t0

    if outs:
        print(f"\nExtracted {len(outs)} cut-out(s) in {dt:.1f} s")
    else:
        print("No cut-outs extracted")

    # PSF extraction
    psf_file = "./PSFEx_model/EUC_VIS_MDL-PSF_250323-250324_20251010T191512.440105Z.psf"
    matches = extractor.find_extensions(ra_deg, dec_deg)
    psfex = EuclidPSFExtractor(psf_file)
    matched_psfs = psfex.psfs_for_matches(
        matches,
        RA=ra_deg,
        DEC=dec_deg,
        p_radec=args.psf_radec,
        p_quad=args.psf_quad,
        p_all=args.psf_all
    )
    matched_psfs.to_pickle(Path(args.output_dir) / "PSFs.pkl")


    if len(matched_psfs) > 0:
        print(f"PSFs extracted for {len(matched_psfs)} extensions")
    else:
        print("No PSFs extracted")
    
if __name__ == "__main__":
    main()
