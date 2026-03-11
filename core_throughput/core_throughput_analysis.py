import numpy as np
from scipy.ndimage import center_of_mass


try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _theta_to_deg(theta):
    """
    Convert an angle to degrees.

    Supports:
    - astropy Quantity with angular units
    - plain float/int already in degrees
    """
    if hasattr(theta, "to_value"):
        return float(theta.to_value("deg"))
    return float(theta)


def _parse_valid_positions(valid_positions):
    """
    Convert valid_positions = [(rho_ld, theta), ...] into numeric arrays.
    
    Parameters
    ----------
    valid_positions : sequence
        Each element must be (rho_ld, theta_deg_or_quantity)

    Returns
    -------
    rho_ld : np.ndarray
        Radial separations in lambda/D units.
    theta_deg : np.ndarray
        Position angles in degrees.
    """
    rho_ld = []
    theta_deg = []

    for item in valid_positions:
        if len(item) != 2:
            raise ValueError(
                "Each valid_positions entry must be a 2-tuple: (rho_ld, theta)."
            )
        rho, theta = item
        rho_ld.append(float(rho))
        theta_deg.append(_theta_to_deg(theta))

    return np.asarray(rho_ld, dtype=float), np.asarray(theta_deg, dtype=float)


def _ld_to_pix(res_mas, pixel_scale_mas):
    """
    Convert lambda/D to pixels.

    Parameters
    ----------
    res_mas : float
        Angular resolution lambda/D in mas.
    pixel_scale_mas : float
        Pixel scale in mas/pixel.

    Returns
    -------
    float
        Pixels per lambda/D.
    """
    return float(res_mas) / float(pixel_scale_mas)


def _polar_ld_to_xy_pix(valid_positions, pixel_scale_mas, res_mas, centre_xy=None):
    """
    Convert valid_positions to detector x, y coordinates in pixels.

    Convention:
    - theta = 0 deg lies on +x
    - theta increases counter clockwise in Cartesian coordinates
    - image y increases downward, so y_img = y0 - r * sin(theta)

    Parameters
    ----------
    valid_positions : sequence
        [(rho_ld, theta), ...]
    pixel_scale_mas : float
        Pixel scale in mas/pixel.
    res_mas : float
        lambda/D in mas.
    centre_xy : tuple or None
        (x0, y0) in pixel coordinates. If None, caller should supply later.

    Returns
    -------
    x : np.ndarray
    y : np.ndarray
    """
    rho_ld, theta_deg = _parse_valid_positions(valid_positions)
    pix_per_ld = _ld_to_pix(res_mas=res_mas, pixel_scale_mas=pixel_scale_mas)

    r_pix = rho_ld * pix_per_ld
    theta_rad = np.deg2rad(theta_deg)

    if centre_xy is None:
        x0, y0 = 0.0, 0.0
    else:
        x0, y0 = map(float, centre_xy)

    x = x0 + r_pix * np.cos(theta_rad)
    y = y0 - r_pix * np.sin(theta_rad)

    return x, y


def _ensure_cube_stack(arr, n_expected=None, name="array"):
    """
    Ensure input is a stack of 2D images with shape (N, Ny, Nx).

    Accepts:
    - (N, Ny, Nx)
    - (Ny, Nx), which becomes (1, Ny, Nx)

    Parameters
    ----------
    arr : np.ndarray
    n_expected : int or None
    name : str

    Returns
    -------
    np.ndarray
        Shape (N, Ny, Nx)
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (N, Ny, Nx) or (Ny, Nx). Got {arr.shape}."
        )

    if n_expected is not None and arr.shape[0] not in (1, n_expected):
        raise ValueError(
            f"{name} first dimension must be 1 or {n_expected}. Got shape {arr.shape}."
        )

    return arr


def _broadcast_reference_cube(no_fpm_psf_cube, n_samples):
    """
    Broadcast the no-FPM PSF cube so it has one entry per valid position.

    Parameters
    ----------
    no_fpm_psf_cube : np.ndarray
        Shape (Ny, Nx) or (1, Ny, Nx) or (N, Ny, Nx)
    n_samples : int

    Returns
    -------
    np.ndarray
        Shape (n_samples, Ny, Nx)
    """
    ref = _ensure_cube_stack(no_fpm_psf_cube, n_expected=n_samples, name="no_fpm_psf_cube")

    if ref.shape[0] == 1 and n_samples > 1:
        ref = np.repeat(ref, n_samples, axis=0)

    return ref


def _circular_aperture_mask(shape, radius_pix, centre=None):
    """
    Create a circular aperture mask.

    Parameters
    ----------
    shape : tuple
        (Ny, Nx)
    radius_pix : float
        Aperture radius in pixels.
    centre : tuple or None
        (y0, x0). If None, use image centre.

    Returns
    -------
    mask : np.ndarray
        Boolean mask.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)

    if centre is None:
        y0 = (ny - 1) / 2.0
        x0 = (nx - 1) / 2.0
    else:
        y0, x0 = map(float, centre)

    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return rr <= float(radius_pix)


def interpolate_tau_xy(
    tau_vec,
    valid_positions,
    img_shape,
    pixel_scale_mas=21.8,
    res_mas=50.0,
    centre_xy=None,
    method="linear",
    fill_value=np.nan,
    nearest_fallback=True,
):
    """
    Interpolate sparse throughput samples defined at valid_positions onto a 2D grid.

    Parameters
    ----------
    tau_vec : array-like
        Throughput samples, shape (N,).
    valid_positions : sequence
        [(rho_ld, theta), ...]
    img_shape : tuple
        (Ny, Nx)
    pixel_scale_mas : float
        Pixel scale in mas/pixel.
    res_mas : float
        lambda/D in mas.
    centre_xy : tuple or None
        (x0, y0) centre of the star on the output image.
        If None, the image centre is used.
    method : str
        Interpolation method. If SciPy is available, supports 'linear', 'nearest',
        and 'cubic'. Without SciPy, only nearest-neighbour fallback is used.
    fill_value : float
        Value outside the convex hull for SciPy interpolation.
    nearest_fallback : bool
        If True and method is not 'nearest', fill NaN regions using nearest neighbour.

    Returns
    -------
    tau_img : np.ndarray
        Interpolated image of shape (Ny, Nx).
    """
    tau_vec = np.asarray(tau_vec, dtype=float)
    ny, nx = img_shape

    if len(tau_vec) != len(valid_positions):
        raise ValueError(
            f"tau_vec length ({len(tau_vec)}) must match len(valid_positions) ({len(valid_positions)})."
        )

    if centre_xy is None:
        x0 = (nx - 1) / 2.0
        y0 = (ny - 1) / 2.0
    else:
        x0, y0 = map(float, centre_xy)

    x_samp, y_samp = _polar_ld_to_xy_pix(
        valid_positions,
        pixel_scale_mas=pixel_scale_mas,
        res_mas=res_mas,
        centre_xy=(x0, y0),
    )

    good = np.isfinite(tau_vec) & np.isfinite(x_samp) & np.isfinite(y_samp)
    x_samp = x_samp[good]
    y_samp = y_samp[good]
    tau_good = tau_vec[good]

    if tau_good.size == 0:
        return np.full((ny, nx), fill_value, dtype=float)

    yy, xx = np.indices((ny, nx), dtype=float)

    if _HAS_SCIPY:
        points = np.column_stack([x_samp, y_samp])
        tau_img = griddata(
            points=points,
            values=tau_good,
            xi=(xx, yy),
            method=method,
            fill_value=fill_value,
        )

        if nearest_fallback and method != "nearest":
            nanmask = ~np.isfinite(tau_img)
            if np.any(nanmask):
                tau_nn = griddata(
                    points=points,
                    values=tau_good,
                    xi=(xx, yy),
                    method="nearest",
                    fill_value=fill_value,
                )
                tau_img[nanmask] = tau_nn[nanmask]

        return tau_img

    # Fallback without SciPy: nearest neighbour interpolation only
    samp = np.column_stack([x_samp, y_samp])
    query = np.column_stack([xx.ravel(), yy.ravel()])

    # squared distances to avoid sqrt
    d2 = (
        (query[:, None, 0] - samp[None, :, 0]) ** 2
        + (query[:, None, 1] - samp[None, :, 1]) ** 2
    )
    idx = np.argmin(d2, axis=1)
    tau_img = tau_good[idx].reshape(ny, nx)

    return tau_img


def interpolate_tau_from_valid_positions(
    tau_vec,
    valid_positions,
    img_shape,
    pixel_scale_mas=21.8,
    res_mas=50.0,
    centre_xy=None,
    method="linear",
    fill_value=np.nan,
    nearest_fallback=True,
):
    """
    Alias for interpolate_tau_xy.

    This name is kept because it reads well for core throughput, but the logic is
    identical for total throughput or any other scalar sampled at valid_positions.
    """
    return interpolate_tau_xy(
        tau_vec=tau_vec,
        valid_positions=valid_positions,
        img_shape=img_shape,
        pixel_scale_mas=pixel_scale_mas,
        res_mas=res_mas,
        centre_xy=centre_xy,
        method=method,
        fill_value=fill_value,
        nearest_fallback=nearest_fallback,
    )


def compute_core_tau_on_valid_positions(
    corgi_prf_cubes,
    no_fpm_psf_cube,
    valid_positions,
    pixel_scale_mas=21.8,
    res_mas=50.0,
    r_core_ld=1.22,
    return_debug=False):

    n_samples = len(valid_positions)

    corgi = _ensure_cube_stack(
        corgi_prf_cubes,
        n_expected=n_samples,
        name="corgi_prf_cubes",
    )
    ref = _broadcast_reference_cube(
        no_fpm_psf_cube,
        n_samples=n_samples,
    )

    r_core_pix = float(r_core_ld) * _ld_to_pix(
        res_mas=res_mas,
        pixel_scale_mas=pixel_scale_mas,
    )
    print(f"Using r_core_pix = {r_core_pix:.2f} pixels")

    edge_pad = int(np.ceil(r_core_pix))

    tau_core_vec = np.full(n_samples, np.nan, dtype=float)
    coro_core    = np.full(n_samples, np.nan, dtype=float)
    ref_core     = np.full(n_samples, np.nan, dtype=float)
    ref_peaks    = np.zeros((n_samples, 2), dtype=float)
    edge_clipped = np.zeros(n_samples, dtype=bool)
    mask_sizes   = np.zeros(n_samples, dtype=int)  # ← added

    for i in range(n_samples):
        coro_i = np.asarray(corgi[i], dtype=float)
        ref_i  = np.asarray(ref[i],   dtype=float)
        ref_total = float(np.nansum(ref[i]))

        # stable centroid-based peak finding on reference PSF
        threshold  = 0.5 * np.nanmax(ref_i)
        bright_mask = ref_i > threshold
        ref_peak_yx = center_of_mass(ref_i * bright_mask)
        ref_peak_yx = (float(ref_peak_yx[0]), float(ref_peak_yx[1]))
        ref_peaks[i] = ref_peak_yx

        # flag if aperture clips the edge
        ny, nx = ref_i.shape
        y0, x0 = ref_peak_yx
        if (y0 < edge_pad or y0 >= ny - edge_pad or
                x0 < edge_pad or x0 >= nx - edge_pad):
            edge_clipped[i] = True
            continue

        mask = _circular_aperture_mask(
            shape=ref_i.shape,
            radius_pix=r_core_pix,
            centre=ref_peak_yx,
        )

        mask_sizes[i]  = int(mask.sum())  # ← added
        coro_core[i]   = np.nansum(coro_i[mask])
        ref_core[i]    = np.nansum(ref_i[mask])

        if np.isfinite(coro_core[i]) and ref_total > 0:
            tau_core_vec[i] = coro_core[i] / ref_total

    n_clipped = edge_clipped.sum()
    if n_clipped > 0:
        print(f"Warning: {n_clipped}/{n_samples} positions clipped by edge and set to NaN.")

    if return_debug:
        debug = {
            "coro_core":    coro_core,
            "ref_core":     ref_core,
            "ref_total":    ref_total,
            "ref_peaks":    ref_peaks,
            "edge_clipped": edge_clipped,
            "r_core_pix":   r_core_pix,
            "edge_pad":     edge_pad,
            "mask_sizes":   mask_sizes,  # ← added
        }
        return tau_core_vec, debug

    return tau_core_vec

def compute_cgperf_core_tau_on_valid_positions(
    corgi_prf_cubes,
    no_fpm_psf_cube,
    valid_positions,
    return_debug=False,
):
    """
    Compute core throughput following the CGPERF/Zellem+2022 definition:

        tau_core = sum(coro[mask]) / sum(ref_total)

    where:
    - mask = pixels where ref flux > 50% of the reference PSF peak
             (peak location found per-sample via nanargmax on ref[i])
    - ref_total = total flux of the unocculted reference PSF (proxy for
                  total flux of the illuminated primary mirror area)

    The aperture is defined on the reference PSF (not the coronagraphic PSF)
    because the coronagraphic PSF is a suppressed donut and has no meaningful
    peak. The reference PSF peak moves with field position (valid_positions),
    so it is found independently for each sample.

    Reference: Zellem et al. 2022, Section 4.1 (arXiv:2202.05923)

    Parameters
    ----------
    corgi_prf_cubes : np.ndarray
        Coronagraphic PRF stamps, shape (N, Ny, Nx) or (Ny, Nx).
    no_fpm_psf_cube : np.ndarray
        Unocculted reference PSF, shape (N, Ny, Nx), (1, Ny, Nx), or (Ny, Nx).
    valid_positions : sequence
        List of (rho_ld, theta). Used to validate N and track field positions.
    return_debug : bool
        If True, return a debug dict alongside tau_core_vec.

    Returns
    -------
    tau_core_vec : np.ndarray, shape (N,)
        Core throughput at each valid position.
    debug : dict (only if return_debug=True)
    """
    n_samples = len(valid_positions)

    corgi = _ensure_cube_stack(
        corgi_prf_cubes,
        n_expected=n_samples,
        name="corgi_prf_cubes",
    )
    ref = _broadcast_reference_cube(
        no_fpm_psf_cube,
        n_samples=n_samples,
    )

    # total reference flux — same for all samples (proxy for primary mirror flux)
    ref_total = float(np.nansum(ref[0]))

    tau_core_vec = np.full(n_samples, np.nan, dtype=float)
    coro_core = np.full(n_samples, np.nan, dtype=float)
    ref_peaks = np.zeros((n_samples, 2), dtype=float)
    mask_sizes = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        coro_i = np.asarray(corgi[i], dtype=float)
        ref_i  = np.asarray(ref[i],   dtype=float)

        # star position from reference PSF peak (moves with field position)
        ref_peak_yx = np.unravel_index(np.nanargmax(ref_i), ref_i.shape)
        ref_peaks[i] = ref_peak_yx
        ref_peak_val = ref_i[ref_peak_yx]

        if not np.isfinite(ref_peak_val) or ref_peak_val <= 0:
            continue

        # CGPERF definition: core = pixels where ref > 50% of ref peak
        mask = ref_i > 0.5 * ref_peak_val
        mask_sizes[i] = mask.sum()

        coro_core[i] = np.nansum(coro_i[mask])

        if np.isfinite(coro_core[i]) and ref_total > 0:
            tau_core_vec[i] = coro_core[i] / ref_total

    if return_debug:
        debug = {
            "coro_core": coro_core,
            "ref_total": ref_total,
            "ref_peaks": ref_peaks,
            "mask_sizes": mask_sizes,
        }
        return tau_core_vec, debug

    return tau_core_vec


def radial_median(tau_vec, rho_ld, r_bins):
    valid = np.isfinite(tau_vec)
    return [
        np.nanmedian(tau_vec[valid & (rho_ld >= r0) & (rho_ld < r1)])
        for r0, r1 in zip(r_bins[:-1], r_bins[1:])
    ]


def radial_mean(tau_vec, rho_ld, r_bins):
    valid = np.isfinite(tau_vec)
    return [
        np.nanmean(tau_vec[valid & (rho_ld >= r0) & (rho_ld < r1)])
        for r0, r1 in zip(r_bins[:-1], r_bins[1:])
    ]

