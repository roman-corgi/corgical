import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import center_of_mass


__all__ = [
    "angle_to_degrees",
    "parse_sample_positions",
    "lambda_d_to_pixels",
    "positions_to_pixel_coordinates",
    "ensure_image_stack",
    "broadcast_reference_stack",
    "make_circular_mask",
    "interpolate_throughput_map",
    "compute_core_throughput_fixed_aperture",
    "compute_core_throughput_half_max",
    "compute_radial_statistic",
    "radial_mean",
    "radial_median",
]

# TODO - we need to be able to calculate these value because some are wavelength dependent. 

DEFAULT_PIXEL_SCALE_MAS = 21.8
DEFAULT_LAMBDA_OVER_D_MAS = 50.0
DEFAULT_CORE_RADIUS_LD = 1.22


def angle_to_degrees(theta):
    """
    Convert an angle to degrees.

    Parameters
    ----------
    theta : float or astropy quantity
        Angle value. If an astropy quantity is provided, it must support
        ``to_value('deg')``.

    Returns
    -------
    float
        Angle in degrees.
    """
    if hasattr(theta, "to_value"):
        return float(theta.to_value("deg"))
    return float(theta)


def parse_sample_positions(sample_positions):
    """
    Convert sampled positions into numeric radius and angle arrays.

    Parameters
    ----------
    sample_positions : sequence of tuple
        Sequence of ``(rho_ld, theta)`` pairs, where ``rho_ld`` is the
        separation in lambda over D and ``theta`` is the position angle.

    Returns
    -------
    rho_ld : ndarray
        Radial separations in lambda over D.
    theta_deg : ndarray
        Position angles in degrees.
    """
    rho_ld = []
    theta_deg = []

    for item in sample_positions:
        if len(item) != 2:
            raise ValueError(
                "Each sample position must be a 2 tuple: (rho_ld, theta)."
            )
        rho, theta = item
        rho_ld.append(float(rho))
        theta_deg.append(angle_to_degrees(theta))

    return np.asarray(rho_ld, dtype=float), np.asarray(theta_deg, dtype=float)


def lambda_d_to_pixels(lambda_d_mas=DEFAULT_LAMBDA_OVER_D_MAS,
                       pixel_scale_mas=DEFAULT_PIXEL_SCALE_MAS):
    """
    Convert lambda over D to detector pixels.

    Parameters
    ----------
    lambda_d_mas : float
        Angular size of one lambda over D in mas.
    pixel_scale_mas : float
        Pixel scale in mas per pixel.

    Returns
    -------
    float
        Pixels per lambda over D.
    """
    return float(lambda_d_mas) / float(pixel_scale_mas)


def positions_to_pixel_coordinates(sample_positions,
                                   pixel_scale_mas=DEFAULT_PIXEL_SCALE_MAS,
                                   lambda_d_mas=DEFAULT_LAMBDA_OVER_D_MAS,
                                   centre_xy=None):
    """
    Convert sampled polar coordinates into image pixel coordinates.

    Parameters
    ----------
    sample_positions : sequence of tuple
        Sequence of ``(rho_ld, theta)`` pairs.
    pixel_scale_mas : float
        Pixel scale in mas per pixel.
    lambda_d_mas : float
        Angular size of one lambda over D in mas.
    centre_xy : tuple, optional
        Image centre as ``(x0, y0)`` in pixel coordinates. If omitted,
        coordinates are returned relative to ``(0, 0)``.

    Returns
    -------
    x : ndarray
        Sample x coordinates in pixels.
    y : ndarray
        Sample y coordinates in pixels.

    Notes
    -----
    The convention is:

    * theta = 0 degrees lies on +x
    * theta increases counter clockwise in Cartesian coordinates
    * image y increases downward, so ``y = y0 - r sin(theta)``
    """
    rho_ld, theta_deg = parse_sample_positions(sample_positions)
    pixels_per_ld = lambda_d_to_pixels(
        lambda_d_mas=lambda_d_mas,
        pixel_scale_mas=pixel_scale_mas,
    )

    radius_pix = rho_ld * pixels_per_ld
    theta_rad = np.deg2rad(theta_deg)

    if centre_xy is None:
        x0, y0 = 0.0, 0.0
    else:
        x0, y0 = map(float, centre_xy)

    x = x0 + radius_pix * np.cos(theta_rad)
    y = y0 - radius_pix * np.sin(theta_rad)
    return x, y


def ensure_image_stack(images, n_expected=None, name="images"):
    """
    Ensure an input array has shape ``(N, Ny, Nx)``.

    Parameters
    ----------
    images : ndarray
        Input image or image stack.
    n_expected : int, optional
        Expected number of frames. If provided, the first dimension must be
        either 1 or ``n_expected``.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray
        Image stack with shape ``(N, Ny, Nx)``.
    """
    images = np.asarray(images)

    if images.ndim == 2:
        images = images[None, ...]
    elif images.ndim != 3:
        raise ValueError(
            f"{name} must have shape (N, Ny, Nx) or (Ny, Nx). Got {images.shape}."
        )

    if n_expected is not None and images.shape[0] not in (1, n_expected):
        raise ValueError(
            f"{name} first dimension must be 1 or {n_expected}. Got {images.shape}."
        )

    return images


def broadcast_reference_stack(reference_images, n_samples):
    """
    Broadcast a reference PSF stack to match the number of samples.

    Parameters
    ----------
    reference_images : ndarray
        Reference image or stack with shape ``(Ny, Nx)``, ``(1, Ny, Nx)``, or
        ``(N, Ny, Nx)``.
    n_samples : int
        Required number of frames.

    Returns
    -------
    ndarray
        Reference stack with shape ``(n_samples, Ny, Nx)``.
    """
    reference_images = ensure_image_stack(
        reference_images,
        n_expected=n_samples,
        name="reference_images",
    )

    if reference_images.shape[0] == 1 and n_samples > 1:
        reference_images = np.repeat(reference_images, n_samples, axis=0)

    return reference_images


def make_circular_mask(shape, radius_pix, centre_yx=None):
    """
    Create a circular boolean mask.

    Parameters
    ----------
    shape : tuple
        Image shape ``(Ny, Nx)``.
    radius_pix : float
        Mask radius in pixels.
    centre_yx : tuple, optional
        Mask centre as ``(y0, x0)``. If omitted, the image centre is used.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)

    if centre_yx is None:
        y0 = (ny - 1) / 2.0
        x0 = (nx - 1) / 2.0
    else:
        y0, x0 = map(float, centre_yx)

    radius = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return radius <= float(radius_pix)


def interpolate_throughput_map(throughput_values,
                               sample_positions,
                               image_shape,
                               pixel_scale_mas=DEFAULT_PIXEL_SCALE_MAS,
                               lambda_d_mas=DEFAULT_LAMBDA_OVER_D_MAS,
                               centre_xy=None,
                               method="linear",
                               fill_value=np.nan,
                               fill_missing_with_nearest=True):
    """
    Interpolate throughput samples onto a 2D detector grid.

    Parameters
    ----------
    throughput_values : array-like
        Throughput values with shape ``(N,)``.
    sample_positions : sequence of tuple
        Sequence of ``(rho_ld, theta)`` pairs.
    image_shape : tuple
        Output image shape ``(Ny, Nx)``.
    pixel_scale_mas : float
        Pixel scale in mas per pixel.
    lambda_d_mas : float
        Angular size of one lambda over D in mas.
    centre_xy : tuple, optional
        Stellar position as ``(x0, y0)`` in the output image. If omitted,
        the image centre is used.
    method : str
        Interpolation method passed to ``scipy.interpolate.griddata``.
    fill_value : float
        Fill value used outside the convex hull.
    fill_missing_with_nearest : bool
        If True, replace NaN pixels from the main interpolation with a nearest
        neighbour interpolation.

    Returns
    -------
    ndarray
        Interpolated throughput image with shape ``(Ny, Nx)``.
    """
    throughput_values = np.asarray(throughput_values, dtype=float)
    ny, nx = image_shape

    if len(throughput_values) != len(sample_positions):
        raise ValueError(
            "throughput_values length must match the number of sample positions."
        )

    if centre_xy is None:
        x0 = (nx - 1) / 2.0
        y0 = (ny - 1) / 2.0
    else:
        x0, y0 = map(float, centre_xy)

    x_samples, y_samples = positions_to_pixel_coordinates(
        sample_positions,
        pixel_scale_mas=pixel_scale_mas,
        lambda_d_mas=lambda_d_mas,
        centre_xy=(x0, y0),
    )

    valid = (
        np.isfinite(throughput_values)
        & np.isfinite(x_samples)
        & np.isfinite(y_samples)
    )

    if not np.any(valid):
        return np.full((ny, nx), fill_value, dtype=float)

    points = np.column_stack([x_samples[valid], y_samples[valid]])
    values = throughput_values[valid]
    yy, xx = np.indices((ny, nx), dtype=float)

    throughput_map = griddata(
        points=points,
        values=values,
        xi=(xx, yy),
        method=method,
        fill_value=fill_value,
    )

    if fill_missing_with_nearest and method != "nearest":
        missing = ~np.isfinite(throughput_map)
        if np.any(missing):
            nearest_map = griddata(
                points=points,
                values=values,
                xi=(xx, yy),
                method="nearest",
                fill_value=fill_value,
            )
            throughput_map[missing] = nearest_map[missing]

    return throughput_map


def compute_core_throughput_fixed_aperture(coronagraphic_images,
                                           reference_images,
                                           sample_positions,
                                           pixel_scale_mas=DEFAULT_PIXEL_SCALE_MAS,
                                           lambda_d_mas=DEFAULT_LAMBDA_OVER_D_MAS,
                                           core_radius_ld=DEFAULT_CORE_RADIUS_LD,
                                           return_debug=False):
    """
    Compute core throughput with a fixed circular aperture.

    The aperture is centred on the reference PSF peak and its radius is set in
    lambda over D. The throughput is defined as the flux inside that aperture
    in the coronagraphic image divided by the total flux of the corresponding
    reference image.

    Parameters
    ----------
    coronagraphic_images : ndarray
        Coronagraphic image or stack with shape ``(Ny, Nx)`` or ``(N, Ny, Nx)``.
    reference_images : ndarray
        Reference image or stack with shape ``(Ny, Nx)``, ``(1, Ny, Nx)``, or
        ``(N, Ny, Nx)``.
    sample_positions : sequence of tuple
        Sequence of ``(rho_ld, theta)`` pairs. Used to validate the number of
        samples.
    pixel_scale_mas : float
        Pixel scale in mas per pixel.
    lambda_d_mas : float
        Angular size of one lambda over D in mas.
    core_radius_ld : float
        Circular aperture radius in lambda over D.
    return_debug : bool
        If True, also return diagnostic arrays.

    Returns
    -------
    throughput : ndarray
        Core throughput for each sampled position.
    debug : dict, optional
        Returned only when ``return_debug=True``.
    """
    n_samples = len(sample_positions)
    coro_stack = ensure_image_stack(
        coronagraphic_images,
        n_expected=n_samples,
        name="coronagraphic_images",
    )
    ref_stack = broadcast_reference_stack(reference_images, n_samples)

    core_radius_pix = core_radius_ld * lambda_d_to_pixels(
        lambda_d_mas=lambda_d_mas,
        pixel_scale_mas=pixel_scale_mas,
    )
    edge_margin = int(np.ceil(core_radius_pix))

    throughput = np.full(n_samples, np.nan, dtype=float)
    core_flux = np.full(n_samples, np.nan, dtype=float)
    reference_core_flux = np.full(n_samples, np.nan, dtype=float)
    reference_total_flux = np.full(n_samples, np.nan, dtype=float)
    reference_peaks = np.full((n_samples, 2), np.nan, dtype=float)
    edge_clipped = np.zeros(n_samples, dtype=bool)
    mask_sizes = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        coro_image = np.asarray(coro_stack[i], dtype=float)
        ref_image = np.asarray(ref_stack[i], dtype=float)
        reference_total_flux[i] = np.nansum(ref_image)

        threshold = 0.5 * np.nanmax(ref_image)
        bright_region = ref_image > threshold
        peak_yx = center_of_mass(ref_image * bright_region)
        peak_yx = (float(peak_yx[0]), float(peak_yx[1]))
        reference_peaks[i] = peak_yx

        ny, nx = ref_image.shape
        y0, x0 = peak_yx
        if (
            y0 < edge_margin
            or y0 >= ny - edge_margin
            or x0 < edge_margin
            or x0 >= nx - edge_margin
        ):
            edge_clipped[i] = True
            continue

        mask = make_circular_mask(
            shape=ref_image.shape,
            radius_pix=core_radius_pix,
            centre_yx=peak_yx,
        )

        mask_sizes[i] = int(mask.sum())
        core_flux[i] = np.nansum(coro_image[mask])
        reference_core_flux[i] = np.nansum(ref_image[mask])

        if np.isfinite(core_flux[i]) and reference_total_flux[i] > 0:
            throughput[i] = core_flux[i] / reference_total_flux[i]

    if return_debug:
        debug = {
            "core_flux": core_flux,
            "reference_core_flux": reference_core_flux,
            "reference_total_flux": reference_total_flux,
            "reference_peaks": reference_peaks,
            "edge_clipped": edge_clipped,
            "core_radius_pix": core_radius_pix,
            "edge_margin": edge_margin,
            "mask_sizes": mask_sizes,
        }
        return throughput, debug

    return throughput


def compute_core_throughput_half_max(coronagraphic_images,
                                     reference_images,
                                     sample_positions,
                                     return_debug=False):
    """
    Compute core throughput using the half maximum footprint of the reference PSF.

    This follows the common definition where the core is the set of pixels for
    which the reference PSF exceeds half of its peak value.

    Parameters
    ----------
    coronagraphic_images : ndarray
        Coronagraphic image or stack with shape ``(Ny, Nx)`` or ``(N, Ny, Nx)``.
    reference_images : ndarray
        Reference image or stack with shape ``(Ny, Nx)``, ``(1, Ny, Nx)``, or
        ``(N, Ny, Nx)``.
    sample_positions : sequence of tuple
        Sequence of ``(rho_ld, theta)`` pairs. Used to validate the number of
        samples.
    return_debug : bool
        If True, also return diagnostic arrays.

    Returns
    -------
    throughput : ndarray
        Core throughput for each sampled position.
    debug : dict, optional
        Returned only when ``return_debug=True``.
    """
    n_samples = len(sample_positions)
    coro_stack = ensure_image_stack(
        coronagraphic_images,
        n_expected=n_samples,
        name="coronagraphic_images",
    )
    ref_stack = broadcast_reference_stack(reference_images, n_samples)

    throughput = np.full(n_samples, np.nan, dtype=float)
    core_flux = np.full(n_samples, np.nan, dtype=float)
    reference_total_flux = np.full(n_samples, np.nan, dtype=float)
    reference_peaks = np.full((n_samples, 2), np.nan, dtype=float)
    mask_sizes = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        coro_image = np.asarray(coro_stack[i], dtype=float)
        ref_image = np.asarray(ref_stack[i], dtype=float)
        reference_total_flux[i] = np.nansum(ref_image)

        peak_index = np.unravel_index(np.nanargmax(ref_image), ref_image.shape)
        peak_value = ref_image[peak_index]
        reference_peaks[i] = peak_index

        if not np.isfinite(peak_value) or peak_value <= 0:
            continue

        mask = ref_image > 0.5 * peak_value
        mask_sizes[i] = int(mask.sum())
        core_flux[i] = np.nansum(coro_image[mask])

        if np.isfinite(core_flux[i]) and reference_total_flux[i] > 0:
            throughput[i] = core_flux[i] / reference_total_flux[i]

    if return_debug:
        debug = {
            "core_flux": core_flux,
            "reference_total_flux": reference_total_flux,
            "reference_peaks": reference_peaks,
            "mask_sizes": mask_sizes,
        }
        return throughput, debug

    return throughput


def compute_radial_statistic(values, radii_ld, radial_bins, statistic="mean"):
    """
    Compute a radial summary statistic in separation bins.

    Parameters
    ----------
    values : array-like
        Values to summarise.
    radii_ld : array-like
        Radial coordinate for each value in lambda over D.
    radial_bins : array-like
        Bin edges in lambda over D.
    statistic : {'mean', 'median'}
        Summary statistic to compute.

    Returns
    -------
    ndarray
        One value per radial bin.
    """
    values = np.asarray(values, dtype=float)
    radii_ld = np.asarray(radii_ld, dtype=float)
    radial_bins = np.asarray(radial_bins, dtype=float)

    valid = np.isfinite(values) & np.isfinite(radii_ld)
    summary = np.full(len(radial_bins) - 1, np.nan, dtype=float)

    for i, (r0, r1) in enumerate(zip(radial_bins[:-1], radial_bins[1:])):
        in_bin = valid & (radii_ld >= r0) & (radii_ld < r1)
        if not np.any(in_bin):
            continue

        if statistic == "mean":
            summary[i] = np.nanmean(values[in_bin])
        elif statistic == "median":
            summary[i] = np.nanmedian(values[in_bin])
        else:
            raise ValueError("statistic must be 'mean' or 'median'.")

    return summary


def radial_mean(values, radii_ld, radial_bins):
    """Return the radial mean in each separation bin."""
    return compute_radial_statistic(values, radii_ld, radial_bins, statistic="mean")


def radial_median(values, radii_ld, radial_bins):
    """Return the radial median in each separation bin."""
    return compute_radial_statistic(values, radii_ld, radial_bins, statistic="median")
