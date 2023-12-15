import numpy as np
import xarray as xr
import scipy.ndimage as scpnd
from shapely.geometry import Polygon


def add_bounds(da, xdim="x", ydim="y", edge="reflect"):
    """Add bounds coordinates to a dataset. Only regular grid supported (1D coordinates).
    Parameters
    ----------
    da : xr.Dataset
      Xarray dataset defining grid centers
    xdim, ydim : str
      Name of the two coordinates.
    edge : {'reflect', 'mean', float}
      How the points at the edge are extrapolated.
      If 'reflect', the boundary grid steps are the same as their immediate neighbors.
      if 'mean', the boundary grid steps are the mean of all grid steps.
      if a number, it is used as the boundary grid step.
    Returns
    -------
    xr.Dataset
      A copy of da with new grid corners coordinates, they have the same name as the center
      coordinates with a '_b' suffix. Ex: lat_b and lon_b.
    """
    x = da[xdim]
    dx = x.diff(xdim, label="lower")
    if edge == "reflect":
        dx_left = dx[0]
        dx_right = dx[-1]
    elif edge == "mean":
        dx_left = dx_right = dx.mean()
    else:
        dx_left = dx_right = edge
    Xs = np.concatenate(([x[0] - dx_left / 2], x + dx / 2, [x[-1] + dx_right]))

    y = da[ydim]
    dy = y.diff(ydim, label="lower")
    if edge == "reflect":
        dy_left = dy[0]
        dy_right = dy[-1]
    elif edge == "mean":
        dy_left = dy_right = dy.mean()
    else:
        dy_left = dy_right = edge
    Ys = np.concatenate(([y[0] - dy_left / 2], y + dy / 2, [y[-1] + dy_right]))

    xdimb = xdim + "_b"
    ydimb = ydim + "_b"
    return da.assign(**{xdimb: ((xdimb,), Xs), ydimb: ((ydimb,), Ys)})


def compute_overlap_fraction(da, polys, dims=["x", "y"], poly_names=None):
    """Compute the area weights of each polygon on each gridcell

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
      A xarray object defining the centers (and optionally the corners) of a grid.
    polys : geopandas.GeoSeries or geopandas.GeoDataframe
      A Series of Polygons, with a defined crs
    dims : Sequence of str
      The names of the two coordinates defining the grid corners in da. Both must be 1D in da.
      If 4 names are passed, the last two are used as the grid corners
    Returns
    -------
    xr.DataArray
        The weights defined along dims[0] and dims[1], according to method mode, for each polygon.
        The first dimension is the same index as in polys.
    """

    assert da.rio.crs.to_epsg() == polys.crs.to_epsg(), "The crs of the grid and the polygons must be the same"

    xdim, ydim = dims
    xdimb, ydimb = xdim + "_b", ydim + "_b"
    da = add_bounds(da[dims], xdim=xdim, ydim=ydim, edge="reflect")

    weights = np.empty((polys.shape[0], da[xdim].size, da[ydim].size), dtype=float)

    if not poly_names:
        poly_names = polys.index.values

    for k, poly in enumerate(polys.geometry):
        xs = da[xdimb].values
        ys = da[ydimb].values
        for i, x in enumerate(da[xdim]):
            for j, y in enumerate(da[ydim]):
                # grid around node
                node_grid = Polygon(
                    [
                        (xs[i], ys[j]),
                        (xs[i], ys[j + 1]),
                        (xs[i + 1], ys[j + 1]),
                        (xs[i + 1], ys[j]),
                        (xs[i], ys[j]),
                    ]
                )
                if node_grid.intersects(poly):
                    w = node_grid.intersection(poly).area / node_grid.area
                else:
                    w = 0.0
                weights[k, i, j] = w
    desc = "The fraction of the gridcell that covers the polygon."
    coords = {crd: crdda for crd, crdda in da.coords.items() if all([dim in [xdim, ydim] for dim in crdda.dims])}
    coords["poly"] = poly_names
    return xr.DataArray(
        weights,
        coords=coords,
        dims=("poly", xdim, ydim),
        name="weights",
        attrs={"description": desc},
    )


def make_xarray_based(name, param, chunks=None):
    """
    Helper function to turn int/float/list into xarray Dataarray

    Parameters
    ----------
    name : str
        Name of the parameter
    param : int or float or list
        Parameter in original format

    Returns
    -------
    param: xarray.DataArray
        Parameter in xarray format
    """
    if isinstance(param, (int, float)):
        param = np.array([param])
    if isinstance(param, list):
        param = np.array(param)
    if isinstance(param, np.ndarray):
        param = xr.DataArray(param, coords={name: param})
    if chunks is not None:
        param = param.chunk({param.dims[0]: chunks})

    return param


def xr_flatten(da):
    """
    Flatten DataArray to a 1D array.

    Parameters
    ----------
    da: xarray.DataArray
        DataArray to be flattened.

    Returns
    -------
    flattened: np.array
        1D array containing the data of the DataArray
    """

    dim_names = list(da.dims)
    crd = [da[d] for d in dim_names]
    coords = dict(zip(dim_names, [("flat", a) for a in [b.flatten() for b in np.meshgrid(*crd, indexing="ij")]]))
    flattened = xr.DataArray(data=da.values.flatten(), coords=coords, dims="flat")
    return flattened


def xr_distance(a, b, dims=["x", "y"]):
    """
    Calculates distance between a and b along dimensions dims.

    Parameters
    ----------
    a : xr.DataArray
    b : xr.DataArray
    dims : list[str], optional

    Returns
    -------

    """


    sqr = 0.0
    for d in dims:
        sqr = sqr + (a[d] - b[d]) ** 2
    d = np.sqrt(sqr)
    return d


def xr_delay(a, b, dim="datetime"):
    """

    Parameters
    ----------
    a
    b
    dim : str, optional

    Returns
    -------

    """

    d = (b[dim] - a[dim]).astype(int) / (24 * 3.6e12)  # convert to fractional days
    return d


def _smooth_nan(U, sigma, mode="constant", cval=0.0):
    """Smooth a 2D array with NaNs.
    Based on
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291#36307291
    But actually entirely written by copilot

    Parameters
    ----------
    U : numpy.ndarray
        2D array with NaNs.
    sigma : float
        Standard deviation for the Gaussian kernel.
    mode : str, optional
        The mode parameter is passed to scipy.ndimage.filters.gaussian_filter.
        The default is "constant".
    cval : float, optional
        The cval parameter is passed to scipy.ndimage.filters.gaussian_filter.
        The default is 0.0.

    Returns
    -------
    numpy.ndarray
        Smoothed array.

    """
    V = U.copy()
    V[np.isnan(V)] = cval
    V = scpnd.gaussian_filter(V, sigma, mode=mode, cval=cval)
    W = np.ones(U.shape)
    W[np.isnan(U)] = 0.0
    W = scpnd.gaussian_filter(W, sigma, mode=mode, cval=cval)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = V / W
    return ret


def xr_smooth(
    dataarray,
    sigma,
    dims=["x", "y"],
    fill_value=0.0,
    ignore_nans=False,
):
    """Smooth a DataArray with NaNs.

    Parameters
    ----------
    dataarray : xarray.DataArray
        DataArray to smooth.
    sigma : float
        Standard deviation for the Gaussian kernel.
    dims : list, optional
        Dimensions to smooth. The default is ["x", "y"].
    fill_value : float, optional
        Value to use for NaNs. The default is 0.0.
    ignore_nans : bool, optional
        If True, NaNs are ignored, and the weighting of the weighted
        sum is based on the available points only. The default is False.
        In this case, the fill_value is used for NaNs.

    Returns
    -------
    xarray.DataArray
        Smoothed DataArray.

    """

    if ignore_nans:
        g_filter = _smooth_nan
    else:
        g_filter = scpnd.gaussian_filter
        dataarray = dataarray.fillna(fill_value)

    step = xr.DataArray([dataarray[d].diff(d).mean().values for d in dims], dims="loc")

    filtered_array = xr.apply_ufunc(
        g_filter,
        dataarray,
        sigma / step,
        kwargs={"mode": "constant", "cval": fill_value},
        input_core_dims=[dims, ["loc"]],
        output_core_dims=[dims],
        vectorize=True,
        keep_attrs=True,
    )

    return filtered_array


def xr_make_nondecreasing(array, dim):
    """


    Parameters
    ----------
    array
    dim

    Returns
    -------

    """

    return xr.apply_ufunc(
        np.maximum.accumulate,
        array,
        kwargs={"axis": -1},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        keep_attrs=True,
    )


def xr_calculate_rate(array, dim="datetime", label="lower"):
    """

    Parameters
    ----------
    array
    dim
    label

    Returns
    -------

    """
    s_diff = array.diff(dim, label=label)
    t_diff = array[dim].diff(dim, label=label).dt.days
    return s_diff / t_diff


# TODO: allowed for "out" specification where the result is to be stored
def samples_to_density_grid(
    samples,
    marginalize_dims,
    target_step,
    weights=None,
    target_start=None,
    target_stop=None,
    target=None,
    operator=np.add,
    order=1,
):
    """Convert samples to a density grid.

    Parameters
    ----------
    samples : xarray.DataArray or xarray.Dataset
        Samples to convert to a density grid. The samples are assumed to be
        stored in a DataArray with a dimension for each sample dimension and
        a dimension for the sample coordinates. The selection of variables or
        dimensions is determined by the target argument.
    marginalize_dims : list of str or str
        Dimensions to marginalize over.
    target_step : float or list of float
        Step size for the target grid.
    weights : xarray.DataArray, optional
        Weights for the samples. The weights are assumed to be stored in a
        DataArray with a dimension for each sample dimension and a dimension
        for the sample coordinates. If not provided, all samples are assumed to
        have equal weight of 1.0
    target_start : float, optional
        Start of the target grid. If not provided, the start of the target
        grid is determined from the samples.
    target_stop : float, optional
        Stop of the target grid. If not provided, the stop of the target
        grid is determined from the samples.
    target : str, list of str, optional
        Pointer to the target grid coordinates. This may be one dimension/axis of
        the supplied xarray.DataArray, or a variable name in the supplied xarray.Dataset.
    operator : callable, optional
        The operator used to combine the weights of samples that fall into the same grid cell.
        Default is np.add. Alternatively, np.multiply can be used to compute the product of
        weights, or np.fmax to compute the max.
        Other operators can be used as well, but they must supply the 'at' method.
    order : int, optional

    Returns
    -------
    grid : xarray.DataArray
        The density grid.
    """
    if weights is None:
        weights = xr.DataArray(1.0)

    if isinstance(samples, xr.Dataset):
        if target is None:
            target = list(samples.data_vars.keys())
        elif isinstance(target, str):
            target = [target]
        elif not isinstance(target, list):
            raise ValueError(
                f"target should be None, a string, or a list of strings, not {type(target)}"
            )

        samples = samples[target].to_array(dim="__var__")
        target = "__var__"

    elif isinstance(samples, xr.DataArray):
        if target is None:
            if samples.name is None:
                raise ValueError(
                    "either target should be specified, or the dataset should have its name attribute set"
                )
            samples = samples.expand_dims(dim={"__var__": [samples.name]})
        elif isinstance(target, str):
            if target not in samples.dims:
                # override dataarray name
                samples = samples.expand_dims(dim={"__var__": [target]})
        else:
            raise ValueError(
                f"target {target} not in samples dimensions {samples.dims}"
            )
        target = "__var__"

    if isinstance(marginalize_dims, str):
        marginalize_dims = [marginalize_dims]

    # if the marginalize_dims are present in both samples and weights,
    # they should be broadcasted to the same shape before flattening and further processing
    samples_has_marginalize_dims = np.intersect1d(samples.dims, marginalize_dims).size > 0
    weights_has_marginalize_dims = np.intersect1d(weights.dims, marginalize_dims).size > 0
    if samples_has_marginalize_dims and weights_has_marginalize_dims:
        exclude_dims = np.setdiff1d(weights.dims + samples.dims, marginalize_dims)
        samples, weights = xr.broadcast(samples, weights, exclude=exclude_dims)

    # flatten the samples and weights
    if samples_has_marginalize_dims:
        # not sure if there is a use case without this condition
        samples = (
            samples.reset_index(marginalize_dims)
            .stack({"__samples__": marginalize_dims})
            .transpose(..., "__samples__", target)
        )

    if weights_has_marginalize_dims:
        weights = (
            weights.reset_index(marginalize_dims)
            .stack({"__samples__": marginalize_dims})
            .transpose(..., "__samples__")
        )
        weight_core_dims = ["__samples__"]
    else:
        weight_core_dims = []
    target_dimensions = samples.coords[target].values

    start_full, stop_full = get_full_start_stop(
        samples.values, target_step, target_start
    )
    if target_start is None:
        target_start = start_full
    if target_stop is None:
        target_stop = stop_full

    start, step, stop = np.broadcast_arrays(target_start, target_step, target_stop)
    grid_size = np.ceil((stop - start) / step).astype(int) + 1

    if "__samples__" in weights.dims:
        weight_core_dims = ["__samples__"]
    else:
        weight_core_dims = []

    grid = xr.apply_ufunc(
        _samples_to_density_grid,
        samples,
        start,
        step,
        grid_size,
        weights,
        kwargs={"operator": operator, "order": order},
        input_core_dims=[["__samples__", target], [], [], [], weight_core_dims],
        exclude_dims=set(("__samples__", target)),
        output_core_dims=[target_dimensions],
    )
    target_size = grid.shape[-len(target_dimensions) :]

    coord_list = [
        strt + np.arange(sz) * stp for strt, sz, stp in zip(start, target_size, step)
    ]
    grid = grid.assign_coords(
        {dim: crd for dim, crd in zip(target_dimensions, coord_list)}
    )

    return grid


# TODO : automatically generate step values from the samples
# p=(10.**(np.floor(np.log10(d.values/20)).astype(int)))
# pp=p[None,:]*np.array([1,2,5])[:,None] # use steps of 1, 2, or 5 times the highest power of 10
# q=pp/(d.values/20) # find highest value lower than 1
def get_full_start_stop(samples, step, anchor=None):
    """Get the start and stop of a grid that covers all samples. The grid is defined by the
    step size and the (optional) anchor point. By default, the anchor point is 0.0.

    Parameters
    ----------
    samples : array_like
        The samples to be converted to a grid. The last dimension is assumed to be the dimension
        of the sample quantities, the second last dimension is assumed to be the dimension of
        the samples. If only one dimension is given, it is assumed that the samples are a 1D
        array of scalars. Allowed shapes: (..., Ns, Nd), (Ns, Nd), (Ns,).
    step : scalar or array_like
        The step size of the grid. Allowed shapes: (Nd,), ().
    anchor : scalar or array_like, optional
        The anchor point of the grid. Allowed shapes: (Nd,), (). Default is 0.0.
    Returns
    -------
    start : ndarray
        The start of the grid.
    stop : ndarray
        The stop of the grid.
    """

    if anchor is None:
        anchor = 0.0

    # convert inputs to numpy arrays
    samples = np.asarray(samples)
    step = np.asarray(step)
    anchor = np.asarray(anchor)

    # accommodate for the case where only one dimension is given
    if len(samples.shape) == 1:
        samples = samples[:, None]

    # determine location of samples in grid relative to anchor
    index = np.floor((samples - anchor) / step).astype(int)

    # determine extrema of grid index range
    axes = tuple(range(len(samples.shape) - 1))
    grid_min = np.asarray(np.min(index, axis=axes))
    grid_max = np.asarray(np.max(index, axis=axes) + 1)

    # determine discretized start and stop of grid
    start = anchor + grid_min * step
    stop = anchor + grid_max * step

    return start, stop


# following function is adapted from code by Reimer Weits, 2022
def _samples_to_density_grid(
    samples,
    target_start,
    target_step,
    target_shape=None,
    weights=None,
    operator=np.add,
    order=1,
    out=None,
):
    """Converts a set of samples in N-dimensions to a density on
    and N-dimensional grid.

    Parameters
    ----------
    samples : array_like
        The samples to be converted to a grid. The last dimension is assumed to be the dimension
        of the target quantities, the second last dimension is assumed to be the dimension of
        the samples. All other dimensions are maintained in the output. If weights are supplied,
        the dimension of the weights should be broadcastable to the dimension of the samples.
        If the target is only one-dimensional the lenght of this dimension should be 1. For
        convenience, this size 1 target dimension can be omitted if the samples are a 1D array of scalars.
        array of scalars. Allowed shapes: (..., Ns, Nd), (Ns, Nd), (Ns,).
    start : scalar or array_like
        The start point of the grid. Allowed shapes: (Nd,), ().
    step : scalar or array_like
        The step size of the grid. Allowed shapes: (Nd,), ().
    weights : array_like, optional
        The scalar weights of the samples. The last dimension is assumed to represent the samples.
        All other dimensions are assumed maintained in the output and should be broadcastable with the
        samples. If weights is not is not supplied,all samples are weighted equally with weight 1.
        Allowed shapes: (..., Ns), (Ns,), ().
        Default is 1.
    operator : callable, optional
        The operator used to combine the weights of samples that fall into the same grid cell.
        Default is np.add. Alternatively, np.multiply can be used to compute the product of
        weights, or np.fmax to compute the max.
        Other operators can be used as well, but they must supply the 'at' method.
    order : int, optional
        The order of the (inverse) interpolation. Default is 1. Order 0 is provided for convenience
        and consistency. It is not efficient, since it just rounds the 1D weights to either 0.0 or 1.0.
    Returns
    -------
    grid : ndarray
        The density grid.

    """
    # basic data topology
    samples = np.asarray(samples)
    target_step = np.asarray(target_step)
    target_start = np.asarray(target_start)

    # set default weights
    if weights is None:
        weights = np.asarray(1.0)
    else:
        weights = np.asarray(weights)

    # accommodate for the case where only one dimensional array of samples is given
    if len(samples.shape) == 1:
        samples = samples[:, None]

    # determine relevant sizes and shapes
    ndim_target = samples.shape[-1]
    sample_size = samples.shape[-2]
    assert ndim_target <= 8, "only up to 8 dimensions are supported"

    # cater for the maintained dimensions of both samples and weights
    maintained_shape, index_shape, weight_shape, output_shape = _get_shapes(
        samples, weights, target_shape
    )

    # generate indices of maintained dimensions
    maintained_indices = _generate_indices(sample_size, maintained_shape)

    # determine target shape
    if target_shape is None:
        if out is None:
            raise ValueError("either target_shape or out must be given")
        assert (
            len(out.shape) >= ndim_target
        ), "out must have at least ndim_target dimensions"
        target_shape = out.shape[-ndim_target:]
    else:
        assert out is None, "out and target_shape cannot be given at the same time"
        assert (
            len(target_shape) == ndim_target
        ), "target shape must have same length as samples"

    # check if out has correct shape
    if out is not None:
        assert (
            out.shape == output_shape
        ), f"out has wrong shape, should be {output_shape}"

    #
    target_start = np.broadcast_to(target_start, (ndim_target,))
    target_step = np.broadcast_to(target_step, (ndim_target,))
    target_size = np.asarray(target_shape)

    # determine location of samples in grid relative to start
    d, r = np.divmod(samples - target_start, target_step)
    grid_index = d.astype(int)
    frac = r / target_step  # normalize to [0, 1]

    # if request order is 0, round to nearest grid point
    if order == 0:
        frac = np.round(frac)

    # broadcast grid indices
    grid_index = np.broadcast_to(grid_index, index_shape)
    frac = np.broadcast_to(frac, index_shape)
    weights = np.broadcast_to(weights, weight_shape)

    full_grid_index = np.concatenate([maintained_indices, grid_index], -1)
    full_dim = full_grid_index.shape[-1]

    # filter out samples that are outside the grid
    flt = np.all(np.logical_and(grid_index >= 0, grid_index < target_size - 1), axis=-1)
    full_grid_index = full_grid_index[flt]
    frac = frac[flt]
    weights = weights[flt]

    # prepare output grid
    if out is None:
        out = np.zeros(output_shape)

    out = _burn_to_grid(
        full_grid_index, frac, weights, out, operator, ndim_target, full_dim
    )

    return out


def _generate_indices(sample_size, maintained_shape):
    maintained_indices = np.moveaxis(np.indices(maintained_shape), 0, -1)
    maintained_indices = np.expand_dims(maintained_indices, axis=-2)
    m_shape = maintained_shape + (sample_size, len(maintained_shape))
    maintained_indices = np.broadcast_to(maintained_indices, m_shape)
    return maintained_indices


def _get_shapes(samples, weights, target_shape):
    ndim_target = samples.shape[-1]
    sample_size = samples.shape[-2]

    maintained_shape = np.broadcast_shapes(samples.shape[:-2], weights.shape[:-1])
    index_shape = maintained_shape + (sample_size, ndim_target)
    weight_shape = maintained_shape + (sample_size,)
    output_shape = maintained_shape + tuple(target_shape)

    return maintained_shape, index_shape, weight_shape, output_shape


def _burn_to_grid(index, frac, weights, out, operator, ndim_target, full_dim):
    # index points to a corner of a hypercube
    # frac is the fractional distance from that corner to the sample
    # weights are the weights of the samples
    # out is the output grid
    # operator is the operator used to combine the weights of samples that fall into the same grid cell
    # ndim_target is the number of dimensions of the target grid

    # iterate over all hypercube corners by a bitwise representation
    # generate sequence of numbers from 0 to 2**ndim, representing all possible
    # combinations of 0 and 1 for ndim dimensions
    sequence = np.arange(2**ndim_target, dtype=np.uint8)

    # unpack the bits of the sequence into a matrix of 0 and 1
    allbits = np.unpackbits(sequence[:, None], axis=1).astype(int)
    offset_hypercube = allbits[:, -ndim_target:]
    ext_offset_hypercube = allbits[:, -full_dim:]

    # iterate over all corners of the hypercube, placing the contributions at the right
    # grid points
    for offset, ext_offset in zip(offset_hypercube, ext_offset_hypercube):
        # compute contribution at this corner
        multilinear_contribution = np.prod(
            ((frac) ** (offset)) * (1 - frac) ** (1 - offset), axis=1
        )
        weighted_contribution = weights * multilinear_contribution
        local_index = tuple((index + ext_offset).T)
        operator.at(out, local_index, weighted_contribution)

    return out
