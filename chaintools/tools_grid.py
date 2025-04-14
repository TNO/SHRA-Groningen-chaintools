import numpy as np
import xarray as xr
import scipy.ndimage as scpnd
import collections


# fraction of interval to ignore for rounding
ROUND_THRESHOLD = 1.0e-6


def bin_diff(value, dim, fill_value=None, direction=-1):
    """
    Calculate the difference of a 1-D value array along a dimension. The
    difference is calculated as the difference between the array values
    and the values shifted in the direction. The default direction is -1,
    which means the subtracted array is shifted to the left. This means
    that the difference is taken with respect to the value on the right.
    The opposite direction is 1, for which the difference is taken with
    respect to the value on the left. The fill_value fills the unknown points.

    Parameters
    ----------
    value : xr.DataArray
        Value array.
    dim : str
        Dimension along which to calculate the difference.
    fill_value : float, optional
        Fill value for the shifted array. The default is None.
    direction : int, optional
        Direction of the shift. The default is -1.

    Returns
    -------
    xr.DataArray
        Difference array.

    """
    if fill_value is None:
        shift_value = value.shift({dim: direction})
    else:
        shift_value = value.shift({dim: direction}, fill_value=fill_value)

    return value - shift_value


def bin_average(value, dim, fill_value=None, direction=-1):
    """
    Average consecutive points along a dimension. The averaging is performed
    by averaging with a left- or right-shifted array. The default direction is
    -1, which means the array is shifted to the left. This means that the average
    at each point is the average of the value on the right and the value itself.

    Parameters
    ----------
    value : xr.DataArray
        Value array.
    dim : str
        Dimension along which to average.
    fill_value : float, optional
        Fill value for the shifted array. The default is None.
    direction : int, optional
        Direction of the shift. The default is -1.

    Returns
    -------
    xr.DataArray
        Interpolated array.
    """
    if fill_value is None:
        shift_value = value.shift({dim: direction})
    else:
        shift_value = value.shift({dim: direction}, fill_value=fill_value)

    return 0.5 * (value + shift_value)


def expectation_by_parts(survival_probabilities, summand, dim, average=True):
    """
    Calculate the expectation value of the summand give the survival probabilities
    along a 1-D dimension. Rather than differencing the survival_probabilities, the
    summand is differenced. This is the principle of summation by parts. It can be
    useful when the array of survival probabilities is large relative to the summand.

    Parameters
    ----------
    survival_probabilities : xr.DataArray
        Survival probabilities.
    summand : xr.DataArray
        Summand.
    dim : str
        Dimension along which to sum.
    interpolate : bool, optional
        Whether to interpolate the summand at the bin centers. The default is True.

    Returns
    -------
    sum : xr.DataArray
        Sum of the product of the survival probabilities and the summand by parts.

    """
    # survival probabilities are defined on the bin edges
    # the summand has to be defined on the bin centers
    # calculate the summand at the bin centers
    # after averaging, the poe is NaN at the last point in dim
    if average:
        store = summand.isel({dim: -1})
        summand = bin_average(summand, dim)
        summand[{dim: -1}] = store

    # difference the summand
    # since we need the difference between the next value and the current,
    # we need to multiply by -1
    diff_summand = -1 * bin_diff(summand, dim)
    diff_summand[{dim: -1}] = 0.0

    # carry out the summation by parts
    return xr.dot(survival_probabilities, diff_summand)


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

    if param is None:
        return None
    param = np.atleast_1d(param)
    param = xr.DataArray(param, coords={name: param})
    if chunks is not None:
        param = param.chunk({name: chunks})

    return param


def xr_distance(a, b, spatial_coordinates=None):
    """
    Calculates distance between a and b along dimensions dims.

    Parameters
    ----------
    a : xr.DataArray
    b : xr.DataArray
    dims : list[str], optional

    Returns
    -------
    xr.DataArray
    """
    if spatial_coordinates is None:
        spatial_coordinates = ["x", "y"]

    sqr = 0.0
    for d in spatial_coordinates:
        sqr = sqr + (a[d] - b[d]) ** 2
    d = np.sqrt(sqr)
    return d


def xr_delay(a, b, variable="datetime"):
    """
    Calculates delay in fractional days between a datetime variable present in both a and b.

    Parameters
    ----------
    a : xr.Dataset or xr.DataArray
    b : xr.Dataset or xr.DataArray
    variable : str, optional


    Returns
    -------
    xr.DataArray

    """

    d = (b[variable] - a[variable]).astype(int) / (
        24 * 3.6e12
    )  # convert to fractional days
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
    dims=None,
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
    if dims is None:
        dims = ["x", "y"]

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
    Make an array nondecreasing along a dimension.

    Parameters
    ----------
    array: xarray.DataArray
        Array to make nondecreasing.
    dim: str
        Dimension to make nondecreasing.

    Returns
    -------
    xarray.DataArray
        Nondecreasing array.

    """

    return xr.apply_ufunc(
        np.maximum.accumulate,
        array,
        kwargs={"axis": -1},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        keep_attrs=True,
    )


def xr_calculate_rate(
    array, dim="datetime", label="lower", time_conversion_factor=None
):
    """
    Calculate the rate of change of an array along a time dimension.

    Parameters
    ----------
    array : xarray.DataArray
        Array to calculate the rate for.
    dim : str, optional
        Dimension along which to calculate the rate. The default is "datetime".
        The unit of the index coordinate is assumed to represent time.
    label : str, optional
        Label to use for the difference. The default is "lower".
    time_conversion_factor : float, optional
        Conversion factor for the time dimension. The default is None.
        If None, the rates is set in terms of sidereal year. If a value of
        1.0 is used, the rate is set in terms of seconds.

    Returns
    -------
    xarray.DataArray
        Rate in units determined by the time_conversion_factor.

    """
    if time_conversion_factor is None:
        SECONDS_IN_SIDEREAL_YEAR = 31_558_149.54
        time_conversion_factor = SECONDS_IN_SIDEREAL_YEAR
    s_diff = array.diff(dim, label=label)
    t_diff = array[dim].diff(dim, label=label).dt.total_seconds()
    return time_conversion_factor * s_diff / t_diff


def aggregate_to_grid(
    samples,
    target_step,
    weights=None,
    target_start=None,
    target_stop=None,
    target=None,
    marginalize_dims=None,
    operator=np.add,
    order=1,
):
    """Aggregate values / samples to a grid based on weights / measures.
    This operation is essentially equal to the construction of a histogram with bins
    of specified polynomial order.

    Parameters
    ----------
    samples : xarray.DataArray or xarray.Dataset
        Samples to aggregate to a grid. The samples are assumed to be
        stored in a DataArray with a dimension for each sample dimension and
        a dimension for the sample coordinates. The selection of variables or
        dimensions is determined by the target argument.
    marginalize_dims : list of str or str
        Dimensions to aggregate / marginalize over.
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

    # handle defaults
    if weights is None:
        weights = xr.DataArray(1.0)
    if marginalize_dims is None:
        marginalize_dims = []
    if isinstance(marginalize_dims, str):
        marginalize_dims = [marginalize_dims]

    # preprocess samples -> organize target in dedicated dimensions
    samples, tg_dim = _prepare_target(samples, target)

    # determine target grid dimensions
    tg_grid_dims = list(samples.coords[tg_dim].data)

    # determine broadcasted dimensions: dims that appear only in the weights
    # and are therefore fully broadcasted
    bc_dims = [d for d in weights.dims if d not in samples.dims]

    # if any of the marginalize_dims are present in the weights, then we do a full broadcast
    # over all sample dims -- this necessary to discard the weights that will be located
    # outside of the grid later on
    if set(weights.dims) & set(marginalize_dims):
        exclude_dims = set(bc_dims) | set([tg_dim])
        samples, weights = xr.broadcast(samples, weights, exclude=exclude_dims)

    # determine marginalized dimensions
    mrg_dims = set(marginalize_dims) & set(samples.dims)
    if mrg_dims < set(marginalize_dims):
        raise ValueError(f"marginalize_dims {set(marginalize_dims)-mrg_dims} not found")
    w_marginalize_dims = [d for d in marginalize_dims if d in weights.dims]

    # determine core dimensions for samples, weights and outputs
    samples_core_dims = tuple(marginalize_dims) + tuple([tg_dim])
    weights_core_dims = tuple(w_marginalize_dims) + tuple(bc_dims)
    output_core_dims = tuple(tg_grid_dims) + tuple(bc_dims)

    # determine target grid
    start, step, grid_size = _determine_target_grid(
        samples, target_step, target_start, target_stop, tg_dim
    )

    grid = xr.apply_ufunc(
        _aggregate_to_grid,
        samples,
        start,
        step,
        grid_size,
        weights,
        kwargs={
            "operator": operator,
            "order": order,
            "n_marginalize": len(mrg_dims),
            "n_broadcast": len(bc_dims),
        },
        input_core_dims=[samples_core_dims, [], [], [], weights_core_dims],
        exclude_dims=set(samples_core_dims),
        output_core_dims=[output_core_dims],
    )

    coord_list = [
        strt + np.arange(sz) * stp for strt, sz, stp in zip(start, grid_size, step)
    ]
    grid = grid.assign_coords({dim: crd for dim, crd in zip(tg_grid_dims, coord_list)})

    return grid


def _full_start_stop(samples, step, anchor=None):
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
    samples = np.atleast_2d(samples)
    step = np.asarray(step)
    anchor = np.asarray(anchor)

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


def _determine_target_grid(samples, target_step, target_start, target_stop, tg_dim):
    start_full, stop_full = _full_start_stop(
        samples.transpose(..., tg_dim).values, target_step, target_start
    )
    if target_start is None:
        target_start = start_full
    if target_stop is None:
        target_stop = stop_full
    start, step, stop = np.broadcast_arrays(
        np.atleast_1d(target_start),
        np.atleast_1d(target_step),
        np.atleast_1d(target_stop),
    )
    grid_size = np.ceil((stop - start) / step - ROUND_THRESHOLD).astype(int) + 1
    return start, step, grid_size


def _prepare_target(samples, target, tg_dim="__target__"):
    # convert dataset to single dataarray
    if isinstance(samples, xr.Dataset):
        if target is None:
            target = list(samples.data_vars.keys())
        elif isinstance(target, str):
            target = [target]
        elif not isinstance(target, list):
            raise ValueError(
                f"target should be None, a string, or a list of strings, not {type(target)}"
            )
        samples = samples[target].to_array(dim=tg_dim)

    # put target variables in a specific dimension
    elif isinstance(samples, xr.DataArray):
        if target is None:
            if samples.name is None:
                raise ValueError(
                    "either target should be specified, or the dataset should have its name attribute set"
                )
            samples = samples.expand_dims(dim={tg_dim: [samples.name]})
        elif isinstance(target, (str, collections.abc.Iterable)):
            if target not in samples.dims:
                targets = np.atleast_1d(target)
                samples = samples.expand_dims(dim={tg_dim: targets})
        else:
            raise ValueError(
                f"target {target} not in samples dimensions {samples.dims}"
            )

    return samples, tg_dim


def _aggregate_to_grid(
    samples,
    target_start,
    target_step,
    target_shape=None,
    weights=None,
    operator=np.add,
    order=1,
    out=None,
    **kwargs,
):
    """Converts a set of samples in N-dimensions to a density on
    and N-dimensional grid.

    Parameters
    ----------
    samples : array_like
        The samples to be aggregated to a grid. The last dimension is assumed to be the dimension
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
    samples = np.atleast_2d(samples)
    target_step = np.asarray(target_step)
    target_start = np.asarray(target_start)

    # set default weights
    if weights is None:
        weights = np.asarray(1.0)
    else:
        weights = np.asarray(weights)

    # determine target shape
    ndim_target = samples.shape[-1]
    assert ndim_target <= 8, "only up to 8 dimensions are supported"

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

    # cater for the maintained dimensions of both samples and weights
    n_mrg = kwargs.get("n_marginalize", 0)
    n_bc = kwargs.get("n_broadcast", 0)
    n_maintained = len(samples.shape) - n_mrg - 1
    mt_shape = samples.shape[:n_maintained]
    bc_shape = weights.shape[-n_bc:] if n_bc > 0 else ()
    output_shape = mt_shape + tuple(target_shape) + bc_shape

    # prepare output grid
    if out is None:
        out = np.zeros(output_shape)
    else:
        assert (
            out.shape == output_shape
        ), f"out has wrong shape, should be {output_shape}"

    # broadcast target start and step
    target_start = np.broadcast_to(target_start, (ndim_target,))
    target_step = np.broadcast_to(target_step, (ndim_target,))
    target_size = np.asarray(target_shape)

    # determine location of samples in grid relative to start
    if order == 0:
        grid_index = np.rint((samples - target_start) / target_step).astype(int)
    elif order == 1:
        d, r = np.divmod(samples - target_start, target_step)
        grid_index = d.astype(int)
        frac = r / target_step  # normalize to [0, 1]
    else:
        raise ValueError("only order 0 and 1 are supported")

    # set up the indices for the maintained sample dimensions
    outer_index = np.indices(samples.shape[:-1])

    # filter out samples that are outside the grid
    flt = np.all(np.logical_and(grid_index >= 0, grid_index < target_size - 1), axis=-1)
    grid_index = grid_index[flt]
    outer_index = outer_index[:n_maintained, flt]
    if len(weights.shape) >= len(flt.shape):
        weights = weights[flt]

    if order == 0:
        out = _burn_to_grid_0(grid_index, weights, out, operator, outer_index)
    elif order == 1:
        frac = frac[flt]
        out = _burn_to_grid_1(grid_index, frac, weights, out, operator, outer_index)

    return out


def _burn_to_grid_0(index, weights, out, operator, outer_index):
    # zeroth order binning / inverse nearest neighbour interpolation
    # index points to the bin index
    # weights are the weights of the samples
    # out is the output grid
    # operator is the operator used to combine the weights of samples that fall into the same grid cell
    # outer_index is the index of the maintained dimensions

    inner_index = np.moveaxis(index, -1, 0)
    local_index = tuple(outer_index) + tuple(inner_index)
    operator.at(out, local_index, weights)

    return out


def _burn_to_grid_1(index, frac, weights, out, operator, outer_index):
    # first order binning / inverse linear interpolation
    # index points to the "lowerleft" corner of a hypercube
    # frac is the fractional distance from that corner to the sample
    # weights are the weights of the samples
    # out is the output grid
    # operator is the operator used to combine the weights of samples that fall into the same grid cell
    # outer_index is the index of the maintained dimensions

    # iterate over all hypercube corners by a bitwise representation
    # generate sequence of numbers from 0 to 2**ndim, representing all possible
    # combinations of 0 and 1 for ndim dimensions
    ndim_target = index.shape[-1]
    sequence = np.arange(2**ndim_target, dtype=np.uint8)

    # unpack the bits of the sequence into a matrix of 0 and 1
    allbits = np.unpackbits(sequence[:, None], axis=1).astype(int)
    offset_hypercube = allbits[:, -ndim_target:]

    n_mt = len(outer_index)
    n_bc = len(out.shape) - ndim_target - n_mt
    new_axes = n_bc * (np.newaxis,)

    # iterate over all corners of the hypercube, placing the contributions at the right
    # grid points
    for offset in offset_hypercube:
        # compute contribution at this corner
        multilinear_contribution = np.prod(
            ((frac) ** (offset)) * (1 - frac) ** (1 - offset), axis=1
        )[..., *new_axes]
        weighted_contribution = weights * multilinear_contribution
        offset_index = np.moveaxis(index + offset, -1, 0)
        local_index = tuple(outer_index) + tuple(offset_index)
        operator.at(out, local_index, weighted_contribution)

    return out


def coarsen_stacked(stacked, coarsening_factor, stacked_dim="loc", dims=None):
    if dims is None:
        dims = ["x", "y"]

    if coarsening_factor <= 1:
        coarse_stacked = stacked
    else:
        unstacked = stacked.unstack(stacked_dim).sortby(dims)
        unstacked = unstacked.coarsen(
            {d: coarsening_factor for d in dims}, boundary="pad"
        ).sum()
        coarse_stacked = unstacked.stack({stacked_dim: dims})
        coarse_stacked = coarse_stacked.dropna(stacked_dim)

    return coarse_stacked
