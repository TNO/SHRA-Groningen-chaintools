"""
Tools for computing weighted fractiles of data.
"""

import numba
import numpy as np
import xarray as xr


def xr_weighted_fractiles(data, weights, fractiles, dim):
    """
    Compute weighted fractiles of data along a specified dimension.

    Parameters
    ----------
    data : xr.DataArray
        Data values to compute fractiles of.
    weights : xr.DataArray
        Weights to use for computing fractiles.
    fractiles : xr.DataArray
        Fractiles to compute.
    dim : str
        Dimension to compute fractiles along.

    Returns
    -------
    xr.DataArray
        Fractile values
    """

    # wrap weighted_fractiles_ufunc to work with xarray DataArrays
    return xr.apply_ufunc(
        _weighted_fractiles_ufunc,
        data,
        weights,
        fractiles,
        input_core_dims=[[dim], [dim], fractiles.dims],
        output_core_dims=[fractiles.dims],
        exclude_dims={dim},
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"output_sizes": fractiles.sizes},
    )


def _weighted_fractiles_ufunc(data, weights, fractiles):
    # Compute fractiles of weighted data
    # Assumes data and weights have a common last dimension and are
    # properly broadcasted for use in np.take_along_axis
    # The weights are expected to be normalized such that they sum to 1
    # for the output shape the last dimension is replaced by the dimensions of fractiles

    # Determine sorting order for data and apply that order to the weights
    order = np.argsort(data, axis=-1)  # shape of data
    weights, order = np.broadcast_arrays(
        weights, order
    )  # np.take_along_axis expects same dimensions
    weights_sorted = np.take_along_axis(weights, order, axis=-1)  # shape of data

    # Compute cumulative sum  (CDF) of sorted, normalized weights
    weights_cumsum = np.cumsum(weights_sorted, axis=-1)  # shape of data
    weights_cumsum /= weights_cumsum[..., -1:]  # normalize to 1

    # Find the insertion indices for the fractiles in the CDF
    sorted_indices = _searchsorted_ufunc(weights_cumsum, fractiles)  # shape of output

    # Use the indices to look up the fractile values in the data
    # Approach in two steps so we don't have to shuffle the entire data array
    original_indices = np.take_along_axis(
        order, sorted_indices, axis=-1
    )  # shape of output
    fractile_values = np.take_along_axis(
        data, original_indices, axis=-1
    )  # shape of output

    return fractile_values


@numba.guvectorize(
    [
        "void(float32[:], float32[:], int64[:])",
        "void(float64[:], float64[:], int64[:])",
    ],
    "(n),(m)->(m)",
    cache=True,
    target="parallel",
    nopython=True,
)
def _searchsorted_ufunc(da, v, res):  # pragma: no cover
    """Use :func:`numba.guvectorize` to convert numpy searchsorted into a vectorized ufunc."""
    res[:] = np.searchsorted(da, v, side="left")
