import collections
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import warnings
import uuid
from pathlib import Path
from copy import deepcopy
import yaml
import builtins
import rioxarray
from typing import Union


def assign_dims(dim_spec, xarray_ds=None):
    if xarray_ds is None:
        xarray_ds = xr.Dataset()

    dim_dict = {}
    for dim, spec in dim_spec.items():
        if isinstance(spec, collections.abc.Mapping):
            if "values" in spec:
                dim_dict[dim] = np.atleast_1d(spec["values"])
            elif "interval" in spec:
                dim_dict[dim] = range_from_dict(spec)
            else:
                dim_dict[dim] = range(spec.get("length", 0))
        else:
            raise SystemError(f"unknown dimension specification {spec}")

    new_ds = xarray_ds.expand_dims(dim_dict)

    for dim, spec in dim_spec.items():
        new_ds[dim].attrs.update(spec)

    return new_ds


def assign_coords(coord_spec, xarray_ds):
    coord_dict = {}
    attrs_dict = {}
    for coord, spec in coord_spec.items():
        # spec can only be a dict
        dim = spec.get("dim", None)
        if dim is None:
            raise SystemError(f"no dimension specified for coordinate {coord}")
        if dim not in xarray_ds.dims:
            raise SystemError(f"dimension {dim} not in xarray dataset")
        len = xarray_ds.sizes[dim]
        if "values" in spec:
            range = spec["values"]
            if range.size != len:
                raise SystemError(
                    f"length of values for coordinate {coord} does not match dimension {dim}"
                )
            coord_dict[coord] = (dim, spec["values"])
        else:
            range = range_from_dict(spec, len=len)
            coord_dict[coord] = (dim, range)
        attrs_dict[coord] = spec

    new_ds = xarray_ds.assign_coords(coord_dict)
    for k, v in attrs_dict.items():
        new_ds[k].attrs.update(v)

    return new_ds


def range_from_dict(spec, len=None):
    interval = spec.get("interval")
    spacing = spec.get("sequence_spacing", "linear")
    multiplier = spec.get("multiplier", 1.0)
    offset = spec.get("offset", 0.0)
    if len is None:
        len = spec.get("length", 2)

    if spacing == "linear":
        range = np.linspace(interval[0], interval[1], len)
    elif spacing in ["log", "exp", "geometric"]:
        range = np.geomspace(interval[0], interval[1], len)
    elif spacing == "lin_values_to_log10":
        range = np.linspace(np.log10(interval[0]), np.log10(interval[1]), len)
    else:
        raise SystemError(f"unknown sequence spacing {spacing}")

    return multiplier * (offset + range)


def select_from_dims(spec, xarray_ds):
    selection = {}
    for k, v in spec.items():
        if k in xarray_ds.dims and xarray_ds.sizes[k] > 0:
            selection[k] = v

    return selection


def chunk(xarray_ds, chunk_spec):
    if not chunk_spec is None:
        chunk_spec = select_from_dims(chunk_spec, xarray_ds)
        xarray_ds = xarray_ds.chunk(chunk_spec)

    return xarray_ds


def sel(xarray_ds, sel_spec):
    sel_spec = select_from_dims(sel_spec, xarray_ds)
    xarray_ds = xarray_ds.sel(sel_spec)

    return xarray_ds


def drop_sel(xarray_ds, sel_spec):
    sel_spec = select_from_dims(sel_spec, xarray_ds)
    xarray_ds = xarray_ds.drop_sel(sel_spec)

    return xarray_ds


def isel(xarray_ds, isel_spec):
    isel_spec = select_from_dims(isel_spec, xarray_ds)
    xarray_ds = xarray_ds.isel(isel_spec)

    return xarray_ds


def drop_isel(xarray_ds, isel_spec):
    isel_spec = select_from_dims(isel_spec, xarray_ds)
    xarray_ds = xarray_ds.drop_isel(isel_spec)

    return xarray_ds


def sel_slice(xarray_ds, slice_spec):
    slice_spec = select_from_dims(slice_spec, xarray_ds)
    slice_spec = {k: slice(*v) for k, v in slice_spec.items()}
    xarray_ds = xarray_ds.sel(slice_spec)

    return xarray_ds


def isel_slice(xarray_ds, islice_spec):
    islice_spec = select_from_dims(islice_spec, xarray_ds)
    islice_spec = {k: slice(*v) for k, v in islice_spec.items()}
    xarray_ds = xarray_ds.isel(islice_spec)

    return xarray_ds


def thin(xarray_ds, thin_spec):
    thin_spec = select_from_dims(thin_spec, xarray_ds)
    xarray_ds = xarray_ds.thin(thin_spec)

    return xarray_ds


def construct_path(path_spec, root_path=None):
    if path_spec is None:
        return None
    if isinstance(path_spec, str):
        path = Path(path_spec)
    elif isinstance(path_spec, collections.abc.Sequence):
        path = Path().joinpath(*path_spec)
    if root_path is not None and not path.is_absolute():
        path = Path(root_path) / path

    return path


def data_source(**kwargs):
    kwargs_full = deepcopy(kwargs)
    type = kwargs.pop("type", None)
    if type is None:
        raise SystemError("No data source type specified")
    variable = kwargs.pop("variable", None)
    to_array = kwargs.pop("to_array", None)
    to_dataset = kwargs.pop("to_dataset", None)
    rename = kwargs.pop("rename", None)
    drop_vars = kwargs.pop("drop_vars", None)
    spatial_dims = kwargs.pop("spatial_dims", None)
    crs = kwargs.pop("crs", None)
    to_geopandas = kwargs.pop("to_geopandas", False)

    xarray_function = {
        "xarray_dataset": xr.open_dataset,
        "xarray_dataarray": xr.open_dataarray,
        "xarray_datatree": xr.open_datatree,
        # "xarray_mfdataset": xr.open_mfdataset,  # also add {parallel : True} to the config
    }

    if type in xarray_function.keys():
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        if "group" in kwargs:
            pth = construct_path(kwargs["group"])
            if pth is None:
                kwargs.pop("group")
            else:
                kwargs["group"] = str(pth.as_posix())
        path = construct_path(kwargs.pop("path"), kwargs.pop("root_path", None))
        if not "engine" in kwargs:
            if path.suffix == ".zarr" or path.suffix == ".zip":
                kwargs["engine"] = "zarr"
            else:
                kwargs["engine"] = "h5netcdf"
        source = xarray_function[type](path, **kwargs)
    elif type == "xarray_dataarray_inline":
        source = xr.DataArray.from_dict(kwargs["from_dict"])
    elif type == "xarray_dataset_inline":
        source = xr.Dataset.from_dict(kwargs["from_dict"])
    elif type in ("csv", "pandas", "table"):
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        path = construct_path(kwargs.pop("path"), kwargs.pop("root_path", None))
        if path.suffix == ".csv":
            source = pd.read_csv(path, **kwargs)
        elif path.suffix == ".xls" or path.suffix == ".xlsx":
            source = pd.read_excel(path, **kwargs)
        elif path.suffix == ".json":
            source = pd.read_json(path, **kwargs)
        else:
            source = pd.read_table(path, **kwargs)
    elif type == "geopandas":
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        path = construct_path(kwargs.pop("path"), kwargs.pop("root_path", None))
        dtype = kwargs.pop("dtype", None)
        index_col = kwargs.pop("index_col", None)
        source = gpd.read_file(path, **kwargs)
        if dtype is not None:
            for k, v in dtype.items():
                source[k] = source[k].astype(v)
        if index_col is not None:
            source.set_index(index_col, inplace=True)
    elif type in ("yml", "yaml"):
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        path = construct_path(kwargs.pop("path"), kwargs.pop("root_path", None))
        with builtins.open(path) as stream:
            xr_dict = yaml.load(stream, Loader=yaml.SafeLoader)
        if "data_vars" in xr_dict:
            source = xr.Dataset.from_dict(xr_dict)
        else:
            source = xr.DataArray.from_dict(xr_dict)
    elif type in ("path", "custom", "raw"):
        # just return the path to the file
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        source = construct_path(kwargs.pop("path"), kwargs.pop("root_path", None))
    else:
        raise SystemError(f"unknown data source type {type}")

    if isinstance(source, pd.DataFrame):
        if isinstance(source, gpd.GeoDataFrame):
            crs = source.crs
        elif crs is not None:
            x_id, y_id = spatial_dims
            xy = gpd.points_from_xy(source[x_id], source[y_id], crs=crs)
            source = gpd.GeoDataFrame(source, geometry=xy)
            # rename columns
            source = source.rename(columns={x_id: "x", y_id: "y"})
        source = xr.Dataset.from_dataframe(source)

    if isinstance(source, (xr.Dataset, xr.DataArray)):
        source = restore_multiindex(source)
        if crs is not None:
            source.rio.write_crs(crs, inplace=True)
        if isinstance(source, xr.Dataset):
            if variable is not None:
                source = source[variable]
            if drop_vars is not None:
                source = source.drop_vars(drop_vars)
            if to_array is not None:
                source = source.to_array(dim=to_array)
            if rename is not None:
                source = source.rename(rename)

        if isinstance(source, xr.DataArray):
            if to_dataset is not None:
                source = source.to_dataset(dim=to_dataset)

    if to_geopandas:
        source = to_geopandas(source, spatial_dims=spatial_dims, crs=crs)

    return source


def to_geopandas(source, spatial_dims=None, crs=None, to_crs=None):
    if crs is None:
        crs = source.rio.crs
    if "geometry" in source:
        geometry = source["geometry"].values
    else:
        if spatial_dims is None:
            spatial_dims = source.rio.spatial_dims
        x_id, y_id = spatial_dims
        geometry = gpd.points_from_xy(source[x_id], source[y_id], crs=crs)
    source = gpd.GeoDataFrame(source.to_dataframe(), geometry=geometry, crs=crs)
    if to_crs:
        source = source.to_crs(to_crs)
    return source


def open(name, config, default=None, **kwargs):
    # within a module we open data sources
    # in a more generic context we allow opening any data store
    section_names = ["data_stores", "data_sources", "data_sinks"]
    spec = None
    for s in section_names:
        section = config.get(s, {})
        if name in section:
            spec = section[name]
            break
    if spec is None:
        return default  # burden on user

    open_result = _open(spec, config, **kwargs)
    if isinstance(open_result, list):
        if "concat" in kwargs:
            open_result = xr.concat(open_result, dim=kwargs["concat"])
        elif kwargs.get("merge", True):
            open_result = xr.merge(open_result)

    return open_result


def _open(spec, config, **kwargs):
    if isinstance(spec, collections.abc.Sequence):
        result = [_open(sp, config, **kwargs) for sp in spec]
    elif isinstance(spec, collections.abc.Mapping):
        kwargs = make_group(spec, kwargs)
        kwargs = spec | kwargs
        chunking_allowed = kwargs.pop("chunking_allowed", True)

        # "local" selection first. Local selection criteria are popped, or data_source will trip over selection keys
        sel_dict = construct_select_dict(kwargs, sanitize=True)
        ds = data_source(**kwargs)
        for key in sel_dict:
            sel_func = get_select_func(key)
            ds = sel_func(ds, sel_dict[key])

        # "global" selection second
        sel_dict = construct_select_dict(config, sanitize=False)
        for key in sel_dict:
            sel_func = get_select_func(key)
            ds = sel_func(ds, sel_dict[key])

        chunk_spec = get_chunk_spec(config)
        if chunk_spec is not None and chunking_allowed:
            ds = chunk(ds, chunk_spec)
        result = ds
    else:
        raise SystemError(f"unknown dimension specification {spec}")

    return result


def get_chunk_spec(config):
    if "chunk" in config:
        return config["chunk"]
    elif "dask" in config and config["dask"] is not False:
        if "chunk" in config["dask"]:
            return config["dask"]["chunk"]
    return None  # no chunking


def open_tree(name, config, **kwargs):
    local_config = deepcopy(config)
    local_config["data_stores"][name]["type"] = "xarray_datatree"
    local_config["data_stores"][name].pop("group")
    ds = open(name, local_config, **kwargs)

    return ds


def dismantle_multiindex(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Remove multiindices from ds and store them in attributes

    :param ds:
    :return:
    """
    # part 1: identify multiindex
    stacked_indexes = identify_multiindex(ds)
    if not stacked_indexes:
        return ds

    multiindex_dict = {}
    for multiindex in stacked_indexes:
        dict_key = "_MultiIndex_" + multiindex
        multiindex_dict[dict_key] = [
            d for d in stacked_indexes[multiindex][1].keys() if d != multiindex
        ]

    # part 2: dismantle multiindex
    ds = ds.reset_index(list(stacked_indexes.keys()))

    # part 3: add multiindex info as attribute
    ds = ds.assign_attrs(multiindex_dict)

    return ds


def identify_multiindex(ds: xr.Dataset | xr.DataArray) -> dict:
    """
    Identify multiindices and store in dictionary
    :param ds:
    :return:
    """

    # if xr.DataArray, convert temporarily to dataset
    if isinstance(ds, xr.DataArray):
        ds = ds._to_temp_dataset()

    # identify MultiIndexes:
    stacked_indexes = {}
    for d in ds.dims:
        idx, idx_vars = ds._get_stack_index(d, multi=True)
        if idx is not None:
            stacked_indexes[d] = idx, idx_vars

    return stacked_indexes


def restore_multiindex(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Restore multiindex coordinates from saved attributes
    :param ds:
    :return:
    """
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        return ds

    identifier = "_MultiIndex_"
    multiindex_list = [a for a in ds.attrs.keys() if identifier in a]

    for multiindex in multiindex_list:

        # case 1: Multiindex has only one index
        if isinstance(ds.attrs[multiindex], str):
            mi_name = multiindex.replace(identifier, "")
            mi_index = ds.attrs.pop(multiindex)
            ds = (
                ds.set_index({mi_name: mi_index})
                .rename({mi_name: mi_index})
                .stack({mi_name: [mi_index]})
            )
        # case 2: Multiindex has multiple indices
        elif isinstance(ds.attrs[multiindex], list):
            ds = ds.set_xindex(ds.attrs.pop(multiindex))
        else:
            raise TypeError("Unkown type in _MultiIndex_ attribute")

    return ds


def sel_all_coords(ds: Union[xr.Dataset, xr.DataArray], sel_dict, method=None, tolerance=None, drop=False):

    if not sel_dict:
        return ds

    # if xr.DataArray, convert temporarily to dataset
    return_da = False
    if isinstance(ds, xr.DataArray):
        ds = ds._to_temp_dataset()
        return_da = True

    non_index_coords = {c: not isinstance(ds.coords[c]._variable, xr.IndexVariable) for c in ds._coord_names}
    sel_non_index_coords = [k for k in sel_dict if non_index_coords[k]]
    ds = ds.set_xindex(sel_non_index_coords)
    ds = ds.sel(sel_dict, method, tolerance, drop)
    ds = ds.drop_indexes(sel_non_index_coords)

    if return_da:
        ds = ds.to_array(dim='__temp__').isel({'__temp__':0}, drop=True)

    return ds


def store(ds, name, config, **kwargs):
    kwargs = make_group(config["data_sinks"][name], kwargs)
    kwargs = config["data_sinks"][name] | kwargs
    _ = kwargs.pop("type")
    path = construct_path(kwargs.pop("path"))
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(ds, xr.DataArray):
        # we force ds into a dataset
        # this will raise an error if ds has no name
        # but we do not want to allow unnamed data arrays anyway
        ds = ds.to_dataset()

    # dismantle multiindex and store remantling data in attrs
    ds = dismantle_multiindex(ds)

    # generate unique identifier
    ds.attrs["uuid"] = uuid.uuid4().hex

    if path.suffix == ".zarr":
        kwargs.setdefault("consolidated", True)
        kwargs.setdefault("mode", "a")
        result = ds.drop_encoding().to_zarr(path, **kwargs)
    else:
        # Ensure .zarr arguments 'append_dim' and 'mode'='w-' are removed, as they are not technically possible to
        # use in .h5 files.
        # For 'append_dim' give a warning, because it only returns one iteration over the dimension.
        append_dimension = kwargs.pop("append_dim", False)
        if "mode" in kwargs and kwargs["mode"] == "w-":
            kwargs["mode"] = "w"
        if append_dimension is not False:
            warnings.warn(
                f"Storing data in .h5 format instead of intended .zarr format. "
                f"Only one iteration over dimension {append_dimension} will be stored.",
                category=UserWarning,
            )
        if not "engine" in kwargs:
            kwargs["engine"] = "h5netcdf"
        result = ds.drop_encoding().to_netcdf(path, **kwargs)

    return result


def make_group(config, kwargs):
    group = None
    if "group" in config:
        group = construct_path(config["group"])
    if "group" in kwargs:
        if group is None:
            group = construct_path(kwargs["group"])
        else:
            group = group / construct_path(kwargs["group"])
    if group is not None:
        kwargs["group"] = str(group)

    return kwargs


def prepare_ds(config):
    drop_vars = [
        d
        for d, v in config["dimensions"].items()
        if "interval" not in v and "values" not in v
    ]
    ds = assign_dims(config["dimensions"])
    if "coordinates" in config:
        ds = assign_coords(config["coordinates"], ds)
    ds = ds.drop_vars(drop_vars)

    return ds


def unpack(da):
    return da.values[()]


def xr_clean_coords(
    ds: xr.Dataset | xr.DataArray, optional_names=None
) -> xr.Dataset | xr.DataArray:
    """
    Remove all coordinates which are not set on dimensions.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
    optional_names : Optional names to be removed

    Returns
    -------
    cleaned xr.Dataset or xr.DataArray

    """
    if optional_names is None:
        optional_names = []

    ds = ds.drop_vars(
        [coord for coord in ds.coords if coord not in list(ds.dims)] + optional_names
    )

    return ds


def prepare_weights(weights, *integrands):
    # if no weights are provided, return empty dataset and empty list
    if weights is None:
        return xr.Dataset(), set()

    # collect all dimensions that support any component of the weight distribution
    weight_support_dims, weight_batch_dims = get_support_batch_dims(weights)

    # get inventory of dimensions present in the integrands, which are all assumed to be
    # batch dimensions (i.e., not supporting a distribution)
    batch_dims = set(weight_batch_dims)
    for ig in integrands:
        batch_dims |= set(ig.dims)

    # identify which support dimensions are to be marginalized and which
    # support dims are to be ignored since there is no corresponding batch dimension
    # in the integrands or in the weights themselves
    marginalize_dims = weight_support_dims & batch_dims
    support_dims_to_ignore = weight_support_dims - marginalize_dims

    # TODO: what is the proper thing to do here? as there may be multivariate weight
    # distributions that are only partially present in the integrands
    # we can either drop the entire distribution or keep it and sum over the missing dimensions
    prepared_weights = weights.sum(support_dims_to_ignore)

    return prepared_weights, marginalize_dims


def get_support_batch_dims(weights):
    # the support dimensions carry the support of the weight distribution
    # other dimensions identify different variations of the weight distribution
    # i.e., conditional dimensions, or, in the parlance of PyMC, batch dimensions
    # the weights variables are expected to carry an attribute "batch_dims" that
    # lists the batch dimensions if present
    global_support_dims = set()
    global_batch_dims = set()
    for v in weights.values():
        dims = set(v.dims)
        support_dims = set(np.atleast_1d(v.attrs.get("support_dims", [])))
        batch_dims = dims - support_dims
        if support_dims & global_support_dims:
            raise ValueError(
                "Support dimensions must be unique across weight variables"
            )
        global_support_dims |= support_dims
        global_batch_dims |= batch_dims
    return global_support_dims, global_batch_dims


def add_suffix(ds, suffix):
    return ds.rename({name: f"{name}{suffix}" for name in ds})


def construct_select_dict(dictionary: dict, sanitize=False) -> dict:
    # get select functions whilst preserving order
    select_funcs = ["filter", "slice", "isel", "islice", "sel", "thin", "drop_sel"]
    dictionary_out = {k: v for k, v in dictionary.items() if k in select_funcs}

    # optional: remove full keys from original dictionary
    if sanitize:
        for key in dictionary_out.keys():
            dictionary.pop(key)

    return dictionary_out


def get_select_func(func_key: str):

    selec_func_dict = {
        "filter": sel,
        "sel": sel,
        "slice": sel_slice,
        "isel": isel,
        "islice": isel_slice,
        "thin": thin,
        "drop_sel": drop_sel,
        "drop_isel": drop_isel,
    }
    return selec_func_dict[func_key]


def weighted_sum(rate, weights, rate_multiplier=None):
    weights, marginalize_dims = prepare_weights(weights, rate)
    if rate_multiplier is None:
        rate_multiplier = xr.DataArray(1.0)

    # marginalize over the core dimensions of the provided weights
    summed_rate = xr.dot(
        rate,
        rate_multiplier,
        *weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )

    return summed_rate
