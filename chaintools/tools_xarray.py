import collections
import numpy as np
import xarray as xr
import datatree as dt
from pathlib import Path


def assign_dims(dim_spec, xarray_ds=None):
    if xarray_ds is None:
        xarray_ds = xr.Dataset()

    dim_dict = {}
    for dim, spec in dim_spec.items():
        # if range_spec is a list return list, if range_spec is a dict, interpret dict
        if isinstance(spec, collections.abc.Sequence):
            dim_dict[dim] = spec

        elif isinstance(spec, collections.abc.Mapping):
            dim_dict[dim] = range_from_dict(spec)

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
        len = xarray_ds.dims[dim]
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
    interval = spec.get("interval", [0.0, 1.0])
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


def select_from_dims(filters, xarray_ds):
    selection = {}
    for k, v in filters.items():
        if k in xarray_ds.dims:
            selection[k] = v

    return selection


def chunk(xarray_ds, chunk_spec):
    chunks = select_from_dims(chunk_spec, xarray_ds)
    xarray_ds = xarray_ds.chunk(chunks)

    return xarray_ds


def filter(xarray_ds, filter_spec):
    filters = select_from_dims(filter_spec, xarray_ds)
    xarray_ds = xarray_ds.sel(filters)

    return xarray_ds


def thin(xarray_ds, thinning_spec):
    thinnings = select_from_dims(thinning_spec, xarray_ds)
    xarray_ds = xarray_ds.thin(thinnings)

    return xarray_ds


def construct_path(path_spec):
    if isinstance(path_spec, str):
        path = Path(path_spec)
    elif isinstance(path_spec, collections.abc.Sequence):
        path = Path().joinpath(*path_spec)

    return path


def data_source(**kwargs):
    kwargs_full = kwargs.copy()
    type = kwargs.pop("type", None)
    if type is None:
        raise SystemError("No data source type specified")

    xarray_function = {
        "xarray_dataset": xr.open_dataset,
        "xarray_dataarray": xr.open_dataarray,
        "xarray_datatree": dt.open_datatree,
        "xarray_mfdataset": xr.open_mfdataset,  # also add {parallel : True} to the config
    }

    if type in xarray_function.keys():
        if "path" not in kwargs:
            raise SystemError(f"no file/path specified for data source {kwargs_full}")
        path = construct_path(kwargs.pop("path"))
        if not "engine" in kwargs:
            if path.suffix == ".zarr":
                kwargs["engine"] = "zarr"
            else:
                kwargs["engine"] = "h5netcdf"
        source = xarray_function[type](path, **kwargs)
    elif type == "inline":
        data = kwargs.pop("data", None)
        if data is None:
            raise SystemError(f"no data specified for inline data_source {kwargs_full}")
        ds_dict = {}
        for k, v in data.items():
            vals = np.array(list(v.values()))
            coords = np.array(list(v.keys()))
            # TODO : the following is too arbitrary; why "weight"?
            ds_dict[k + "_weight"] = xr.DataArray(vals, coords={k: coords})
        source = xr.Dataset(ds_dict)

    return source


def open(name, config, **kwargs):
    # within a module we open data sources
    # in a more generic context we allow opening any data store
    group_name = "data_sources"
    if group_name not in config:
        group_name = "data_stores"
    chunking_allowed = kwargs.pop("chunking_allowed", True)
    ds = (
        data_source(**config[group_name][name], **kwargs)
        .pipe(filter, config.get("filters", {}))
        .pipe(thin, config.get("thinnings", {}))
    )
    # chunking with empty dict is not a non-op
    if "chunks" in config and chunking_allowed:
        ds = chunk(ds, config["chunks"])

    return ds


def store(ds, name, config, **kwargs):
    kwargs = config["data_sinks"][name] | kwargs
    type = kwargs.pop("type", None)
    path = construct_path(kwargs.pop("path"))
    if path.suffix == ".zarr":
        if not "consolidated" in kwargs:
            kwargs["consolidated"] = True
        if not "mode" in kwargs:
            kwargs["mode"] = "a"
        result = ds.drop_encoding().to_zarr(path, **kwargs)
    else:
        if not "engine" in kwargs:
            kwargs["engine"] = "h5netcdf"
        result = ds.to_netcdf(path, **kwargs)

    return result


def prepare_ds(config):
    ds = assign_dims(config["dimensions"])
    if "coordinates" in config:
        ds = assign_coords(config["coordinates"], ds)
    if "chunks" in config:
        ds = chunk(ds, config["chunks"])
    return ds


def unpack(da):
    return da.values[()]
