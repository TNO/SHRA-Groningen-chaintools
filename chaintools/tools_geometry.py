from shapely.geometry import Point
import xarray as xr

"""
Contains spatial/geographic functions for xarray-based data arrays.   
"""

def xr_cell_polygon_overlap_fraction(xrdata, polygons, cell_radius, cap_style="square"):
    """
    Calculates the fraction of overlap between a polygon and a cell. Overlap = 1 means cell is entirely within polygon.
    This function is an xarray u_func wrapper around _overlap_fraction function.

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Array with data, must contain spatial dimensions 'x' and 'y'.
    polygons :

    cell_radius : float
        Radius of cell.
    cap_style : str, optional


    Returns
    -------

    """
    return xr.apply_ufunc(
        _overlap_fraction,
        xrdata["x"],
        xrdata["y"],
        polygons,
        cell_radius,
        kwargs={"cap_style": cap_style},
        vectorize=True,
    )


def _overlap_fraction(x, y, poly, xy_buffer, cap_style):
    """
    Calculates the fraction of overlap between a polygon and a cell. Overlap = 1 means cell is entirely within polygon.

    Parameters
    ----------
    x : float or array-like
        X-coordinate(s) of point(s). Float or 1D array
    y : float or array-like
        Y-coordinate(s) of point(s). Float or 1D array
    poly

    xy_buffer

    cap_style

    Returns
    -------

    """

    cell = Point(x, y).buffer(xy_buffer, cap_style=cap_style)
    return cell.intersection(poly).area / cell.area


def xr_point_polygon_distance(xrdata, polygons):
    """
    Calculate distance between point and polygon

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Data, must contain spatial dimensions 'x' and 'y'.
    polygons

    Returns
    -------

    """
    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).distance(poly),
        xrdata["x"],
        xrdata["y"],
        polygons,
        vectorize=True,
    )


def xr_point_inside_polygon(xrdata, polygons):
    """
    Check if point falls inside polygon

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Data, must contain spatial dimensions 'x' and 'y'.
    polygons

    Returns
    -------

    """

    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).covered_by(poly),
        xrdata["x"],
        xrdata["y"],
        polygons,
        vectorize=True,
    )
