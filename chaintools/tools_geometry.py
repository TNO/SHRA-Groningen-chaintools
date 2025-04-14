"""
Contains spatial/geographic functions for xarray-based data arrays.   
"""

from shapely.geometry import Point
from shapely import points, multipoints
import xarray as xr
import geopandas as gpd
import numpy as np

def xr_cell_polygon(xrdata, cell_radius=None, cap_style="square", spatial_coordinates=None):
    """
    Create a cell around each point in xrdata, then buffer to create a polygon or multipolygon.

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Array with data, must contain spatial coordinates 'x' and 'y'.
    cell_radius : float
        Radius of cell. When default square cap_style is used, this is half the side length of the square.
    cap_style : str, optional
        Cap style for cell. Default is 'square'.
    spatial_coordinates : list, optional
        Names of spatial coordinates in xrdata. Default is ['x', 'y'].

    Returns
        Polygon or MultiPolygon
    -------

    """
    if spatial_coordinates is None:
        spatial_coordinates = ["x", "y"]
    x, y = spatial_coordinates
    x_xr, y_xr = xr.broadcast(xrdata[x], xrdata[y])

    polygon = multipoints(points(x_xr, y_xr)).buffer(cell_radius, cap_style=cap_style)
    return polygon


def xr_cell_polygon_overlap_fraction(
    xrdata, polygons, cell_radius=None, cap_style="square", spatial_coordinates=None
):
    """
    Calculates the fraction of overlap between a polygon and a cell. Overlap = 1 means cell is entirely within polygon.
    This function is an xarray u_func wrapper around _overlap_fraction function.

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Array with data, must contain spatial coordinates 'x' and 'y'.
    polygons : list of shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polygon(s) to calculate overlap fraction with.
    cell_radius : float
        Radius of cell. When default square cap_style is used, this is half the side length of the square.
    cap_style : str, optional
        Cap style for cell. Default is 'square'.
    spatial_coordinates : list, optional
        Names of spatial coordinates in xrdata. Default is ['x', 'y'].

    Returns
    -------

    """
    if spatial_coordinates is None:
        spatial_coordinates = ["x", "y"]
    x, y = spatial_coordinates

    if cell_radius is None:
        # retrieve unique, sorted values of x and y to get minimum distance between points
        dx = np.min(np.diff(np.unique(xrdata[x].data)))
        dy = np.min(np.diff(np.unique(xrdata[y].data)))
        d = np.min([dx, dy])
        cell_radius = d / 2.0

    return xr.apply_ufunc(
        _overlap_fraction,
        xrdata[x],
        xrdata[y],
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
    poly: shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polygon to calculate overlap fraction with.
    xy_buffer: float
        Radius of cell. When "square" cap_style is used, this is half the side length of the square.
    cap_style: str
        Cap style for cell buffer.
    
    Returns
    -------
    float or array-like
        Fraction of overlap between polygon and cell

    """

    cell = Point(x, y).buffer(xy_buffer, cap_style=cap_style)
    return cell.intersection(poly).area / cell.area


def xr_point_polygon_distance(xrdata, polygons, spatial_coordinates=None):
    """
    Calculate distance between point and polygon

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Data, must contain spatial coordinates 'x' and 'y'.
    polygons:  list of shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polygon(s) to calculate distance to.
    spatial_coordinates : list, optional
        Names of spatial coordinates in xrdata. Default is ['x', 'y'].

    Returns
    -------

    """
    if spatial_coordinates is None:
        spatial_coordinates = ["x", "y"]
    x, y = spatial_coordinates

    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).distance(poly),
        xrdata[x],
        xrdata[y],
        polygons,
        vectorize=True,
    )


def xr_point_inside_polygon(xrdata, polygons, spatial_coordinates=None):
    """
    Check if point falls inside polygon

    Parameters
    ----------
    xrdata : xr.DataArray or xr.Dataset
        Data, must contain spatial coordinates 'x' and 'y'.
    polygons: list of shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polygon(s) to check if point is inside.
    spatial_coordinates : list, optional
        Names of spatial coordinates in xrdata. Default is ['x', 'y'].

    Returns
    -------

    """
    if spatial_coordinates is None:
        spatial_coordinates = ["x", "y"]
    x, y = spatial_coordinates

    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).covered_by(poly),
        xrdata[x],
        xrdata[y],
        polygons,
        vectorize=True,
    )


def get_zone_assignment(input_gdf, zone_gdf, zone_id=None, zone_distance_id=None):
    """Assigns the nearest zone in the zone_gdf to each item in the input_gdf.

    Parameters
    ----------
    input_gdf : geopandas.GeoDataFrame
        The input geodataframe.
    zone_gdf : geopandas.GeoDataFrame
        The zone geodataframe.
    zone_id : str, optional
        The name of the column in the zone_gdf that contains the zone id. If None specified it will be called 'index_right'.
    zone_distance_id : str, optional
        The name of the column in the output geodataframe that contains the distance to the nearest zone.
    Returns
    -------
    geopandas.GeoDataFrame
        A geodataframe with the same geometry as input_gdf and an additional column that contains the zone id,
        and, if requested, a column with the distance.

    """

    zone_assignment = gpd.sjoin_nearest(
        input_gdf[["geometry"]],
        zone_gdf[["geometry"]],
        how="left",
        distance_col=zone_distance_id,
    )
    zone_index_name = zone_gdf.index.name
    if zone_index_name is None:
        zone_index_name = "index_right"

    if zone_id is not None:
        zone_assignment = zone_assignment.rename(columns={zone_index_name: zone_id})

    return zone_assignment


def apply_zone_assignment(input_gdf, zone_gdf, zone_id=None, zone_distance_id=None):
    """Assigns the nearest zone in the zone_gdf to each item in the input_gdf and includes it in the input.

    Parameters
    ----------
    input_gdf : geopandas.GeoDataFrame
        The input geodataframe.
    zone_gdf : geopandas.GeoDataFrame
        The zone geodataframe.
    zone_id : str, optional
        The name of the column in the zone_gdf that contains the zone id. If None specified it will be called 'index_right'.
    zone_distance_id : str, optional
        The name of the column in the output geodataframe that contains the distance to the nearest zone.

    Returns
    -------
    geopandas.GeoDataFrame
       The input_gdf with an additional column that contains the zone id,
        and, if requested, a column with the distance.
    """

    zone_assignment = get_zone_assignment(
        input_gdf,
        zone_gdf,
        zone_id=zone_id,
        zone_distance_id=zone_distance_id,
    ).drop(columns=["geometry"])

    return input_gdf.join(zone_assignment)


def define_grid_spanning_zonation(grid_spacing, grid_anchor, zonation_gdf):
    """Defines a grid that spans the entire zone geodataframe.

    Parameters
    ----------
    grid_spacing : float
        The spacing of the grid.
    grid_anchor : list
        The anchor point of the grid. This is a hypothetical point that is the origin of the grid.
        All other points are calculated relative to this point. The anchor point is not necessarily located
        inside the span.
    zonation_gdf : geopandas.GeoDataFrame
        The zone geodataframe. Its CRS is also used for the grid.

    Returns
    -------
    xarray.Dataset
        An xarray dataset with the grid. The CRS is adopted from the zonation geodataframe.
    """

    anchor = np.asarray(grid_anchor)
    spacing = np.asarray(grid_spacing)

    minx, miny, maxx, maxy = zonation_gdf.total_bounds
    min = np.array([minx, miny])
    max = np.array([maxx, maxy])
    min = np.floor((min - anchor) / spacing) * spacing + anchor
    max = np.ceil((max - anchor) / spacing) * spacing + anchor

    x_range = np.arange(min[0], max[0] + spacing, spacing)
    y_range = np.arange(min[1], max[1] + spacing, spacing)
    surface_grid = xr.Dataset({"x": x_range, "y": y_range})
    surface_grid = surface_grid.rio.write_crs(zonation_gdf.crs.to_epsg())

    return surface_grid


def get_grid_support_polygon(support_grid, grid_spacing=None, dims=None):
    """Calculates the polygon that envelopes the nonzero cells of the grid. Assumes the grid is regular.

    Parameters
    ----------
    support_grid : xarray.Dataset
        The grid that is supported.
    grid_spacing : float, optional
        The spacing of the grid. If None, the average spacing is calculated, which should be
        the precise spacing if the grid is regular.
    dims : list, optional
        The dimensions that are used to calculate the support polygon. Default is ['x', 'y'].

    Returns
    -------
    shapely.geometry.Polygon
        The polygon that supports the grid.
    """
    if dims is None:
        dims = ["x", "y"]
    if grid_spacing is None:
        grid_spacing = np.mean([support_grid[dim].diff(dim).mean() for dim in dims])
    grid_stacked = support_grid.stack({"loc": dims}).fillna(0) > 0
    reduced_contrib_stacked = grid_stacked.where(grid_stacked).dropna("loc")
    support_polygon = xr_cell_polygon(reduced_contrib_stacked, grid_spacing / 2.0)

    return support_polygon
