from shapely.geometry import Point
import xarray as xr


def xr_cell_polygon_overlap_fraction(xrdata, polygons, cell_radius, cap_style="square"):
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
    cell = Point(x, y).buffer(xy_buffer, cap_style=cap_style)
    return cell.intersection(poly).area / cell.area


def xr_point_polygon_distance(xrdata, polygons):
    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).distance(poly),
        xrdata["x"],
        xrdata["y"],
        polygons,
        vectorize=True,
    )


def xr_point_inside_polygon(xrdata, polygons):
    return xr.apply_ufunc(
        lambda x, y, poly: Point(x, y).covered_by(poly),
        xrdata["x"],
        xrdata["y"],
        polygons,
        vectorize=True,
    )
