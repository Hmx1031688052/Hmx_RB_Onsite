"""Runtime compatibility for legacy gloplan OpenDRIVE spiral parsing.

The platform maps use OpenDRIVE clothoids extensively, while the legacy
``ros2_map/src/gloplan`` parser handles only line, arc, and poly3 geometry.
This module patches that external dependency at startup so the fix travels
with Hmx_RB_Onsite even when only this repository is pulled on the vehicle.
"""

import math


def install_spiral_support():
    """Teach an old external gloplan package to sample ``<spiral>`` nodes."""
    from gloplan.elements import geometry as geometry_module
    from gloplan.elements import road as road_module

    road_class = road_module.Road
    if getattr(road_class, "_hmx_spiral_compat_installed", False):
        return False

    # A newer ros2_map may already contain the native implementation.  In
    # that case leave it untouched.
    if (
        hasattr(geometry_module, "GeometrySpiral")
        and hasattr(road_module.PlanView, "setSpiral")
    ):
        road_class._hmx_spiral_compat_installed = True
        return False

    class GeometrySpiral(geometry_module.Geometry):
        def __init__(self, node_geometry):
            super().__init__(node_geometry)
            self.geo_type = getattr(
                geometry_module.GeometryType,
                "Spiral",
                geometry_module.GeometryType.unknown,
            )
            node_spiral = node_geometry.find("spiral")
            self.curv_start = (
                geometry_module.get_float(
                    node_spiral, "curvStart"
                )
                if node_spiral is not None
                else 0.0
            )
            self.curv_end = (
                geometry_module.get_float(
                    node_spiral, "curvEnd"
                )
                if node_spiral is not None
                else self.curv_start
            )
            self.calPosition(self.step)

        def calPosition(self, step):
            length = max(0.0, self.getLength())
            if length <= 0.0:
                return
            curvature_rate = (
                self.curv_end - self.curv_start
            ) / length
            x = self.getstartPosition().getX()
            y = self.getstartPosition().getY()
            distance = 0.0
            points = []
            while distance < length:
                theta = (
                    self.getHeading()
                    + self.curv_start * distance
                    + 0.5
                    * curvature_rate
                    * distance
                    * distance
                )
                points.append(
                    geometry_module.Point(
                        x,
                        y,
                        theta,
                        self.getS() + distance,
                    )
                )
                delta = min(float(step), length - distance)
                midpoint = distance + 0.5 * delta
                midpoint_theta = (
                    self.getHeading()
                    + self.curv_start * midpoint
                    + 0.5
                    * curvature_rate
                    * midpoint
                    * midpoint
                )
                x += math.cos(midpoint_theta) * delta
                y += math.sin(midpoint_theta) * delta
                distance += delta
            self.setPoints(points)

    original_init = road_class.__init__

    def patched_road_init(self, node_road):
        original_init(self, node_road)
        node_planview = node_road.find("planView")
        if (
            node_planview is None
            or node_planview.find("geometry/spiral") is None
        ):
            return

        # Rebuild in source order. Appending spirals after the old parser has
        # run would reorder road geometry and create another discontinuity.
        geometries = []
        for node_geometry in node_planview.findall("geometry"):
            if node_geometry.find("line") is not None:
                geometries.append(
                    road_module.GeometryLine(node_geometry)
                )
            elif node_geometry.find("arc") is not None:
                geometries.append(
                    road_module.GeometryArc(node_geometry)
                )
            elif node_geometry.find("spiral") is not None:
                geometries.append(GeometrySpiral(node_geometry))
            elif node_geometry.find("poly3") is not None:
                geometries.append(
                    road_module.GeometryPoly3(node_geometry)
                )
        self.planView.geometries = geometries

    geometry_module.GeometrySpiral = GeometrySpiral
    road_module.GeometrySpiral = GeometrySpiral
    road_class.__init__ = patched_road_init
    road_class._hmx_spiral_compat_installed = True
    return True
