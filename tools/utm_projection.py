from osgeo import osr
import numpy as np

class Projection:
    def __init__(self, wtk_str):
        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromWkt(wtk_str)
        # Keep axis order stable across GDAL/PROJ versions: (lon, lat) for geographic CRS.
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            self.spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # geo(lat, lon) and projection(x, y) transfer
    def project(self, points, reverse = False):
        shape = points.shape
        reshaped_points = points.reshape(-1, 2)
        if reverse:
            output = self.proj2geo(reshaped_points)
        else:
            output = self.geo2proj(reshaped_points)

        return output.reshape(shape)
    
    def geo2proj(self, geopts):
        geo_sr = self.spatial_reference.CloneGeogCS()
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            geo_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(geo_sr, self.spatial_reference)
        coords = ct.TransformPoints(geopts)
        projpts = np.array(coords)
        return projpts[:, :2]
    
    def proj2geo(self, projpts):
        geo_sr = self.spatial_reference.CloneGeogCS()
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            geo_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(self.spatial_reference, geo_sr)
        coords = ct.TransformPoints(projpts)
        geopts = np.array(coords)
        return geopts[:, :2]