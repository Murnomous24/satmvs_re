from osgeo import osr
import numpy as np

class Projection:
    def __init__(self, wtk_str):
        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromWkt(wtk_str)
    
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
        ct = osr.CoordinateTransformation(geo_sr, self.spatial_reference)
        coords = ct.TransformPoints(geopts)
        projpts = np.array(coords)
        return projpts[:, :2]
    
    def proj2geo(self, projpts):
        geo_sr = self.spatial_reference.CloneGeogCS()
        ct = osr.CoordinateTransformation(self.spatial_reference, geo_sr)
        coords = ct.TransformPoints(projpts)
        geopts = np.array(coords)
        return geopts[:, :2]