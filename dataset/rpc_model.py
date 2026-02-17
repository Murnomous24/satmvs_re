import numpy as np
import os

# rpc model for dsm prediction(no inverse rpc provide)
class RPCModel:
    def __init__(self, data = np.zeros(170, dtype = np.float64)):
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LON_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LON_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]
    
    # read rpc content from file
    def load_rpc_from_file(self, file_path):
        if os.path.exists(file_path) is False:
            raise Exception("load_rpc_from_file: pfm file not find")
    
        with open(file_path, 'r') as f:
            text = f.read().splitlines()

        data = [line.split(' ')[1] for line in text]
        data = np.array(data, dtype = np.float64)

        # directly assign
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LON_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LON_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        # inverse member need to be calculated, cause these dataset only contains forward arugments
        self.calculate_inverse_member()
    
    def calculate_inverse_member(self):
        sample_grid = self.build_sample_grid()
        self.solve_inverse_member(sample_grid)

    def build_sample_grid(self, xy_sample_num = 30, z_sample_num = 20):
        # get grid's bounding box
        lat_max = self.LAT_OFF + self.LAT_SCALE # lat_off is center, lat_scale is length
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LON_OFF + self.LON_SCALE
        lon_min = self.LON_OFF - self.LON_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        # build mesh grid
        lat = np.linspace(lat_min, lat_max, xy_sample_num)
        lon = np.linspace(lon_min, lon_max, xy_sample_num)
        hei = np.linspace(hei_min, hei_max, z_sample_num)
        
        lat, lon, hei = np.meshgrid(lat, lon, hei)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)
        
        # forward sample
        samp, line = self.obj2photo(lat, lon, hei)
        grid = np.stack((samp, line, lat, lon, hei), axis = -1)
        
        selected_grid = []
        for sample in grid:
            flag = (sample[0] < samp_min) & (sample[0] > samp_max) & (sample[1] < line_min) & (sample[1] > line_max)
            if flag:
                continue
            else:
                selected_grid.append(sample)
        
        grid = np.array(selected_grid)
        return grid
    
    def solve_inverse_member(self, sample_grid):
        # get properties from sample grid
        samp, line, lat, lon, hei = np.hsplit(sample_grid.copy(), 5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        # normalize
        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE
        line -= self.LINE_OFF
        line /= self.LINE_SCALE
        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE
        lon -= self.LON_OFF
        lon /= self.LON_SCALE
        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        # calculate coef
        coef = self.coef_calculate(samp, line, hei)
        sample_num = coef.shape[0]

        # solving inverse member
        
        # build A
        A = np.zeros((sample_num * 2, 78))
        # Lat part
        A[0: sample_num, 0:20] = - coef
        A[0: sample_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        # Lon part
        A[sample_num:, 39:59] = - coef
        A[sample_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        # build l
        l = np.concatenate((lat, lon), -1)
        l = -l

        # solve Ax = l
        x, res, rank, sv = np.linalg.lstsq(A, l, rcond = None) # TODO: solving setting

        # inverse member
        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]
    
    def coef_calculate(self, P, L, H):
        sample_num = P.shape[0]
        coef = np.zeros((sample_num, 20))

        coef[:, 0] = 1.0
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        
        coef[:, 10] = P * coef[:, 5]
        coef[:, 11] = L * coef[:, 7]
        coef[:, 12] = L * coef[:, 8]
        coef[:, 13] = L * coef[:, 9]
        coef[:, 14] = L * coef[:, 4]
        coef[:, 15] = P * coef[:, 8]
        coef[:, 16] = P * coef[:, 9]
        coef[:, 17] = L * coef[:, 5]
        coef[:, 18] = P * coef[:, 6]
        coef[:, 19] = H * coef[:, 9]

        return coef
    
    # TODO: check for inverse parameter acc
    def check(
        self,
        width,
        height,
        xy_sample_num,
        z_sample_num
    ):
        height_min, height_max = self.get_height_min_max()

        # build grid
        x = np.linspace(0, width, xy_sample_num)
        y = np.linspace(0, height, xy_sample_num)
        h = np.linspace(height_min, height_max, z_sample_num)
        x, y, h = np.meshgrid(x, y, h)
        x = x.reshape(-1)
        y = y.reshape(-1)
        h = h.reshape(-1)

        # forward & backward project
        lat, lon = self.photo2obj(x, y, h)
        reproj_x, reproj_y = self.obj2photo(lat, lon, h)
        error_x = (reproj_x - x) * (reproj_x - x)
        error_y = (reproj_y - y) * (reproj_y - y)
        
        reproj_error = np.sqrt(error_x + error_y)
        reproj_error_x = np.sqrt(error_x)
        reproj_error_y = np.sqrt(error_y)
        
    def get_height_min_max(self):
        height_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        height_min = self.HEIGHT_OFF - self.HEIGHT_SCALE

        return height_min, height_max
    
    def obj2photo(self, inlat, inlon, inhei):
        lat = np.copy(inlat)
        lon = np.copy(inlon)
        hei = np.copy(inhei)

        # normalization (lat - lat_off) / lat_scale
        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE

        # normalization (lon - lon_off) / lon_scale
        lon -= self.LON_OFF
        lon /= self.LON_SCALE

        # normalization (hei - hei_off) / hei_scale
        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE
        
        # projection
        coef = self.coef_calculate(lat, lon, hei)
        denom_samp = np.sum(coef * self.SDEM, axis = -1)
        denom_line = np.sum(coef * self.LDEM, axis = -1)
        denom_samp = np.where(np.abs(denom_samp) < 1e-8, 1e-8, denom_samp)
        denom_line = np.where(np.abs(denom_line) < 1e-8, 1e-8, denom_line)
        samp = np.sum(coef * self.SNUM, axis = -1) / denom_samp
        line = np.sum(coef * self.LNUM, axis = -1) / denom_line

        # from noramlization to pixel
        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        return samp, line
    
    def photo2obj(self, insamp, inline, inhei):
        samp = np.copy(insamp)
        line = np.copy(inline)
        hei = np.copy(inhei)

        # normalization (samp - samp_off) / samp_scale
        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE

        # normalization (lon - lon_off) / lon_scale
        line -= self.LINE_OFF
        line /= self.LINE_SCALE

        # normalization (hei - hei_off) / hei_scale
        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE
        
        # projection
        coef = self.coef_calculate(samp, line, hei)
        denom_lat = np.sum(coef * self.LATDEM, axis = -1)
        denom_lon = np.sum(coef * self.LONDEM, axis = -1)
        denom_lat = np.where(np.abs(denom_lat) < 1e-8, 1e-8, denom_lat)
        denom_lon = np.where(np.abs(denom_lon) < 1e-8, 1e-8, denom_lon)
        lat = np.sum(coef * self.LATNUM, axis = -1) / denom_lat
        lon = np.sum(coef * self.LONNUM, axis = -1) / denom_lon

        # from noramlization to pixel
        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LON_SCALE
        lon += self.LON_OFF

        return lat, lon