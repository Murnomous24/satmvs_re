from torch.utils.data import Dataset
from dataset.data_io import *
from dataset.preprocess import *
from dataset.gen_list import *
import copy

# rpc dataset
class MVSDataset(Dataset):
    def __init__(
            self,
            data_path,
            mode,
            view_num,
            ref_view = 2,
            depth_range = None):
        super(MVSDataset, self).__init__()

        self.data_path = data_path
        self.mode = mode
        assert self.mode in ["train", "test", "valid", "pred"]
        self.view_num = view_num
        self.ref_view = ref_view
        self.depth_range = depth_range
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def get_depth_range(self, default_min, default_max):
        if self.depth_range is not None and len(self.depth_range) >= 2:
            range_min = float(self.depth_range[0])
            range_max = float(self.depth_range[1])
            if range_min < range_max:
                return range_min, range_max

        return float(default_min), float(default_max)
    
    # build sample(image, parameter, depth)
    def build_list(self):
        if self.mode == "pred" or self.ref_view < 0:
            sample_list = gen_list_rpc(self.data_path, self.view_num)
        else:
            sample_list = gen_ref_list_rpc(self.data_path, self.view_num, self.ref_view)
        
        return sample_list
    
    def __len__(self):
        return self.sample_num

    def read_depth(self, file_name):
        height_image = np.float32(load_pfm(file_name))

        return np.array(height_image)
    
    # get single sample
    def get_sample(self, idx):
        if idx < 0 or idx > self.sample_num - 1:
            raise Exception(f'get_sample: out of bound idx, get {idx}')
        
        data = self.sample_list[idx] # get sample(ref and source)
        # data: [ref_img, ref_para, source1_img, source1_para, ..., ref_height]

        centered_images = [] # TODO: why this name
        rpc_para = []
        _, height_min, height_max = load_rpc_as_array(data[2 * self.ref_view + 1]) # read height range from rpc file

        # read ref/source image and rpc parameters
        for v_idx in range(self.view_num):
            # image
            if self.mode == "train":
                image = image_augment(read_img(data[2 * v_idx]))
            else:
                image = read_img(data[2 * v_idx])
            image = np.array(image)

            # rpc parameters
            rpc, _, _ = load_rpc_as_array(data[2 * v_idx + 1])

            rpc_para.append(rpc)
            centered_images.append(center_image(image))
        # change shape
        rpc_para = np.stack(rpc_para)
        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2]) # assume [view_num, H, W, C] -> [view_num, C, H, W](opencv -> pytorch)

        # height range
        height_min, height_max = self.get_depth_range(height_min, height_max)
        height_values = np.array([height_min, height_max], dtype=np.float32)

        # multi stage height map
        height_image = self.read_depth(data[2 * self.view_num]) # height map file path stay last
        h, w = height_image.shape
        height_ms = {
            "stage1": cv2.resize(height_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(height_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": height_image
        }

        # multi stage mask map
        mask = np.float32((height_image >= height_min) * 1.0) * np.float32((height_image <= height_max) * 1.0)
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # multi stage rpc para
        stage1_rpc = rpc_para.copy()
        stage1_rpc[:, 0] = stage1_rpc[:, 0] / 4 # line_off / 4
        stage1_rpc[:, 1] = stage1_rpc[:, 1] / 4 # sample_off / 4 
        stage1_rpc[:, 5] = stage1_rpc[:, 5] / 4 # line_scale / 4
        stage1_rpc[:, 6] = stage1_rpc[:, 6] / 4 # sample_scale / 4
        stage2_rpc = rpc_para.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2 # line_off / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2 # sample_off / 2 
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2 # line_scale / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2 # sample_scale / 2
        rpc_para_ms = {
            "stage1": stage1_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_para
        }

        view_idx = data[0].split("/")[-2] # 0/1/2
        view_name = os.path.splitext(data[0].split("/")[-1])[0] # base0000block0016

        # return pinhole name (e.g. rpc_para_ms -> camera_ms)
        return {
            "images": centered_images,
            "cameras_para": rpc_para_ms,
            "depth": height_ms,
            "mask": mask_ms,
            "depth_values": height_values,
            "view_idx": view_idx,
            "view_name": view_name
        }

    # get sample in prediction mode
    def get_pred_sample(self, idx):
        if idx < 0 or idx > self.sample_num - 1:
            raise Exception(f'get_sample: out of bound idx, get {idx}')
        
        data = self.sample_list[idx] # get sample(ref and source)
        # data: [ref_img, ref_para, source1_img, source1_para, ..., ref_height]

        centered_images = [] # TODO: why this name
        rpc_para = []
        _, height_min, height_max = load_rpc_as_array(data[1]) # read height range from rpc file

        # read ref/source image and rpc parameters
        for v_idx in range(self.view_num):
            # image
            if self.mode == "train":
                image = image_augment(read_img(data[2 * v_idx]))
            else:
                image = read_img(data[2 * v_idx])
            image = np.array(image)

            # rpc parameters
            rpc, _, _ = load_rpc_as_array(data[2 * v_idx + 1])

            rpc_para.append(rpc)
            centered_images.append(center_image(image))
        # change shape
        rpc_para = np.stack(rpc_para)
        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2]) # assume [view_num, H, W, C] -> [view_num, C, H, W](opencv -> pytorch)

        # height range
        height_min, height_max = self.get_depth_range(height_min, height_max)
        height_values = np.array([height_min, height_max], dtype=np.float32)

        # multi stage rpc para
        stage1_rpc = rpc_para.copy()
        stage1_rpc[:, 0] = stage1_rpc[:, 0] / 4 # line_off / 4
        stage1_rpc[:, 1] = stage1_rpc[:, 1] / 4 # sample_off / 4 
        stage1_rpc[:, 5] = stage1_rpc[:, 5] / 4 # line_scale / 4
        stage1_rpc[:, 6] = stage1_rpc[:, 6] / 4 # sample_scale / 4
        stage2_rpc = rpc_para.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2 # line_off / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2 # sample_off / 2 
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2 # line_scale / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2 # sample_scale / 2
        rpc_para_ms = {
            "stage1": stage1_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_para
        }

        view_idx = data[0].split("/")[-2] # 0/1/2
        view_name = os.path.splitext(data[0].split("/")[-1])[0] # base0000block0016

        # return pinhole name (e.g. rpc_para_ms -> camera_ms)
        return {
            "images": centered_images,
            "cameras_para": rpc_para_ms,
            "depth_values": height_values,
            "view_idx": view_idx,
            "view_name": view_name
        }
    
    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            return self.get_sample(index)
        else:
            return self.get_pred_sample(index)