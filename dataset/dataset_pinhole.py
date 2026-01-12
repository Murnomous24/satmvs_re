from torch.utils.data import Dataset
from data_io import *
from preprocess import *
from gen_list import *

# pinhole dataset Class
class MVSDataset(Dataset):
    def __init__(
            self,
            data_path,
            mode,
            view_num,
            ref_view = 2):
        super(MVSDataset, self).__init__()

        self.data_path = data_path
        self.mode = mode
        assert self.mode in ["train", "test", "valid", "pred"]
        self.view_num = view_num
        self.ref_view = ref_view
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    # build sample(image, parameter, depth)
    def build_list(self):
        if self.mode == "pred" or self.ref_view < 0:
            sample_list = gen_list_cam(self.data_path, self.view_num)
        else:
            sample_list = gen_ref_list_cam(self.data_path, self.view_num, self.ref_view)
        
        return sample_list

    def __len__(self):
        return self.sample_num

    def read_depth(self, file_name):
        depth_image = np.float32(load_pfm(file_name))

        return np.array(depth_image)
    
    # get single sample
    def get_sample(self, idx):
        if idx < 0 or idx > self.sample_num - 1:
            raise Exception(f'get_sample: out of bound idx, get {idx}')
        
        data = self.sample_list[idx] # get sample(ref and source)
        # data: [ref_img, ref_para, source1_img, source1_para, ..., ref_height]

        centered_images = [] # TODO: why this name
        proj_matrices = []
        depth_min = None
        depth_max = None

        # read ref/source image and camera parameters
        for v_idx in range(self.view_num):
            # image
            if self.mode == "train":
                image = image_augment(read_img(data[2 * v_idx]))
            else:
                image = read_img(data[2 * v_idx])
            image = np.array(image)

            # camera parameters
            camera = load_pin_as_nn(data[2 * v_idx + 1])

            if v_idx == 0: # ref
                depth_min = camera[1][3][0]
                depth_max = camera[1][3][3]

            extrinsics = camera[0, :, :]
            intrinsics = camera[1, 0:3, 0:3]
            proj_matrice = extrinsics.copy() # [R|t]
            proj_matrice[:3, :4] =  np.matmul(intrinsics, proj_matrice[:3, :4]) # K[R|t]

            proj_matrices.append(proj_matrice)
            centered_images.append(center_image(image))
        # change shape
        proj_matrices = np.stack(proj_matrices)
        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2]) # assume [view_num, H, W, C] -> [view_num, C, H, W](opencv -> pytorch)

        # depth range
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        # multi stage depth map
        depth_image = self.read_depth(data[2 * self.view_num]) # depth map file path stay last
        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }

        # multi stage mask map
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # multi stage project matrice
        stage1_project_matrices = proj_matrices.copy()
        stage1_project_matrices[:, :2, :] = proj_matrices[:, :2, :] / 4
        stage2_project_matrices = proj_matrices.copy()
        stage2_project_matrices[:, :2, :] = proj_matrices[:, :2, :] / 2
        proj_matrices_ms = {
            "stage1": stage1_project_matrices,
            "stage2": stage2_project_matrices,
            "stage3": proj_matrices
        }

        view_idx = data[0].split("/")[-2] # 0/1/2
        view_name = os.path.splitext(data[0].split("/")[-1])[0] # base0000block0016

        return {
            "images": centered_images,
            "cameras_para": proj_matrices_ms,
            "depth": depth_ms,
            "mask": mask_ms,
            "depth_values": depth_values,
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
        proj_matrices = []
        depth_min = None
        depth_max = None

        # read ref/source image and camera parameters
        for v_idx in range(self.view_num):
            # image
            if self.mode == "train":
                image = image_augment(read_img(data[2 * v_idx]))
            else:
                image = read_img(data[2 * v_idx])
            image = np.array(image)

            # camera parameters
            camera = load_pin_as_nn(data[2 * v_idx + 1])

            if v_idx == 0: # ref
                depth_min = camera[1][3][0]
                depth_max = camera[1][3][3]

            extrinsics = camera[0, :, :]
            intrinsics = camera[1, 0:3, 0:3]
            proj_matrice = extrinsics.copy() # [R|t]
            proj_matrice[:3, :4] =  np.matmul(intrinsics, proj_matrice[:3, :4]) # K[R|t]

            proj_matrices.append(proj_matrice)
            centered_images.append(image)
        # change shape
        proj_matrices = np.stack(proj_matrices)
        centered_images = np.stack(center_image(centered_images)).transpose([0, 3, 1, 2]) # assume [view_num, H, W, C] -> [view_num, C, H, W](opencv -> pytorch)

        # depth range
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        # multi stage project matrice
        stage1_project_matrices = proj_matrices.copy()
        stage1_project_matrices[:, :2, :] = proj_matrices[:, :2, :] / 4
        stage2_project_matrices = proj_matrices.copy()
        stage2_project_matrices[:, :2, :] = proj_matrices[:, :2, :] / 2
        proj_matrices_ms = {
            "stage1": stage1_project_matrices,
            "stage2": stage2_project_matrices,
            "stage3": proj_matrices
        }

        view_idx = data[0].split("/")[-2] # 0/1/2
        view_name = os.path.splitext(data[0].split("/")[-1])[0] # base0000block0016

        return {
            "images": centered_images,
            "cameras_para": proj_matrices_ms,
            "depth_values": depth_values,
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
        

# test code
# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader

#     DATA_PATH = "/home/murph_dl/Paper_Re/SatMVS_Re/test_file/test_dataset_pinhole"
#     MODE = "train"
#     VIEW_NUM = 3
#     REF_VIEW = 0

#     try:
#         dataset = MVSDataset(
#             data_path = DATA_PATH,
#             mode = MODE,
#             view_num = VIEW_NUM,
#             ref_view = REF_VIEW
#         )
#         print(f"1: load dataset ok, length: {len(dataset)}")

#         data_loader = DataLoader(dataset, batch_size = 1, shuffle = False)

#         print("2: load data from dataloader")
#         for idx, sample in enumerate(data_loader):
#             print(f"2.1: sample index: {idx}")

#             print(f"2.2: image shape [B, V, C, H, W]: {sample['images'].shape}")

#             print(f"2.3: depth range (min, max): {sample['depth_values'][0].numpy()}")

#             print(f"2.4: camera intri & project matri")
#             for stage in sample['cameras_para']:
#                 print(f"  - cameras_para {stage}: {sample['cameras_para'][stage].shape}")
                
#             if dataset.mode != "pred":
#                 for stage in sample['depth']:
#                     print(f"  - depth {stage}: {sample['depth'][stage].shape}")

                
#                 for stage in sample['mask']:
#                         print(f"  - mask {stage}: {sample['mask'][stage].shape}")
            
#             print(f"2.5: ref index: {sample['view_idx']}")
#             print(f"2.6: file name: {sample['view_name']}")

#             break

#     except Exception as e:
#         print(f"test failed: {e}")
