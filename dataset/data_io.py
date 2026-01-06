import numpy as np
import os
import sys
import re
from PIL import Image

def load_pfm(file_name):
    if os.path.exists(file_name) is False:
        raise Exception("load_pfm: pfm file not find")
    file = open(file_name, 'rb')
    
    ident = file.readline().decode('ascii').strip()
    if ident == 'PF':
        color = True
    elif ident == 'Pf':
        color = False
    else:
        raise Exception("load_pfm: Not a pfm file")

    img_dim = file.readline().decode('ascii').strip()
    dim_match = re.match(r'^(\d+)\s+(\d+)\s*$', img_dim)
    if dim_match:
        width, height = map(int, img_dim.split())
    else:
        raise Exception("load_pfm: img_dim is invaild")

    ratio = file.readline().decode('ascii').strip()
    scale = float(ratio)
    if scale < 0:
        data_type = '<f'
    else:
        data_type = '>f'
    
    # endian = "Little" if scale < 0 else "Big"

    # print(f"ident:{ident}")
    # print(f"res:{width} * {height}")
    # print(f"scale:{abs(scale)}")
    # print(f"endian:{endian}")

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0) # pfm file start the pixel BOTTOM LEFT, OpenCV start TOP LEFT, so we need reverse the row order

    return data

def save_pfm(file_name, image, scale=1):
    file = open(file_name, "wb")

    if image.dtype != np.float32:
        raise Exception(f"save_pfm: image dtype must be 'float32', but receive {image.dtype}")

    if image.ndim == 3 and image.shape[2] == 3:
        color = True
        ident = "PF"
    elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        color = False
        ident = "Pf"
    else:
        raise Exception(f"save_pfm: Image's shape is invaild, receive {image.shape}")

    endian = image.dtype.byteorder
    if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
        scale = -scale

    header = f"{ident}\n{image.shape[1]} {image.shape[0]}\n{scale}\n"
    file.write(header.encode('ascii'))

    data = np.flip(image, 0)
    endian = '<f' if scale < 0 else '>f'
    data.tofile(file)

def load_rpc_as_array(file_name):
    if os.path.exists(file_name) is False:
        raise Exception("load_rpc_as_array: pfm file not find")

    file = open(file_name, 'r')

    full_text = file.read().splitlines()
    data = [line.split(' ')[1] for line in full_text]
    # print(data)

    data = np.array(data, dtype = np.float64)
    
    h_min = data[4] - data[9]
    h_max = data[4] + data[9]

    return data, h_min, h_max

def read_img(file_name):
    if os.path.exists(file_name) is False:
        raise Exception("read_img: img file not find")
    
    img = Image.open(file_name)
    imgs = img.split()

    if len(imgs) == 3:
        res = img
    elif len(imgs) == 1:
        res = Image.merge("RGB", (imgs[0], imgs[0], imgs[0]))
    else:
        raise Exception(f"read_img: image's channel must be 3(rgb) or 1(gray), but receive {len(imgs)}")
    
    return res

def read_camera(file_name):
    if os.path.exists(file_name) is False:
        raise Exception("read_camera: camera file not find")
    
    file = open(file_name, "r")
    all_text = file.read().splitlines()

    E = np.array(
        [[float(e) for e in all_text[0].split()],
         [float(e) for e in all_text[1].split()],
         [float(e) for e in all_text[2].split()],
         [float(e) for e in all_text[3].split()]]
    )

    K_text = [float(k) for k in all_text[5].split()]
    K = np.array(
        [[K_text[0], 0, K_text[1]],
         [0, K_text[0], K_text[2]],
         [0, 0, 1]]
    )
    
    d_min, d_max, d_inter = [float(d) for d in all_text[7].split()]

    # print(f"E:{E}")
    # print(f"K:{K}")
    # print(f"d_min, d_max, d_inter:{d_min}, {d_max}, {d_inter}")

    return K, E, d_min, d_max, d_inter

def load_pin_as_nn(file_name):
    K, E, d_min, d_max, d_inter = read_camera(file_name)

    cam = np.zeros((2, 4, 4), dtype=np.float64)

    # intrinsic
    cam[0] = E.astype(np.float64)

    # extrinsic
    cam[1, 0:3, 0:3] = K.astype(np.float64)
    cam[1][3][0] = np.float64(d_min)
    cam[1][3][1] = np.float64(d_inter)
    cam[1][3][3] = np.float64(d_max)

    return cam

# func test
# load_pfm("./test_file/test.pfm")

# depth_map = np.random.rand(480, 640).astype(np.float32)
# save_pfm("./test_file/save_pfm.pfm", depth_map)
# load_pfm("./test_file/save_pfm.pfm")

# load_rpc_as_array("./test_file/test.rpc")

# read_camera("./test_file/camera.txt")


