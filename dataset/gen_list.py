import os

def gen_ref_list_cam(data_path, view_num, ref_view=2):
    sample_list = []

    ref_image_path = os.path.join(data_path, (f'image/{ref_view}'))
    ref_camera_path = os.path.join(data_path, (f'camera/{ref_view}'))
    ref_depth_path = os.path.join(data_path, (f'depth/{ref_view}'))

    image_files = os.listdir(ref_image_path)
    for file in image_files:
        sample = []

        name = os.path.splitext(file)[0] # base0000block0016.png -> base0000block0016
        ref_image = os.path.join(ref_image_path, f'{name}.png') # TODO: suffix jpg / png ?
        ref_camera = os.path.join(ref_camera_path, f'{name}.txt') # TODO: suffix ?
        ref_depth = os.path.join(ref_depth_path, f'{name}.pfm')

        # add ref content
        sample.append(ref_image)
        sample.append(ref_camera)

        # add source content
        for idx in range(view_num):
            source_idx = (ref_view + idx) % view_num # TODO: can we use idx != ref_view ?

            if source_idx != ref_view: # not the ref index
                source_image = os.path.join(data_path, f'image/{source_idx}/{name}.png')
                source_camera = os.path.join(data_path, f'camera/{source_idx}/{name}.txt')
                
                sample.append(source_image)
                sample.append(source_camera)
        
        # add ref depth
        sample.append(ref_depth)

        # add one block(1 ref, 2 view) to sample list
        sample_list.append(sample)
    
    return sample_list

def gen_list_cam(data_path, view_num):
    sample_list = []

    for ref_idx in range(view_num):
        ref_image_path = os.path.join(data_path, (f'image/{ref_idx}'))
        ref_camera_path = os.path.join(data_path, (f'camera/{ref_idx}'))
        ref_depth_path = os.path.join(data_path, (f'depth/{ref_idx}'))

        image_files = os.listdir(ref_image_path)
        for file in image_files:
            sample = []
            
            name = os.path.splitext(file)[0] # base0000block0016.png -> base0000block0016
            ref_image = os.path.join(ref_image_path, f'{name}.png') # TODO: suffix jpg / png ?
            ref_camera = os.path.join(ref_camera_path, f'{name}.txt') # TODO: suffix ?
            ref_depth = os.path.join(ref_depth_path, f'{name}.pfm')

            # add ref content
            sample.append(ref_image)
            sample.append(ref_camera)

            # add source content
            for idx in range(view_num):
                source_idx = (idx + ref_idx) % view_num

                if source_idx != ref_idx: # not the ref index
                    source_image = os.path.join(data_path, f'image/{source_idx}/{name}.png')
                    source_camera = os.path.join(data_path, f'camera/{source_idx}/{name}.txt')
                    
                    sample.append(source_image)
                    sample.append(source_camera)
                
            # add ref depth
            sample.append(ref_depth)

            # add one block(1 ref, 2 view) to sample list
            sample_list.append(sample) 

    return sample_list 

def gen_ref_list_rpc(data_path, view_num, ref_view=2):
    sample_list = []

    ref_image_path = os.path.join(data_path, (f'image/{ref_view}'))
    ref_rpc_path = os.path.join(data_path, (f'rpc/{ref_view}'))
    ref_height_path = os.path.join(data_path, (f'height/{ref_view}'))

    image_files = os.listdir(ref_image_path)
    for file in image_files:
        sample = []

        name = os.path.splitext(file)[0] # base0000block0016.png -> base0000block0016
        ref_image = os.path.join(ref_image_path, f'{name}.png') # TODO: suffix jpg / png ?
        ref_rpc = os.path.join(ref_rpc_path, f'{name}.rpc') # TODO: suffix ?
        ref_height = os.path.join(ref_height_path, f'{name}.pfm')

        # add ref content
        sample.append(ref_image)
        sample.append(ref_rpc)

        # add source content
        for idx in range(view_num):
            source_idx = (ref_view + idx) % view_num # TODO: can we use idx != ref_view ?

            if source_idx != ref_view: # not the ref index
                source_image = os.path.join(data_path, f'image/{source_idx}/{name}.png')
                source_rpc = os.path.join(data_path, f'rpc/{source_idx}/{name}.rpc')
                
                sample.append(source_image)
                sample.append(source_rpc)
        
        # add ref depth
        sample.append(ref_height)

        # add one block(1 ref, 2 view) to sample list
        sample_list.append(sample)
    
    return sample_list    

def gen_list_rpc(data_path, view_num):
    sample_list = []

    for ref_idx in range(view_num):
        ref_image_path = os.path.join(data_path, (f'image/{ref_idx}'))
        ref_rpc_path = os.path.join(data_path, (f'rpc/{ref_idx}'))
        ref_height_path = os.path.join(data_path, (f'height/{ref_idx}'))

        image_files = os.listdir(ref_image_path)
        for file in image_files:
            sample = []
            
            name = os.path.splitext(file)[0] # base0000block0016.png -> base0000block0016
            ref_image = os.path.join(ref_image_path, f'{name}.png') # TODO: suffix jpg / png ?
            ref_rpc = os.path.join(ref_rpc_path, f'{name}.rpc') # TODO: suffix ?
            ref_height = os.path.join(ref_height_path, f'{name}.pfm')

            # add ref content
            sample.append(ref_image)
            sample.append(ref_rpc)

            # add source content
            for idx in range(view_num):
                source_idx = (idx + ref_idx) % view_num

                if source_idx != ref_idx: # not the ref index
                    source_image = os.path.join(data_path, f'image/{source_idx}/{name}.png')
                    source_rpc = os.path.join(data_path, f'rpc/{source_idx}/{name}.rpc')
                    
                    sample.append(source_image)
                    sample.append(source_rpc)
                
            # add ref depth
            sample.append(ref_height)

            # add one block(1 ref, 2 view) to sample list
            sample_list.append(sample) 

    return sample_list     