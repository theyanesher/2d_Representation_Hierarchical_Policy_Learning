import h5py
import cv2
import numpy as np
import os
from tqdm import tqdm

def resize_dataset(src_path, dest_path, target_size=(84, 84)):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    files = [f for f in os.listdir(src_path) if f.endswith('.h5')]
    
    for filename in tqdm(files, desc="Processing H5 files"):
        with h5py.File(os.path.join(src_path, filename), 'r') as f_src:
            with h5py.File(os.path.join(dest_path, filename), 'w') as f_dest:
                
                def copy_and_resize(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        is_scalar = obj.shape == ()
                        data = obj[()] if is_scalar else obj[:]

                        if name == 'obs/rgb':
                            # Resizing RGB
                            N, num_cams, H, W, C = data.shape
                            resized_data = np.zeros((N, num_cams, target_size[0], target_size[1], C), dtype=data.dtype)
                            
                            for t in range(N):
                                for cam in range(num_cams):
                                    resized_data[t, cam] = cv2.resize(data[t, cam], target_size, interpolation=cv2.INTER_AREA)
                            
                            # Use auto-chunking for resized data
                            dst_obj = f_dest.create_dataset(name, data=resized_data, chunks=True, compression="gzip")
                        else:
                            # Safely handle chunks for other datasets
                            if is_scalar:
                                dst_obj = f_dest.create_dataset(name, data=data)
                            else:
                                # Only copy chunks if they fit the data shape; otherwise, let h5py auto-chunk
                                chunks = obj.chunks
                                if chunks:
                                    for i, dim in enumerate(obj.shape):
                                        if chunks[i] > dim:
                                            chunks = True # Fallback to auto-chunking
                                            break
                                
                                dst_obj = f_dest.create_dataset(
                                    name, 
                                    data=data, 
                                    chunks=chunks, 
                                    compression=obj.compression
                                )
                        
                        for attr_name, attr_value in obj.attrs.items():
                            dst_obj.attrs[attr_name] = attr_value

                    elif isinstance(obj, h5py.Group):
                        if name not in f_dest:
                            dst_obj = f_dest.create_group(name)
                            for attr_name, attr_value in obj.attrs.items():
                                dst_obj.attrs[attr_name] = attr_value

                f_src.visititems(copy_and_resize)

src_dir = 'data/rgb/41510'
dest_dir = f'{src_dir}_84'
resize_dataset(src_dir, dest_dir)