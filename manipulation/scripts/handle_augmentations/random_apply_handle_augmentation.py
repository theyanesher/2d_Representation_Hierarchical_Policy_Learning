import numpy as np
#Uncomment these lines just to run main from this script
from manipulation.scripts.handle_augmentations.handle_scaling import Scale_Handle
from manipulation.scripts.handle_augmentations.handle_shifting_aug import Handle_Shift
import pickle
import xml.etree.ElementTree as ET
import os
#import matplotlib.pyplot as plt
class RandomApply():
    def __init__(self,transforms_and_probs):
        self.transforms_and_probs = transforms_and_probs
        
    def __call__(self, asset_id, link_name = "link_0"):
        output_urdf_path = f"data/dataset/{asset_id}/mobility.urdf"
        random_number = np.random.rand()
        itr = 0
        final_shift_coeff = None
        for transform_prob in self.transforms_and_probs:
            #print(transform_prob[1])
            assert transform_prob[1] <= 1.0 and transform_prob[1] >= 0.0, "Augmentation probabilities much be less than 1 and greater than 0"
            if random_number < transform_prob[1]:
                if itr == 0:
                    root, shift_coeff =  transform_prob[0](asset_id = asset_id, multiaug_flag = False, link_name=link_name)
                else:
                    root, shift_coeff =  transform_prob[0](asset_id = asset_id, multiaug_flag = True, link_name = link_name)
                if shift_coeff is not None:
                    final_shift_coeff = shift_coeff
                modified_urdf_string = ET.tostring(root, encoding="unicode")
                output_urdf_path = f"data/dataset/{asset_id}/mobility_modified.urdf"  # Permanent file path
                os.makedirs(os.path.dirname(output_urdf_path), exist_ok=True)
                with open(output_urdf_path, 'w') as f:
                    f.write(modified_urdf_string)
                itr += 1
        return output_urdf_path, final_shift_coeff
    

if __name__ == "__main__":
    scaling_factor = 2
    asset_id = 44817
    mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj"
    handle_scaling =  Scale_Handle(scaling_factor = scaling_factor, mesh_output_path = mesh_output_path)
    handle_shifting = Handle_Shift(handle_shift_sigma = np.array([0.1, 0.1, 0]), num_samples_to_generate = 5, max_iters = 100)
    transforms_and_probs = [[handle_shifting,1], [handle_scaling, 1]]
    rand_apply = RandomApply(transforms_and_probs)
    output_urdf_path = rand_apply(asset_id = asset_id)
    print("OUTPUT URDF PATH =",output_urdf_path)





