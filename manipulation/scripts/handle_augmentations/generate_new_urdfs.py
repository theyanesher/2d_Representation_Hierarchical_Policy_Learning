import numpy as np
#Uncomment these lines just to run main from this script
from manipulation.scripts.handle_augmentations.handle_scaling import Scale_Handle
from manipulation.scripts.handle_augmentations.handle_shifting_aug import Handle_Shift
from manipulation.scripts.handle_augmentations.random_apply_handle_augmentation import RandomApply
import pickle
import xml.etree.ElementTree as ET
import os
from termcolor import cprint

# Storage Furnitures
generate_urdf_ids = [
	45413, 45420, 45427, 45443, 45504, 45594, 45620, 45623, 45633, 45636, 45667, 45767, 45841, 45916, 45922, 45936, 45937, 45950, 45984, 46092, 45670, 46107, 46109, 46130, 46134, 46197, 46334, 46401, 46443, 46456, 47178, 47180, 47182, 47187, 47227, 47238, 47254, 47466, 47565, 47577, 47648, 47742, 47747, 47808, 47817, 47954, 47963, 47976, 48010, 48013, 
	48036, 48258, 48379, 48381, 48797, 48855, 48859, 49042, 49182, 49188, 
	45007, 45087, 45091, 45092, 45130, 45134, 45135, 45146, 45159, 45164, 45166, 45168, 45173, 45177, 45189, 45212, 45213, 45247, 45261, 45267, 45354, 45372, 45374, 45385, 45387, 45403, 45415, 45419, 45423, 45503, 45505, 45524, 45573, 45575, 45606, 45612, 45621, 45622, 45632, 45638, 45642, 45645, 45662, 45671, 45676, 45677, 45687, 45699, 45710, 45746, 45756, 45776, 45779, 45783, 45784, 45790, 45801, 45822, 45853, 45855,
	45908, 45915, 45940, 45948, 45949, 45963, 45964, 46002, 46019, 46029, 46033, 46037, 46044, 46045, 46060, 46084, 46108, 46117, 46120, 46123, 46132, 46145, 46179, 46180, 46199, 46230, 46277, 46380, 46427, 46430, 46439, 46452, 46466, 46537, 46549, 46556, 46598, 46616, 46699, 46700, 46741, 46744, 46847, 46856, 46859, 46889, 46906, 46944, 46955, 46981, 47021, 47024, 47088, 47089, 47183, 47185, 47207, 47233, 47252, 47278, 
	47290, 47296, 47388, 47391, 47419, 47438, 47514, 47585, 47595, 47601, 47613, 47632, 47701, 47729, 47853, 47926, 48018, 48023, 48051, 48271, 48413, 48452, 48467, 48490, 48491, 48513, 48517, 48519, 48686, 48721, 48740, 48746, 48878, 49140, 35059, 41004, 41083, 41529, 44781, 44826, 44853, 
]

failed_urdf_ids = [
	45443, 45504, 45633, 45667, 45767, 45841, 45922, 46107, 46109, 47180, 47238, 47466, 47565, 47817, 47954, 47963, 48013, 48036, 48381, 49182, 45007, 45087, 45091, 45134, 45166, 45177, 45247, 45385, 45403, 45419, 45642, 45671, 45699, 45746, 45779, 45908, 45915, 46044, 46117, 46132, 46380, 46430, 46699, 46744, 46906, 47391, 47419, 47514, 47601, 47632, 47729, 48023, 48051, 48413, 48467, 48490, 48491, 48519, 49140
]

# Microwave, Dishwasher, Refrigerator, Oven
# generate_urdf_ids = [
# 	7119, 7167, 7263, 7310, 11622, 11661, 11700, 11826, 12065, 12085, 12092, 12259, 12414, 12428, 12480, 12484, 12530, 12531, 12536, 12540, 12543, 12552, 12553, 12559, 12560, 12561, 12562, 12563, 12565, 12579, 12580, 12583, 12587, 12590, 12592, 12594, 12596, 12597, 12605, 12606, 12614, 12617, 7290, 7220, 7187, 7332, 7120, 7201, 7179, 
# 	11550, 12252, 12043, 11211, 11178, 10143, 10036, 10144, 11304, 12509, 10867, 11712, 10751, 10797, 10944, 10685, 10638, 10655, 10068, 12250, 10489, 12054, 12036, 12042
# ]

# failed_urdf_ids = [7119, 11661, 11826, 12065, 12085, 12414, 12428, 12480, 12484, 12530, 12560, 12563, 12565, 12579, 12583, 12592, 12594, 12606, 12614, 7332, 7120, 7201, 7179, 12043, 10036, 12509, 10751, 10797, 10685, 10638, 10655, 10068, 12250, 10489, 12042]

# 10 furnitures for evaluation
generate_urdf_ids = [
	40147, 44817, 44962, 45132, 45219, 45243, 45332, 45378, 45384, 45463
]

failed_urdf_ids = []

failed_ids = []

generate_different_urdf_times = 5

for urdf_id in generate_urdf_ids:
	# if urdf_id not in failed_urdf_ids:
	# 	continue
	cprint("======================= Generating new urdfs =======================", "green")
	cprint(f"Generating new urdfs for {urdf_id}", "green")
	link_name = "link_0"
	try:
		for i in range(generate_different_urdf_times):
			# copy the original obj folder to a new folder
			if os.path.exists(f"data/dataset/{urdf_id}_{i}"):
				os.system(f"rm -r data/dataset/{urdf_id}_{i}")

			os.system(f"cp -r data/dataset/{urdf_id} data/dataset/{urdf_id}_{i}")
			new_urdf_id = f"{urdf_id}_{i}"

			# generate a new urdf file
			scaling_factor = np.random.uniform(0.75, 1.2)
			mesh_output_path = f"data/dataset/{new_urdf_id}/handle_modified_scaled.obj"
			handle_scaling =  Scale_Handle(scaling_factor = scaling_factor, mesh_output_path = mesh_output_path)
			handle_shifting = Handle_Shift(handle_shift_sigma = np.array([0.06, 0.06, 0]), max_iters = 500)
			transforms_and_probs = [[handle_shifting,1], [handle_scaling, 1]]
			rand_apply = RandomApply(transforms_and_probs)
			output_urdf_path = rand_apply(asset_id=new_urdf_id, link_name=link_name, output_urdf_path=f"data/dataset/{new_urdf_id}/mobility.urdf", )
	except Exception as e:
		cprint(f"Failed to generate urdf for {urdf_id}", "red")
		failed_ids.append(urdf_id)

print(f"Failed ids: {failed_ids}")
