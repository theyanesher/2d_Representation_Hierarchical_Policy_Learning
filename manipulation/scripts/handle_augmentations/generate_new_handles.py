from generate_urdf_new_handles import generate_new_handle_dict, modify_urdf
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from manipulation.scripts.get_handle import get_handle

def put_handles_on_doors(urdf_file_path, link_name, base_directory, asset_id, new_handle = True):
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    #print("CALLEDDDDDDD")
    # Example modification: Change the mass of a link
    link = root.find(f".//link[@name='{link_name}']")
    #print(link)
    itr = 0
    if link is not None:
        for visual in link.findall('visual'):
            name = visual.get('name')
            if name.find("handle") != -1:
                #print(name)
                origin = visual.find('origin')
                xyz = origin.get("xyz")
                xyz = list(map(float, xyz.split()))
                #print(xyz)
    #print("BASE DIRECTORY", base_directory)
    dirs = os.listdir(base_directory)
    for dir in dirs:
        full_dir = base_directory + dir
        #print(base_directory + dir)
        inside_dirs = os.listdir(full_dir)
        for inside_dir in inside_dirs:
            handle_folder = full_dir +"/" + inside_dir
            #print(handle_folder, xyz, name)
            print("NAMEEEEEEE", name)
            handle_list = generate_new_handle_dict(handle_name = name, handle_position = xyz, handle_folder = handle_folder, base_path_name = "../../Handle_Dataset_Partnet/")
            os.makedirs(f"data/dataset/{asset_id}/New_Handle_Based_urdf/", exist_ok=True)
            output_file = f"data/dataset/{asset_id}/New_Handle_Based_urdf/mobility" + "_" + dir + "_" + inside_dir + ".urdf"
            #print(output_file)
            modify_urdf(urdf_file = urdf_file_path, output_file = output_file, handle_to_remove =  name, new_handles = handle_list)
            os.makedirs(f"data/get_handle/Generated_Handles_Samples/{asset_id}/", exist_ok=True)
            save_path = f"data/get_handle/Generated_Handles_Samples/{asset_id}/" + str(asset_id) + "_" + dir + "_" + inside_dir + ".mp4"
            get_handle(output_file, save_path, new_handle=new_handle, asset_id = asset_id)
        #print(files)
    #print(dirs)




asset_id = 44817
urdf_file_path = f"data/dataset/{asset_id}/mobility.urdf"
link_name = "link_0"
#obj_directory = "data/Handle_Dataset_Partnet/"
obj_directory = "data/Test_Handle_Dataset_Partnet/"
put_handles_on_doors(urdf_file_path, link_name, obj_directory, asset_id, new_handle = True)
