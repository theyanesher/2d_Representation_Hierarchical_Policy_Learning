import torch
import requests
import numpy as np
from io import BytesIO
from diffusers import DiffusionPipeline
from PIL import Image
import os


import xml.etree.ElementTree as ET
import numpy as np
# 0.1036405 * 3
class Scale_Handle():
    def __init__(self, scaling_factor, mesh_output_path, threshold_for_handles = 0.1036405 * 3, threshold_for_knob_handles = 0.047419 * 2):
        self.threshold_for_handles = threshold_for_handles
        self.threshold_for_knob_handles = threshold_for_knob_handles
        self.scaling_factor = scaling_factor
        assert scaling_factor != 0, "The scaling factor should not be = 0"
        self.mesh_output_path = mesh_output_path

    def __call__(self, asset_id, multiaug_flag=False, link_name = 'link_0', urdf_file = ""):
        self.link_name = link_name
        print(urdf_file)
        #link_name = 'link_0'  # The link whose data you want to extract
        link_dict = self.extract_link_data(urdf_file, link_name)
        visual_data_list = link_dict["visual"]
        global_coordinates_max_list = []
        global_coordinates_min_list = []
        global_coordinates_handle = None
        origin_handle = None
        global_coordinates_max_handle = None
        global_coordinates_min_handle = None
        vertices_handle = None
        scaled_vertices_handle = None
        mesh_path = None
        mean_handle = None
        imp_indices = None
        self.mesh_output_name_list = []
        mesh_path_list = []
        scaled_global_coordinates_handle_list = []
        vertices_handle_list = []
        mean_handle_list = []
        
        #print("VISUAL DATA LIST", visual_data_list)
        for visual_data in visual_data_list:
            if visual_data["name"].find("front") != -1 or visual_data["name"].find("door") != -1:
                #print("FRONTTTTTTTT", visual_data["name"])
                mesh_path = visual_data["geometry"]["mesh"]["filename"]
                vertices, imp_indices = self.read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
                origin = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
                global_coordinates = vertices + origin
                global_coordinates_max = np.max(global_coordinates, axis = 0)
                global_coordinates_min = np.min(global_coordinates, axis = 0)
                global_coordinates_max_list.append(global_coordinates_max)
                global_coordinates_min_list.append(global_coordinates_min)
                #print(visual_data["name"], global_coordinates_max, global_coordinates_min)
            if visual_data["name"].find("handle") != -1:
                mesh_path = visual_data["geometry"]["mesh"]["filename"]
                first_index, last_index = self.find_number_coordinates(mesh_path)
                self.mesh_output_name_list.append(self.mesh_output_path[:-4] + "/" + mesh_path[first_index:last_index+1] + ".obj")
                directory_path = os.path.dirname(self.mesh_output_path[:-4] + "/" + mesh_path[first_index:last_index+1] + ".obj")
                os.makedirs(directory_path, exist_ok=True)
                total_mesh_file = f"data/dataset/{asset_id}/{mesh_path}"
                mesh_path_list.append(total_mesh_file)
                vertices_handle_org, imp_indices = self.read_obj_file(total_mesh_file)
                #print("VVVVVVVVV IMPPPPPPPPPP", imp_indices)
                #vertices_handle_list.append(vertices_handle_org)
                origin_handle = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
                #print("VVVVVVVVV IMPPPPPPPPPP", imp_indices)
                vertices_handle = [vertices_handle_org[i] for i in imp_indices]
                #print(" id ont know what is happeneing", np.array(vertices_handle)[:,2])
                mean_handle = np.mean(vertices_handle, axis = 0)
                mean_handle_list.append(mean_handle)
                scaled_vertices_handle = (vertices_handle - mean_handle) * self.scaling_factor + mean_handle
                #scaled_vertices_handle = vertices_handle
                scaled_global_coordinates_handle = scaled_vertices_handle + origin_handle
                vertices_handle_list.append(scaled_global_coordinates_handle)
                scaled_global_coordinates_handle_list.append(scaled_global_coordinates_handle)
                scaled_global_coordinates_max_handle = np.max(scaled_global_coordinates_handle, axis = 0)
                scaled_global_coordinates_min_handle = np.min(scaled_global_coordinates_handle, axis = 0)
                #print("MEANNNNNNN HANDLE", mean_handle, scaled_global_coordinates_max_handle, scaled_global_coordinates_min_handle )
                #break
        #print("PRINTTTTTTTTT", scaled_global_coordinates_handle_list, vertices_handle)
        scaled_global_coordinates_max_handle = np.array([-np.inf, -np.inf, -np.inf])
        scaled_global_coordinates_min_handle = np.array([np.inf, np.inf, np.inf])
        for scaled_global_coordinates_handle in scaled_global_coordinates_handle_list:
            #scaled_global_coordinates_handle[:, 2] += 10
            scaled_global_coordinates_max_handle_temp = np.max(scaled_global_coordinates_handle, axis = 0)
            scaled_global_coordinates_min_handle_temp = np.min(scaled_global_coordinates_handle, axis = 0)
            if scaled_global_coordinates_max_handle_temp[2] > scaled_global_coordinates_max_handle[2]:
                scaled_global_coordinates_max_handle = scaled_global_coordinates_max_handle_temp.copy()
            if scaled_global_coordinates_min_handle_temp[2] < scaled_global_coordinates_min_handle[2]:
                scaled_global_coordinates_min_handle = scaled_global_coordinates_min_handle_temp.copy()
        stacked_arrays = np.vstack(mean_handle_list)
        mean_handle = np.mean(stacked_arrays, axis=0)
        #print("BEGINNINGGGGGGGGGGG", scaled_global_coordinates_max_handle, scaled_global_coordinates_min_handle, mean_handle_list)
        size_of_handle = scaled_global_coordinates_max_handle - scaled_global_coordinates_min_handle
        scaling_factor_now = 1
        #print("HERREEEEEEEE", size_of_handle)
        if (size_of_handle[0] > size_of_handle[1] * 3):
            #print("HERREEEEEEEE nowwwwwwwwwwwwww")
            if size_of_handle[0] < self.threshold_for_handles:
                #print(urdf_file)
                scaling_factor_now = self.threshold_for_handles/size_of_handle[0]
                #print("TRIGGGGGERRRRRRRRREDDDDDDDD", scaling_factor_now)
                for vertices_handle, mean_handle in zip(vertices_handle_list, mean_handle_list):
                    #print("MEANNNNNNNNN", mean_handle)
                    scaled_handle = (vertices_handle - origin_handle - mean_handle) * scaling_factor_now + mean_handle + origin_handle
                    #scaled_handle = scaled_handle + origin_handle
                    scaled_global_coordinates_handle_list.append(scaled_handle)
        elif (size_of_handle[1] > size_of_handle[0] * 3):
            #print("HERREEEEEEEE NOWWWWWWWWWWWWWW")
            if size_of_handle[1] < self.threshold_for_handles:
                #print(urdf_file)
                scaling_factor_now = self.threshold_for_handles/size_of_handle[1]
                #print("TRIGGGGGERRRRRRRRREDDDDDDDD", scaling_factor_now)
                for vertices_handle, mean_handle in zip(vertices_handle_list, mean_handle_list):
                    #print("MEANNNNNNNNN", mean_handle)
                    scaled_handle = (vertices_handle - origin_handle - mean_handle) * scaling_factor_now + mean_handle + origin_handle
                    #scaled_handle = scaled_handle + origin_handle
                    scaled_global_coordinates_handle_list.append(scaled_handle)
        else:
            print("COMING HEREEEEEEE")
            if (size_of_handle[1] < self.threshold_for_knob_handles) or (size_of_handle[0] < self.threshold_for_knob_handles):
                print("COMING HEREEEEEEE 22222222222")
                scaling_factor_now = self.threshold_for_knob_handles/size_of_handle[1]
                #print(urdf_file)
                for vertices_handle, mean_handle in zip(vertices_handle_list, mean_handle_list):
                    #print("MEANNNNNNNNN", mean_handle)
                    scaled_handle = (vertices_handle - origin_handle - mean_handle) * scaling_factor_now + mean_handle + origin_handle
                    #scaled_handle = scaled_handle + origin_handle
                    scaled_global_coordinates_handle_list.append(scaled_handle)


        scaled_global_coordinates_max_handle = np.array([-np.inf, -np.inf, -np.inf])
        scaled_global_coordinates_min_handle = np.array([np.inf, np.inf, np.inf])
        for scaled_global_coordinates_handle in scaled_global_coordinates_handle_list:
            scaled_global_coordinates_max_handle_temp = np.max(scaled_global_coordinates_handle, axis = 0)
            scaled_global_coordinates_min_handle_temp = np.min(scaled_global_coordinates_handle, axis = 0)
            if scaled_global_coordinates_max_handle_temp[2] > scaled_global_coordinates_max_handle[2]:
                scaled_global_coordinates_max_handle = scaled_global_coordinates_max_handle_temp.copy()
            if scaled_global_coordinates_min_handle_temp[2] < scaled_global_coordinates_min_handle[2]:
                scaled_global_coordinates_min_handle = scaled_global_coordinates_min_handle_temp.copy()


        global_max_xyz = np.max(global_coordinates_max_list, axis = 0)
        global_min_xyz = np.min(global_coordinates_min_list, axis = 0)
        average = (global_max_xyz + global_min_xyz)/2
        possible_shift_one = scaled_global_coordinates_min_handle - average
        possible_shift_two = scaled_global_coordinates_max_handle - average
        if possible_shift_one[2] < possible_shift_two[2]:
            shift_needed = possible_shift_one
        else:
            shift_needed = possible_shift_two
        shift_needed[0:2] = mean_handle[0:2] - average[0:2]
        scaled_global_coordinates_handle = scaled_global_coordinates_handle - shift_needed
        print("MEAN BEFORE", mean_handle)
        mean_handle = mean_handle - shift_needed
        print(average, mean_handle)
        if average[2] > mean_handle[2]:
            print("REFLECTED")  
            reflection_coefficient_list = np.zeros(vertices_handle_org.shape[0]) 
            itr = 0
            imp_indices = list(imp_indices)
            for i in range(scaled_global_coordinates_handle.shape[0]):
                reflection_coefficient = 2 * (scaled_global_coordinates_handle[i][2] - average[2])
                reflection_coefficient_list[imp_indices[itr]] = reflection_coefficient
                scaled_global_coordinates_handle[i][2] = scaled_global_coordinates_handle[i][2] + reflection_coefficient
                itr = itr + 1
        else:
            reflection_coefficient_list = np.zeros(vertices_handle_org.shape[0])
        self.save_modified_handle_obj(mesh_path_list, mean_handle_list, shift_needed=shift_needed, reflection_coefficient_list = reflection_coefficient_list, imp_indices= imp_indices, scaling_factor=scaling_factor_now)
        root = self.modify_urdf(urdf_file, link_name)
        return root, None

    def find_number_coordinates(self,s):
        first_index = -1
        last_index = -1
        
        for index, char in enumerate(s):
            if char.isdigit():
                if first_index == -1:  # If it's the first number found
                    first_index = index
                last_index = index  # Update last_index with the current number's index
                
        return first_index, last_index

    def extract_link_data(self, urdf_path, link_name):
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        # Initialize an empty dictionary to store link data
        link_data = {}

        # Search for the specific link by name
        link = root.find(f".//link[@name='{link_name}']")

        if link is not None:
            # Extract the link's data (visual, collision, and inertial properties)
            
            # Extract link name
            link_data['name'] = link.get('name')
            # Extract visual properties
            visual_data = []
            itr = 0
            for visual in link.findall('visual'):
                visual_info = {}
                visual_info['name'] = visual.get('name')
                origin = visual.find('origin')
                if origin is not None:
                    visual_info['origin'] = {
                        'xyz': origin.get('xyz'),
                        'rpy': origin.get('rpy')
                    }
                geometry = visual.find('geometry')
                if geometry is not None:
                    # Handle geometry types like mesh, box, cylinder, etc.
                    geometry_type = "mesh"
                    visual_info['geometry'] = {geometry_type: geometry.find(geometry_type).attrib}
                visual_data.append(visual_info)
            link_data['visual'] = visual_data

            # Extract collision properties
            collision_data = []
            for collision in link.findall('collision'):
                collision_info = {}
                origin = collision.find('origin')
                if origin is not None:
                    collision_info['origin'] = {
                        'xyz': origin.get('xyz'),
                        'rpy': origin.get('rpy')
                    }
                geometry = collision.find('geometry')
                if geometry is not None:
                    # Handle geometry types like mesh, box, cylinder, etc.
                    geometry_type = "mesh"
                    collision_info['geometry'] = {geometry_type: geometry.find(geometry_type).attrib}
                collision_data.append(collision_info)
            link_data['collision'] = collision_data

            # Extract inertial properties (if available)
            inertial = link.find('inertial')
            if inertial is not None:
                inertial_data = {}
                mass = inertial.find('mass')
                if mass is not None:
                    inertial_data['mass'] = mass.attrib
                inertia = inertial.find('inertia')
                if inertia is not None:
                    inertial_data['inertia'] = inertia.attrib
                link_data['inertial'] = inertial_data

        else:
            print(f"Link with name '{link_name}' not found.")

        return link_data
    

    def read_obj_file(self, obj_file_path):
        vertices = []
        faces = []
        used_indices = set()  # To store the indices of vertices used in faces

        # Read the OBJ file
        with open(obj_file_path, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace
                line = line.strip()

                # Skip empty lines or comments
                if not line or line.startswith('#'):
                    continue

                # Process vertex lines (v x y z)
                if line.startswith('v '):
                    vertex = list(map(float, line.split()[1:]))  # Convert to float and store
                    vertices.append(vertex)

                # Process face lines (f v1 v2 v3 ...)
                elif line.startswith('f '):
                    # Extract vertex indices from the face line (e.g., f 1 2 3 or f 1/1/1 2/2/2 3/3/3)
                    face_vertices = line.split()[1:]  # Ignore the "f" part
                    for vertex_data in face_vertices:
                        vertex_index = int(vertex_data.split('/')[0]) - 1  # Get vertex index (1-based to 0-based)
                        used_indices.add(vertex_index)
        return np.array(vertices), used_indices


    def modify_urdf(self, urdf_path, link_name):
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        #print("CALLEDDDDDDD")
        # Example modification: Change the mass of a link
        link = root.find(f".//link[@name='{self.link_name}']")
        #print(link)
        itr = 0
        if link is not None:
            for visual in link.findall('visual'):
                name = visual.get('name')
                if name.find("handle") != -1:
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        # Handle geometry types like mesh, box, cylinder, etc.
                        geometry_type = "mesh"
                        geometry.find(geometry_type).attrib = {'filename': self.mesh_output_name_list[itr]}
                        itr += 1
                        origin = visual.find('origin')
                        xyz = origin.get("xyz")
                        #print("ORIGINNNNNNN", xyz )
        return root

    def save_modified_handle_obj(self, mesh_path_list, mean_handle_list, shift_needed, reflection_coefficient_list, imp_indices, scaling_factor):

        for mesh_path, mesh_output_path, mean_handle in zip(mesh_path_list, self.mesh_output_name_list, mean_handle_list):
            mesh_list = []
            itr = 0
            sum = 0
            k =1
            maximum = -999999
            minimum = 9000000
            with open(mesh_path, 'r') as infile, open(mesh_output_path, 'w') as outfile:
                for line in infile:
                    # If the line starts with 'v', it represents a vertex
                    if line.startswith('v '):
                        # Modify the vertex (apply translation)
                        
                        if itr in imp_indices:
                            #print("IFFFF", itr)
                            parts = line.split()
                            x, y, z = map(float, parts[1:4])
                            if scaling_factor == 1:
                                x = (x - mean_handle[0])* self.scaling_factor + mean_handle[0] - shift_needed[0]
                                y = (y - mean_handle[1]) * self.scaling_factor + mean_handle[1] - shift_needed[1]
                                z = (z - mean_handle[2]) * self.scaling_factor + mean_handle[2] - shift_needed[2] - reflection_coefficient_list[itr]
                            else:
                                #print(mesh_output_path)
                                x = (x - mean_handle[0])* scaling_factor + mean_handle[0] - shift_needed[0]
                                y = (y - mean_handle[1]) * scaling_factor + mean_handle[1] - shift_needed[1]
                                z = (z - mean_handle[2]) * scaling_factor + mean_handle[2] - shift_needed[2] - reflection_coefficient_list[itr]
                                
                        else:
                            #print("ELSEEEEEEEEE", itr)
                            parts = line.split()
                            x, y, z = map(float, parts[1:4])
                        itr = itr + 1
                        # Write the modified vertex to the output file
                        outfile.write(f"v {x} {y} {z}\n")
                    else:
                        # For all other lines (faces, normals, etc.), just copy as is
                        outfile.write(line)



if __name__ == "__main__":
    asset_id = 40147
    mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj"
    scale_handle = Scale_Handle(scaling_factor = 2, mesh_output_path = mesh_output_path)
    scale_handle(asset_id = 40147)









