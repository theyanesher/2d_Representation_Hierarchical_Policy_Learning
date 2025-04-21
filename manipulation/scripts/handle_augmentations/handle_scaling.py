import xml.etree.ElementTree as ET
import numpy as np

class Scale_Handle():
    def __init__(self, scaling_factor, mesh_output_path):
        
        self.scaling_factor = scaling_factor
        assert scaling_factor != 0, "The scaling factor should not be = 0"
        self.mesh_output_path = mesh_output_path

    def __call__(self, asset_id, input_urdf=None, link_name = 'link_0'):

        self.link_name = link_name
        # if multiaug_flag:
        #     urdf_file = f"data/dataset/{asset_id}/mobility_modified.urdf"
        # else:
        #     urdf_file = f"data/dataset/{asset_id}/mobility.urdf"
        
        if input_urdf is not None:
            urdf_file = input_urdf
        else:
            urdf_file = f"data/dataset/{asset_id}/mobility.urdf"
        
        # link_name = 'link_0'  # The link whose data you want to extract
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
        self.mesh_output_name_list = []
        mesh_path_list = []
        scaled_global_coordinates_handle_list = []
        vertices_handle_list = []
        mean_handle_list = []
        for visual_data in visual_data_list:
            if visual_data["name"].find("front") != -1 or visual_data["name"].find("door") != -1:
                mesh_path = visual_data["geometry"]["mesh"]["filename"]
                vertices = self.read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
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
                self.mesh_output_name_list.append(self.mesh_output_path[:-4] + mesh_path[first_index:last_index+1] + ".obj")
                print(visual_data["name"], mesh_path, mesh_path[first_index:last_index+1])
                total_mesh_file = f"data/dataset/{asset_id}/{mesh_path}"
                mesh_path_list.append(total_mesh_file)
                vertices_handle = self.read_obj_file(total_mesh_file)
                vertices_handle_list.append(vertices_handle)
                origin_handle = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
                mean_handle = np.mean(vertices_handle, axis = 0)
                mean_handle_list.append(mean_handle)
                scaled_vertices_handle = (vertices_handle - mean_handle) * self.scaling_factor + mean_handle
                scaled_global_coordinates_handle = scaled_vertices_handle + origin_handle
                print("ORIGIN HANDLE", origin_handle)
                scaled_global_coordinates_handle_list.append(scaled_global_coordinates_handle)
                scaled_global_coordinates_max_handle = np.max(scaled_global_coordinates_handle, axis = 0)
                scaled_global_coordinates_min_handle = np.min(scaled_global_coordinates_handle, axis = 0)
                #break
        global_max_xyz = np.max(global_coordinates_max_list, axis = 0)
        global_min_xyz = np.min(global_coordinates_min_list, axis = 0)
        #print("global check", global_max_xyz, global_min_xyz, scaled_global_coordinates_max_handle, scaled_global_coordinates_min_handle, len(scaled_global_coordinates_handle_list), scaled_global_coordinates_handle_list[0][:,:2].shape, scaled_global_coordinates_handle_list[1].shape, scaled_global_coordinates_handle_list[2].shape, scaled_global_coordinates_handle.shape)
        unsucessful_augmentation = True
        overflow_cond = True
        while unsucessful_augmentation:
            for scaled_global_coordinates_handle in  scaled_global_coordinates_handle_list:
                print("HOW MANY TIMES")
                scaled_global_coordinates_max_handle = np.max(scaled_global_coordinates_handle, axis = 0)
                scaled_global_coordinates_min_handle = np.min(scaled_global_coordinates_handle, axis = 0)
                print("global check", global_max_xyz, global_min_xyz, scaled_global_coordinates_max_handle, scaled_global_coordinates_min_handle)
                if np.any(scaled_global_coordinates_handle[:,:2] > global_max_xyz[:2]) or np.any(scaled_global_coordinates_handle[:, :2] < global_min_xyz[:2]):
                    overflow_cond = True
                    print("BREAK")
                    break
                else:
                    overflow_cond = False
            if overflow_cond:
                print("TRUEEE")
                unsucessful_augmentation = True
                self.scaling_factor = self.scaling_factor/2
                print("NEW SCALING FACTOR", self.scaling_factor)
                for i, vertices_handle in enumerate(vertices_handle_list):
                    scaled_vertices_handle = (vertices_handle - mean_handle) * self.scaling_factor + mean_handle
                    #scaled_vertices_handle = (vertices_handle - mean_handle) * self.scaling_factor
                    scaled_global_coordinates_handle = scaled_vertices_handle + origin_handle
                    scaled_global_coordinates_handle_list[i] = scaled_global_coordinates_handle
                    print("MEAN HANDLE", mean_handle, "ORIGIN HANDLE", origin_handle, "TOTAL", np.max(scaled_global_coordinates_handle, axis = 0), np.min(scaled_global_coordinates_handle, axis = 0) )
                #break
            else:
                unsucessful_augmentation = False
        print("SELF SCALING FACTOR", self.scaling_factor)
        self.save_modified_handle_obj(mesh_path_list, mean_handle_list)
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

                
        return np.array(vertices)

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

    def save_modified_handle_obj(self, mesh_path_list, mean_handle_list):
        
        for mesh_path, mesh_output_path, mean_handle in zip(mesh_path_list, self.mesh_output_name_list, mean_handle_list):
            mesh_list = []
            with open(mesh_path, 'r') as infile, open(mesh_output_path, 'w') as outfile:
                for line in infile:
                    # If the line starts with 'v', it represents a vertex
                    if line.startswith('v '):
                        # Modify the vertex (apply translation)
                        parts = line.split()
                        x, y, z = map(float, parts[1:4])
                        # Apply translation to the vertex coordinates
                        #print("SCALING FACTORRRRRRRRRRRRRRRRRRRRRR", self.scaling_factor)
                        x = (x - mean_handle[0])* self.scaling_factor + mean_handle[0]
                        y = (y - mean_handle[1]) * self.scaling_factor + mean_handle[1]
                        z = (z - mean_handle[2]) * self.scaling_factor + mean_handle[2]
                        '''x = (x - mean_handle[0])* self.scaling_factor
                        y = (y - mean_handle[1]) * self.scaling_factor
                        z = (z - mean_handle[2]) * self.scaling_factor'''
                        # Write the modified vertex to the output file
                        outfile.write(f"v {x} {y} {z}\n")
                    else:
                        # For all other lines (faces, normals, etc.), just copy as is
                        outfile.write(line)



















def extract_link_data(urdf_path, link_name):
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


def read_obj_file(obj_file_path):
    vertices = []
    faces = []

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

            
    return np.array(vertices)

def modify_urdf(urdf_path, link_name, modified_obj_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    #print("CALLEDDDDDDD")
    # Example modification: Change the mass of a link
    link = root.find(f".//link[@name='{link_name}']")
    #print(link)
    if link is not None:
        for visual in link.findall('visual'):
            name = visual.get('name')
            if name.find("handle") != -1:
                geometry = visual.find('geometry')
                if geometry is not None:
                    # Handle geometry types like mesh, box, cylinder, etc.
                    geometry_type = "mesh"
                    geometry.find(geometry_type).attrib = {'filename': modified_obj_path}
                    origin = visual.find('origin')
                    xyz = origin.get("xyz")
                    #print("ORIGINNNNNNN", xyz )
    return root

def save_modified_handle_obj(mesh_path, mesh_output_path, scaling_factor, mean_handle):
    with open(mesh_path, 'r') as infile, open(mesh_output_path, 'w') as outfile:
        for line in infile:
            # If the line starts with 'v', it represents a vertex
            if line.startswith('v '):
                # Modify the vertex (apply translation)
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                # Apply translation to the vertex coordinates
                x = (x - mean_handle[0])* scaling_factor + mean_handle[0]
                y = (y - mean_handle[1]) * scaling_factor + mean_handle[1]
                z = (z - mean_handle[2]) * scaling_factor + mean_handle[2]
                # Write the modified vertex to the output file
                outfile.write(f"v {x} {y} {z}\n")
            else:
                # For all other lines (faces, normals, etc.), just copy as is
                outfile.write(line)




def scale_handle():
    # Example usage:
    asset_id = 45132
    urdf_file = f"data/dataset/{asset_id}/mobility.urdf"
    link_name = 'link_0'  # The link whose data you want to extract
    scaling_factor = 1
    link_dict = extract_link_data(urdf_file, link_name)
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
    for visual_data in visual_data_list:
        if visual_data["name"].find("front") != -1:
            mesh_path = visual_data["geometry"]["mesh"]["filename"]
            vertices = read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
            origin = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
            global_coordinates = vertices + origin
            global_coordinates_max = np.max(global_coordinates, axis = 0)
            global_coordinates_min = np.min(global_coordinates, axis = 0)
            global_coordinates_max_list.append(global_coordinates_max)
            global_coordinates_min_list.append(global_coordinates_min)
            #print(visual_data["name"], global_coordinates_max, global_coordinates_min)
        if visual_data["name"].find("handle") != -1:
            mesh_path = visual_data["geometry"]["mesh"]["filename"]
            #print(visual_data["name"], mesh_path)
            total_mesh_file = f"data/dataset/{asset_id}/{mesh_path}"
            vertices_handle = read_obj_file(total_mesh_file)
            origin_handle = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
            mean_handle = np.mean(vertices_handle, axis = 0)
            scaled_vertices_handle = (vertices_handle - mean_handle) * scaling_factor + mean_handle
            scaled_global_coordinates_handle = scaled_vertices_handle + origin_handle
            scaled_global_coordinates_max_handle = np.max(scaled_global_coordinates_handle, axis = 0)
            scaled_global_coordinates_min_handle = np.min(scaled_global_coordinates_handle, axis = 0)
    global_max_xyz = np.max(global_coordinates_max_list, axis = 0)
    global_min_xyz = np.min(global_coordinates_min_list, axis = 0)
    #print(global_max_xyz, global_min_xyz, scaled_global_coordinates_max_handle, scaled_global_coordinates_min_handle)
    unsucessful_augmentation = True
    while unsucessful_augmentation:
        if np.any(scaled_global_coordinates_handle[:,:2] > global_max_xyz[:2]) or np.any(scaled_global_coordinates_handle[:, :2] < global_min_xyz[:2]):
            #print("TRUEEE")
            unsucessful_augmentation = True
            scaling_factor = scaling_factor/2
            scaled_vertices_handle = (vertices_handle - mean_handle) * scaling_factor + mean_handle
            scaled_global_coordinates_handle = scaled_vertices_handle + origin_handle
        else:
            unsucessful_augmentation = False
    #print(scaling_factor)
    #mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj"
    save_modified_handle_obj(total_mesh_file, mesh_output_path, scaling_factor, mean_handle)
    root = modify_urdf(urdf_file, link_name, mesh_output_path, )
    return root
    


if __name__ == "__main__":
    asset_id = 40147
    mesh_output_path = f"data/dataset/{asset_id}/handle_modified_scaled.obj"
    scale_handle = Scale_Handle(scaling_factor = 2, mesh_output_path = mesh_output_path)
    scale_handle(asset_id = 40147)