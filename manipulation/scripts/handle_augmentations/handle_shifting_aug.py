import xml.etree.ElementTree as ET
import numpy as np

class Handle_Shift():
    def __init__(self, handle_shift_sigma, num_samples_to_generate, max_iters):
        
        assert handle_shift_sigma[2] == 0, "There should be no shift ALong the Z axis and hence the handle_shift in the Z direction should be 0"
        self.handle_shift_cov = np.diag(handle_shift_sigma**2)
        self.num_samples = num_samples_to_generate
        self.max_iters = max_iters


    def __call__(self, asset_id, multiaug_flag = False, link_name = 'link_0'):
        self.link_name = link_name  # The link whose data you want to extract
        if multiaug_flag:
            urdf_file = f"data/dataset/{asset_id}/mobility_modified.urdf"
        else:
            urdf_file = f"data/dataset/{asset_id}/mobility.urdf"
        link_dict = extract_link_data(urdf_file, self.link_name)
        visual_data_list = link_dict["visual"]
        global_coordinates_max_list = []
        global_coordinates_min_list = []
        global_coordinates_handle = None
        origin_handle = None
        global_coordinates_max_handle = None
        global_coordinates_min_handle = None
        for visual_data in visual_data_list:
            if visual_data["name"].find("front") != -1 or visual_data["name"].find("door") != -1:
                mesh_path = visual_data["geometry"]["mesh"]["filename"]
                vertices = read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
                origin = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
                global_coordinates = vertices + origin
                global_coordinates_max = np.max(global_coordinates, axis = 0)
                global_coordinates_min = np.min(global_coordinates, axis = 0)
                global_coordinates_max_list.append(global_coordinates_max)
                global_coordinates_min_list.append(global_coordinates_min)
                print(global_coordinates_max_list)
                #print(visual_data["name"], global_coordinates_max, global_coordinates_min)
            if visual_data["name"].find("handle") != -1:
                mesh_path = visual_data["geometry"]["mesh"]["filename"]
                print(visual_data["name"], mesh_path)
                vertices_handle = read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
                origin_handle = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
                global_coordinates_handle = vertices_handle + origin_handle
                global_coordinates_max_handle = np.max(global_coordinates_handle, axis = 0)
                global_coordinates_min_handle = np.min(global_coordinates_handle, axis = 0)
        global_max_xyz = np.max(global_coordinates_max_list, axis = 0)
        global_min_xyz = np.min(global_coordinates_min_list, axis = 0)
        print(global_max_xyz, global_min_xyz, global_coordinates_max_handle, global_coordinates_min_handle)
        samples = np.random.multivariate_normal(np.zeros(3), self.handle_shift_cov, size=self.num_samples)
        print(samples)
        sample = samples[0]
        shifted_global_coordinates_handle = global_coordinates_handle + sample
        print(global_coordinates_handle)
        print(shifted_global_coordinates_handle)
        unsucessful_augmentation = True
        shifted_global_coordinates_handle_list = []
        samples = None
        iters = 0
        while unsucessful_augmentation and iters < self.max_iters:
            samples = np.random.multivariate_normal(np.zeros(3), self.handle_shift_cov, size=self.num_samples)
            for sample in samples:
                shifted_global_coordinates_handle = global_coordinates_handle + sample
                #print(shifted_global_coordinates_handle, global_max_xyz, global_min_xyz)
                #print(shifted_global_coordinates_handle, global_max_xyz)
                #print(shifted_global_coordinates_handle > global_max_xyz)
                if np.any(shifted_global_coordinates_handle[:,:2] > global_max_xyz[:2]) or np.any(shifted_global_coordinates_handle[:, :2] < global_min_xyz[:2]):
                    print("TRUEEE")
                    unsucessful_augmentation = True
                    shifted_global_coordinates_handle_list = []
                    self.handle_shift_cov = self.handle_shift_cov/2
                    print("DIVIDING")
                    break
                else:
                    print("falseeee")
                    shifted_global_coordinates_handle_list.append(global_coordinates_handle + sample)
                    unsucessful_augmentation = False
            iters += 1
        print(self.handle_shift_cov)
        print(samples)
        if iters == self.max_iters:
            print("AUGMENTATIONS COULD NOT BE POSSIBLE")
            return None
        root, shift_coeff = self.modify_urdf(urdf_file, samples, origin_handle)
        return root, shift_coeff


    def extract_link_data(self, urdf_path):
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        # Initialize an empty dictionary to store link data
        link_data = {}

        # Search for the specific link by name
        link = root.find(f".//link[@name='{self.link_name}']")

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
                    inertial_data['inertia'] = inertia.attribextract_link_data
                link_data['inertial'] = inertial_data

        else:
            print(f"Link with name '{self.link_name}' not found.")

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

    def modify_urdf(self, urdf_path, samples, origin_sample):
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        print("CALLEDDDDDDD")
        # Example modification: Change the mass of a link
        link = root.find(f".//link[@name='{self.link_name}']")
        print(link)
        if link is not None:
            for visual in link.findall('visual'):
                name = visual.get('name')
                if name.find("handle") != -1:
                    origin = visual.find('origin')
                    xyz = origin.get("xyz")
                    print(origin_sample, xyz, origin_sample + samples[0], ' '.join(map(str, origin_sample + samples[0])))
                    #print(name, )
                    origin.attrib['xyz'] = ' '.join(map(str, origin_sample + samples[0]))
        return root, samples[0]



























































# ROUGH CODE (working code)
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
                inertial_data['inertia'] = inertia.attribextract_link_data
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

def modify_urdf(urdf_path, link_name, samples, origin_sample):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    print("CALLEDDDDDDD")
    # Example modification: Change the mass of a link
    link = root.find(f".//link[@name='{link_name}']")
    print(link)
    if link is not None:
        for visual in link.findall('visual'):
            name = visual.get('name')
            if name.find("handle") != -1:
                origin = visual.find('origin')
                xyz = origin.get("xyz")
                print(origin_sample, xyz, origin_sample + samples[0], ' '.join(map(str, origin_sample + samples[0])))
                #print(name, )
                origin.attrib['xyz'] = ' '.join(map(str, origin_sample + samples[0]))
    return root



def center_shift_coeff(asset_id, ):
    # Example usage:
    #asset_id = 40147
    urdf_file = f"data/dataset/{asset_id}/mobility.urdf"
    link_name = 'link_0'  # The link whose data you want to extract
    handle_shift_sigma = np.array([0.1, 0.1, 0])
    handle_shift_cov = np.diag(handle_shift_sigma**2)
    print("INITIAL HANDLE SHIFT COV", handle_shift_cov)
    num_samples = 1
    link_dict = extract_link_data(urdf_file, link_name)
    visual_data_list = link_dict["visual"]
    global_coordinates_max_list = []
    global_coordinates_min_list = []
    global_coordinates_handle = None
    origin_handle = None
    global_coordinates_max_handle = None
    global_coordinates_min_handle = None
    for visual_data in visual_data_list:
        if visual_data["name"].find("front") != -1 or visual_data["name"].find("door") != -1:
            mesh_path = visual_data["geometry"]["mesh"]["filename"]
            vertices = read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
            origin = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
            global_coordinates = vertices + origin
            global_coordinates_max = np.max(global_coordinates, axis = 0)
            global_coordinates_min = np.min(global_coordinates, axis = 0)
            global_coordinates_max_list.append(global_coordinates_max)
            global_coordinates_min_list.append(global_coordinates_min)
            print(global_coordinates_max_list)
            #print(visual_data["name"], global_coordinates_max, global_coordinates_min)
        if visual_data["name"].find("handle") != -1:
            mesh_path = visual_data["geometry"]["mesh"]["filename"]
            print(visual_data["name"], mesh_path)
            vertices_handle = read_obj_file(f"data/dataset/{asset_id}/{mesh_path}")
            origin_handle = np.array([float(x) for x in visual_data["origin"]["xyz"].split()])
            global_coordinates_handle = vertices_handle + origin_handle
            global_coordinates_max_handle = np.max(global_coordinates_handle, axis = 0)
            global_coordinates_min_handle = np.min(global_coordinates_handle, axis = 0)
    global_max_xyz = np.max(global_coordinates_max_list, axis = 0)
    global_min_xyz = np.min(global_coordinates_min_list, axis = 0)
    print(global_max_xyz, global_min_xyz, global_coordinates_max_handle, global_coordinates_min_handle)
    samples = np.random.multivariate_normal(np.zeros(3), handle_shift_cov, size=num_samples)
    print(samples)
    sample = samples[0]
    shifted_global_coordinates_handle = global_coordinates_handle + sample
    print(global_coordinates_handle)
    print(shifted_global_coordinates_handle)
    unsucessful_augmentation = True
    shifted_global_coordinates_handle_list = []
    samples = None
    while unsucessful_augmentation:
        samples = np.random.multivariate_normal(np.zeros(3), handle_shift_cov, size=num_samples)
        for sample in samples:
            shifted_global_coordinates_handle = global_coordinates_handle + sample
            #print(shifted_global_coordinates_handle, global_max_xyz, global_min_xyz)
            #print(shifted_global_coordinates_handle, global_max_xyz)
            #print(shifted_global_coordinates_handle > global_max_xyz)
            if np.any(shifted_global_coordinates_handle[:,:2] > global_max_xyz[:2]) or np.any(shifted_global_coordinates_handle[:, :2] < global_min_xyz[:2]):
                print("TRUEEE")
                unsucessful_augmentation = True
                shifted_global_coordinates_handle_list = []
                handle_shift_cov = handle_shift_cov/2
                print("DIVIDING")
                break
            else:
                print("falseeee")
                shifted_global_coordinates_handle_list.append(global_coordinates_handle + sample)
                unsucessful_augmentation = False
    print(handle_shift_cov)
    print(samples)
    root = modify_urdf(urdf_file, link_name, samples, origin_handle)
    return root
    


if __name__ == "__main__":
    '''root = center_shift_coeff()
    print(root)
    link_name = 'link_0'
    link = root.find(f".//link[@name='{link_name}']")
    print(link)
    if link is not None:
        for visual in link.findall('visual'):
            name = visual.get('name')
            if name.find("handle") != -1:
                origin = visual.find('origin')
                xyz = origin.get("xyz")
                print("XYZZZZZZZZZZZZ", xyz)'''
    handle_shift = Handle_Shift(handle_shift_sigma = np.array([0.1, 0.1, 0]), num_samples_to_generate = 5, max_iters = 100)
    root = handle_shift(asset_id = 40147)
    link_name = 'link_0'
    link = root.find(f".//link[@name='{link_name}']")
    print(link)
    if link is not None:
        for visual in link.findall('visual'):
            name = visual.get('name')
            if name.find("handle") != -1:
                origin = visual.find('origin')
                xyz = origin.get("xyz")
                print("XYZZZZZZZZZZZZ", xyz)
    






