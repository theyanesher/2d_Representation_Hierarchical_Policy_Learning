import xml.etree.ElementTree as ET
import os
from pathlib import Path

'''def remove_handle_by_name(root, handle_name):
    """
    Remove all visual and collision elements that match the handle name.
    """
    # Iterate over all visual and collision elements to remove the ones with the given handle_name

    #visual_elements = root.findall(".//visual")
    for element in root.findall(".//visual"):
        print(f"Visual element name: {element.get('name')}")
    for link in root.findall(".//link"):
        for element in link.findall("visual"):  # Copy of visual elements
            name = element.get("name")
            print(f"NAMEEEEEEEEEEEEE: {name}")
            #link.remove(element)
            if name and handle_name in name:
                print(f"Removing visual element: {name}", element.find("geometry"), element.get("name"))
                link.remove(element)
    

def add_handle(root, handle_name, mesh_filename, position):
    """
    Add a new handle with the specified mesh file and position.
    """
    # Create the visual element
    visual = ET.SubElement(root, "visual", name=handle_name)
    origin = ET.SubElement(visual, "origin", xyz=" ".join(map(str, position)))
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh", filename=mesh_filename)
    
    # Create the collision element (it usually mirrors the visual element)
    collision = ET.SubElement(root, "collision", name=handle_name)
    origin = ET.SubElement(collision, "origin", xyz=" ".join(map(str, position)))
    geometry = ET.SubElement(collision, "geometry")
    mesh = ET.SubElement(geometry, "mesh", filename=mesh_filename)

def modify_urdf(urdf_file, output_file, handle_to_remove=None, new_handles=None):
    """
    Modify the URDF file by removing a handle and adding new ones.
    
    urdf_file: path to the URDF file
    output_file: path to the new URDF file to write the modified content
    handle_to_remove: the name of the handle to remove (None if no handle needs removal)
    new_handles: list of dictionaries containing new handle data to be added
    """
    # Parse the URDF XML
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Remove the specified handle
    if handle_to_remove:
        remove_handle_by_name(root, handle_to_remove)

    # Add new handles
    if new_handles:
        for handle in new_handles:
            add_handle(root, handle['name'], handle['mesh_filename'], handle['position'])
    
    # Write the modified XML to a new URDF file
    tree.write(output_file)'''

import xml.etree.ElementTree as ET

def remove_handle_by_name(root, handle_name, link_name = "link_0"):
    """
    Remove all visual and collision elements that match the handle name from the URDF.
    This function ensures the element is removed from the correct parent (<link>).
    """
    print("INSIDE remove handle by name", handle_name)
    # Iterate through all <link> elements in the tree
    link = root.find(f".//link[@name='{link_name}']")
    # Iterate through all visual elements inside each <link>
    for element in link.findall("visual"):
        name = element.get("name")
        if name and handle_name in name:
            #print(f"Identified element for removal: {name}")
            link.remove(element)  # Remove the element from its parent <link>
    return link

def modify_urdf(urdf_file, output_file, handle_to_remove=None, new_handles=None, link_to_modify = "link_0"):
    """
    Modify the URDF file by removing a handle and adding new ones to the same <link>.
    
    urdf_file: path to the URDF file
    output_file: path to the new URDF file to write the modified content
    handle_to_remove: the name of the handle to remove (None if no handle needs removal)
    new_handles: list of dictionaries containing new handle data to be added
    """
    # Parse the URDF XML
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Remove the specified handle and get the <link> where the handle was removed
    link_to_modify = remove_handle_by_name(root, handle_to_remove)

    # Add new handles to the same link if any
    if new_handles and link_to_modify is not None:
        for handle in new_handles:
            add_handle(link_to_modify, handle['name'], handle['mesh_filename'], handle['position'])

    # Write the modified XML to a new URDF file
    tree.write(output_file)
    #print(f"Modified URDF saved to: {output_file}")

def add_handle(link, name, mesh_filename, position):
    """
    Add a new handle to the specified <link> by creating a new visual element.
    """
    
    # Create the new visual element
    visual = ET.Element('visual', name=name)
    origin = ET.SubElement(visual, 'origin', xyz=" ".join(map(str, position)))
    geometry = ET.SubElement(visual, 'geometry')
    mesh = ET.SubElement(geometry, 'mesh', filename=mesh_filename)

    # Append the new visual element to the provided <link> element
    link.append(visual)
    #print(f"Added new handle: {name}, at position: {position}")      

        
def generate_new_handle_dict(handle_name, handle_position, handle_folder, base_path_name = "../../Handle_Dataset_Partnet/"):
    files = os.listdir(handle_folder)
    #print("FILES", files)
    file_path_list = []
    path = Path(handle_folder)
    last_two_parts = '/'.join(path.parts[-2:])
    for file in files:
        file_path_list.append(base_path_name + last_two_parts + "/" + file)
    #print(file_path_list)
    handle_list = []
    for file_path in file_path_list:
        handle_list.append({"name": handle_name, "mesh_filename": file_path, "position": handle_position})
    #print("HANDLE_LIST", handle_list)
    return handle_list


'''handle_list = generate_new_handle_dict(handle_name="handle-15", handle_position=[0.06925157730966305, 0, -0.3877265588637183], handle_folder="data/Handle_Dataset_Partnet/7305/13/")
urdf_path = "data/dataset/40147/mobility.urdf"
link_name = "link_0"
#handle = [{"name": "handle-15", "mesh_filename": "../../Handle_Dataset_Partnet/7305/13/new-9.obj", 'position': [0.06925157730966305, 0, -0.3877265588637183]}]
output_file = "data/dataset/40147/New_Handle_Based_urdf/mobility_7305_13.urdf"
modify_urdf(urdf_file = urdf_path, output_file = output_file, handle_to_remove = "handle-15", new_handles = handle_list)'''