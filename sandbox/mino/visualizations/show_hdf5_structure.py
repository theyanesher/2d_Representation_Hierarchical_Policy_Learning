import h5py
import sys

def print_hdf5_structure(name, obj, indent=0):
    """Recursive function to print structure of HDF5 object."""
    spacer = '  ' * indent
    if isinstance(obj, h5py.Group):
        print(f"{spacer}Group: {name}")
        for key in obj:
            print_hdf5_structure(key, obj[key], indent + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{spacer}Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

def explore_hdf5_file(file_path):
    """Opens and prints the structure of the HDF5 file."""
    with h5py.File(file_path, 'r') as hdf_file:
        print(f"File: {file_path}")
        hdf_file.visititems(lambda name, obj: print_hdf5_structure(name, obj, indent=name.count('/')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hdf5_structure.py <path_to_hdf5_file>")
    else:
        explore_hdf5_file(sys.argv[1])

