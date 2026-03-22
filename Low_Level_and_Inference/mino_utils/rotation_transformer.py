import numpy as np
import torch
import pytorch3d.transforms
from typing import Callable

class RotationTransformer:
    """
    A class to convert between different rotation representations, with
    conventions fixed at initialization.
    
    Supported representations:
    - 'matrix': (..., 3, 3) rotation matrix
    - 'quaternion': (..., 4) quaternion (w, x, y, z)
    - 'axis_angle': (..., 3) axis-angle vector (norm is angle in radians)
    - 'euler_angles': (..., 3) euler angles
    - 'rotation_6d': (..., 6) 6D continuous representation
    """

    def __init__(self, in_repr: str, out_repr: str, 
                 euler_convention: str = 'XYZ'):
        """
        Initializes the transformer.
        euler convention can be one of 'XYZ', 'ZYX'
        """
        self.valid_reprs = ['matrix', 'quaternion', 'axis_angle', 'euler_angles', 'rotation_6d']
        
        if in_repr not in self.valid_reprs:
            raise ValueError(f"Invalid in_repr '{in_repr}'. Must be one of {self.valid_reprs}")
        if out_repr not in self.valid_reprs:
            raise ValueError(f"Invalid out_repr '{out_repr}'. Must be one of {self.valid_reprs}")
            
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.euler_convention = euler_convention

        # --- Set up transform functions (in_repr -> out_repr) ---
        self._in_to_matrix = self._get_to_matrix_fn(self.in_repr, self.euler_convention)
        self._matrix_to_out = self._get_from_matrix_fn(self.out_repr, self.euler_convention)

        # --- Set up inverse transform functions (out_repr -> in_repr) ---
        self._out_to_matrix = self._get_to_matrix_fn(self.out_repr, self.euler_convention)
        self._matrix_to_in = self._get_from_matrix_fn(self.in_repr, self.euler_convention)

    def _get_to_matrix_fn(self, repr_name: str, convention: str) -> Callable:
        """Helper to create a "..._to_matrix" function."""
        if repr_name == 'matrix':
            return lambda data: data
        fn_name = f"{repr_name}_to_matrix"
        try:
            _fn = getattr(pytorch3d.transforms, fn_name)
        except AttributeError:
            raise ValueError(f"Could not find transform function: {fn_name}")
        if repr_name == 'euler_angles':
            return lambda data: _fn(data, convention)
        else:
            return lambda data: _fn(data)

    def _get_from_matrix_fn(self, repr_name: str, convention: str) -> Callable:
        """Helper to create a "matrix_to_..." function."""
        if repr_name == 'matrix':
            return lambda matrix: matrix
        fn_name = f"matrix_to_{repr_name}"
        try:
            _fn = getattr(pytorch3d.transforms, fn_name)
        except AttributeError:
            raise ValueError(f"Could not find transform function: {fn_name}")
        if repr_name == 'euler_angles':
            return lambda matrix: _fn(matrix, convention)
        else:
            return lambda matrix: _fn(matrix)

    def _convert_input(self, data):
        """Handles np.ndarray vs torch.Tensor input."""
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data_torch = torch.from_numpy(data.astype(np.float32))
        elif isinstance(data, torch.Tensor):
            data_torch = data
        else:
            raise TypeError(f"Input data must be np.ndarray or torch.Tensor, got {type(data)}")
        if data_torch.dtype == torch.float16:
            data_torch = data_torch.float()
        return data_torch, is_numpy

    def _convert_output(self, data_torch, is_numpy):
        """Handles torch.Tensor vs np.ndarray output."""
        if is_numpy:
            return data_torch.detach().numpy()
        else:
            return data_torch

    def transform(self, data):
        data_torch, is_numpy = self._convert_input(data)
        matrix = self._in_to_matrix(data_torch)
        output_torch = self._matrix_to_out(matrix)
        return self._convert_output(output_torch, is_numpy)

    def inverse_transform(self, data):
        data_torch, is_numpy = self._convert_input(data)
        matrix = self._out_to_matrix(data_torch)
        output_torch = self._matrix_to_in(matrix)
        return self._convert_output(output_torch, is_numpy)
    

if __name__ == "__main__":
    # Simple test
    rot_transformer = RotationTransformer(in_repr='quaternion', out_repr='rotation_6d')
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    rot6d = rot_transformer.transform(quat)
    quat_reconstructed = rot_transformer.inverse_transform(rot6d)
    print("Original Quaternion:", quat)
    print("6D Representation:", rot6d)
    print("Reconstructed Quaternion:", quat_reconstructed)

    quat_wrong = rot_transformer.inverse_transform(quat) # why doesn't this throw an error wtf
    