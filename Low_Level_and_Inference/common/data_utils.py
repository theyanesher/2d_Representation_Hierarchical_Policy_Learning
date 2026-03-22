import numpy as np

def process_pointmap(pointmap, range_limit=4.0, compress=True):
    """
    Handles 3-channel pointmaps (x, y, z) with values in [-range_limit, range_limit].
    """
    # Use 32760 to stay safely within the signed int16 range [-32768, 32767]
    scale_factor = 32760 / range_limit
    
    if compress:
        # 1. Clip to ensure values stay within [-range_limit, range_limit]
        clipped = np.clip(pointmap, -range_limit, range_limit)
        # 2. Scale and cast
        return (clipped * scale_factor).astype(np.int16)
    else:
        # 3. Unscale back to float32 meters
        return pointmap.astype(np.float32) / scale_factor
    
def process_plucker(raymap, max_val=1.1, compress=True):
    """
    Handles 6-channel Plucker maps (d_x, d_y, d_z, m_x, m_y, m_z).
    """
    # Using 32760 to leave a tiny bit of headroom below the int16 max (32767)
    scale_factor = 32760 / max_val 
    
    if compress:
        clipped = np.clip(raymap, -max_val, max_val)
        return (clipped * scale_factor).astype(np.int16)
    else:
        return raymap.astype(np.float32) / scale_factor

def process_depth(depth, max_depth=1.6, compress=True):
    if compress:
        # Scale: 0.0m -> 0, 10.0m -> 10000
        clipped = np.clip(depth, 0, max_depth)
        return (clipped * 1000.0).astype(np.uint16)
    else:
        # Unscale: 10000 -> 10.0m
        return depth.astype(np.float32) / 1000.0