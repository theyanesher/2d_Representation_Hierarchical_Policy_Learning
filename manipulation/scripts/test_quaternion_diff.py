import numpy as np
import pybullet as p

# for i in range(100):
#     q_a = np.random.rand(4)
#     q_b = np.random.rand(4)
#     q_a /= np.linalg.norm(q_a)
#     q_b /= np.linalg.norm(q_b)
#     q_diff_1 = np.arccos(2 * np.dot(q_a, q_b)**2 - 1)
    # q_diff_2 = 2 * np.arccos(np.abs(np.dot(q_a, q_b)))
    
    # print("diff: ", np.abs(q_diff_1 - q_diff_2))
    # assert np.abs(q_diff_1 - q_diff_2) < 1e-6
    
euler_1 = [0, 0, np.pi]
euler_2 = [0, 0, 0]
q_a = p.getQuaternionFromEuler(euler_1)
q_b = p.getQuaternionFromEuler(euler_2)    
q_diff_2 = 2 * np.arccos(np.abs(np.dot(q_a, q_b)))
print("diff: ", q_diff_2)
print("diff: ", np.abs(q_diff_2 - 2*np.pi))
print("diff: ", np.rad2deg(q_diff_2))
