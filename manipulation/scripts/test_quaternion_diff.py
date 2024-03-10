import numpy as np

for i in range(100):
    q_a = np.random.rand(4)
    q_b = np.random.rand(4)
    q_a /= np.linalg.norm(q_a)
    q_b /= np.linalg.norm(q_b)
    q_diff_1 = np.arccos(2 * np.dot(q_a, q_b)**2 - 1)
    q_diff_2 = 2 * np.arccos(np.abs(np.dot(q_a, q_b)))
    
    # print("diff: ", np.abs(q_diff_1 - q_diff_2))
    assert np.abs(q_diff_1 - q_diff_2) < 1e-6