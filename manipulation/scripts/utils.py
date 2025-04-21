import pybullet as p
import time

def show_in_bullet(obj_id):
    id = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    p.loadURDF("data/dataset/{}/mobility.urdf".format(obj_id), useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=id)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=id)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=id)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=id)
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1)
    p.disconnect(id)
    
if __name__ == "__main__":
    show_in_bullet("12540")