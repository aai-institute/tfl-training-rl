from training_rl.offline_rl.load_env_variables import load_env_variables

load_env_variables()
import mujoco_py

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <camera name="cam0" pos="-0.1 0.1 0.0" quat="0.707 0.707 0 0" />
        <body name="box" pos="0 0 0.2">
            <geom size="0.15 0.15 0.15" type="box"/>
            <joint axis="1 0 0" name="box:x" type="slide"/>
            <joint axis="0 1 0" name="box:y" type="slide"/>
        </body>
        <body name="floor" pos="0 0 0.025">
            <geom size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
</mujoco>
"""
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
image = sim.render(width=300, height = 300, camera_name='cam0', depth=False)
print(image)