import mujoco.viewer
import mujoco

# Load your new XML
# model = mujoco.MjModel.from_xml_path("furuta_pendulum.xml")
model = mujoco.MjModel.from_xml_path("furuta_pendulum.xml")
# model = mujoco.MjModel.from_xml_path("unmodified-furuta_pendulum.xml")
data = mujoco.MjData(model)

# Launch the interactive viewer
mujoco.viewer.launch(model, data)