import mujoco.viewer
import mujoco

# Load your new XML
model = mujoco.MjModel.from_xml_path("models/furuta_pendulum.xml")

data = mujoco.MjData(model)

# Launch the interactive viewer
mujoco.viewer.launch(model, data)