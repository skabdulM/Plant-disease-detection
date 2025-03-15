from roboflow import Roboflow

rf = Roboflow(api_key="T33ELYAEzATjRBk9NmT6")
project = rf.workspace("projectfinal-kwcso").project("agro-0km0x")
version = project.version(1)
dataset = version.download("yolov8", location="./datasets/dataset-yolov8")
