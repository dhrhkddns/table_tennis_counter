from roboflow import Roboflow
rf = Roboflow(api_key="IOJEJZBxTKUuevke1feJ")
project = rf.workspace("pingpong-ojuhj").project("ping-pong-detection-0guzq")
version = project.version(3)
dataset = version.download("yolov11")
                
                