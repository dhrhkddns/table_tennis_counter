from pygrabber.dshow_graph import FilterGraph

def list_cameras():
    graph = FilterGraph()
    cameras = graph.get_input_devices()
    for i, cam in enumerate(cameras):
        print(f"Camera {i}: {cam}")

list_cameras()
