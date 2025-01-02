if __name__ == "__main__":
    from ultralytics import YOLO
    import multiprocessing

    multiprocessing.freeze_support()  # Windows 환경에서 필요한 경우 사용

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\data.yaml",
        epochs=100,
        imgsz=640
    )