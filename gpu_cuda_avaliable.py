import torch

# GPU 사용 가능 여부 확인
print("GPU 사용 가능:", torch.cuda.is_available())

# 사용 가능한 GPU 개수
print("사용 가능한 GPU 개수:", torch.cuda.device_count())

if torch.cuda.is_available():
    # 현재 사용 중인 GPU 이름
    print("현재 GPU:", torch.cuda.get_device_name(0))
    
    # GPU 메모리 정보
    print("총 GPU 메모리:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    print("사용 가능한 GPU 메모리:", torch.cuda.memory_allocated(0) / 1024**3, "GB")

