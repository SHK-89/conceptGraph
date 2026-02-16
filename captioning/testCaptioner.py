import numpy as np
from intervlCaptioner_OLD import InterVLCaptioner

cap = InterVLCaptioner(
    model_path="C:/SCIoI/weights/InternVL-Chat-2B-V1_5.pth"
)

dummy = np.zeros((200,200,3), dtype=np.uint8)
print(cap.caption_object(dummy))