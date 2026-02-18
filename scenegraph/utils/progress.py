# scenegraph/utils/progress.py

import time
from scenegraph.utils.gpu_monitor import get_gpu_mem, format_time

def update_progress_bar(pbar, start_time):
    used, total = get_gpu_mem()
    elapsed = format_time(time.time() - start_time)

    pbar.set_postfix({
        "GPU": f"{used:.1f}/{total:.1f} GB",
        "Time": elapsed
    })
