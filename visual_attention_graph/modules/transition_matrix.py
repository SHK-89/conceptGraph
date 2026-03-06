import numpy as np


class TransitionMatrixBuilder:

    def __init__(self, object_ids):
        self.object_ids = sorted(object_ids)

        self.id_to_idx = {obj: i for i, obj in enumerate(self.object_ids)}

        n = len(self.object_ids)

        self.matrix = np.zeros((n, n))

    def update(self, scanpath):

        for i in range(len(scanpath) - 1):

            src_objects = scanpath[i]
            dst_objects = scanpath[i + 1]

            for src in src_objects:
                for dst in dst_objects:

                    src_idx = self.id_to_idx[src]
                    dst_idx = self.id_to_idx[dst]

                    self.matrix[src_idx, dst_idx] += 1