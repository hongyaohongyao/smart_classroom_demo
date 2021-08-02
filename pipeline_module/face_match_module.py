import numpy as np
from numpy import ndarray

from face_recog.models import face_recog
from pipeline_module.core.base_module import TASK_DATA_OK, BaseModule


class FaceMatchModule(BaseModule):

    def __init__(self, known_encodings: ndarray, skippable=True):
        super(FaceMatchModule, self).__init__(skippable=skippable)
        self.known_encodings = known_encodings

    def process_data(self, data):
        face_encodings = data.face_encodings
        known_encodings = self.known_encodings

        face_distances = [face_recog.face_distance(known_encodings,
                                                   face_encoding).tolist() for face_encoding in face_encodings]
        face_distances = np.array(face_distances)

        raw_face_labels = np.argmin(face_distances, axis=1) if face_distances.shape[0] > 0 else np.array([])
        # 处理标签相同的结果
        vis = np.empty(known_encodings.shape[0], dtype=np.int)
        vis.fill(-1)
        face_labels = np.empty(face_distances.shape[0], dtype=np.int)
        face_labels.fill(-1)
        for i, lbl in enumerate(raw_face_labels):
            if lbl == -1:
                continue
            argue_i = vis[lbl]
            if vis[lbl] < 0:
                vis[lbl] = i
                face_labels[i] = lbl
            else:
                argue_lbl = raw_face_labels[argue_i]
                argue_prob = face_distances[argue_i, argue_lbl]
                cur_prob = face_distances[i, lbl]
                if cur_prob < argue_prob:
                    vis[lbl] = i
                    face_labels[i] = lbl
                    face_labels[argue_i] = -1

        data.face_labels = face_labels
        data.face_probs = np.array([face_distances[i, lbl] if lbl > 0 else 1 for i, lbl in enumerate(face_labels)])
        data.raw_face_labels = np.array(raw_face_labels)
        data.raw_face_probs = np.array([face_distances[i, lbl] for i, lbl in enumerate(face_labels)])

        return TASK_DATA_OK

    def open(self):
        super(FaceMatchModule, self).open()
        pass
