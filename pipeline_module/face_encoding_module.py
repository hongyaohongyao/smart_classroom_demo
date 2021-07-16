from face_recog.models import face_recog
from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK


class FaceEncodingModule(BaseModule):

    def __init__(self, skippable=True):
        super(FaceEncodingModule, self).__init__(skippable=skippable)

    def process_data(self, data):
        data.face_encodings = face_recog.face_encodings(data.frame, data.face_locations)
        return TASK_DATA_OK

    def open(self):
        super(FaceEncodingModule, self).open()
        pass
