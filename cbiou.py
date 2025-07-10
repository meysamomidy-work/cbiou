from track import *
from utils import match_bboxes, tlbr_to_xywh, tlbr_to_tlwh

class CBIOUTracker:
    def __init__(self, config:dict={}):
        self.max_age = config.get('max_age', 30)
        self.buffer_scale1 = config.get('buffer_scale1', 0.3)
        self.buffer_scale2 = config.get('buffer_scale2', 0.5)
        self.biou_threshold1 = config.get('biou_threshold1', 0.6)
        self.biou_threshold2 = config.get('biou_threshold2', 0.4)
        self.detection_threshold = config.get('detection_threshold', 0.4)
        self.min_frames_to_predict = config.get('min_frames_to_predict', 3)
        Track.init(self.max_age, self.min_frames_to_predict)

    def update(self, boxes): # tlwh
        detections = boxes[boxes[:, 4] > self.detection_threshold][:, :4]
        scores = boxes[boxes[:, 4] > self.detection_threshold][:, 4]
        tracks = Track.alive_tracks()
        dets = [tlbr_to_xywh(detection) for detection in detections]
        trks = [track.xywh for track in tracks]
        matches, unmatched_track_indices, unmatched_detection_indices = match_bboxes(trks, dets, self.biou_threshold1, self.buffer_scale1)
        for track_index, detection_index in matches:
            tracks[track_index].update(tlbr_to_tlwh(detections[detection_index]), score=scores[detection_index])
        remained_detections = [detections[unmatched_detection_index] for unmatched_detection_index in unmatched_detection_indices]
        remained_scores = [scores[unmatched_detection_index] for unmatched_detection_index in unmatched_detection_indices]
        remained_tracks = [tracks[unmatched_track_index] for unmatched_track_index in unmatched_track_indices]
        remained_dets = [tlbr_to_xywh(remained_detection) for remained_detection in remained_detections]
        remained_trks = [remained_track.xywh for remained_track in remained_tracks]
        matches, unmatched_track_indices, unmatched_detection_indices = match_bboxes(remained_trks, remained_dets, self.biou_threshold2, self.buffer_scale2)
        for track_index, detection_index in matches:
            remained_tracks[track_index].update(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index])
        for detection_index in unmatched_detection_indices:
            Track(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index])
            # if Track.FRAME_NUMBER == 0:
            #     Track(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index], state=STATE_NEW)
            # else:
            #     Track(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index])
        Track.predict_all()