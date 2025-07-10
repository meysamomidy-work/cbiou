import numpy as np
import textwrap

STATE_UNCONFIRMED = 0
STATE_NEW = 1
STATE_TRACKING = 2
STATE_LOST = 3
STATE_DELETED = 4
STATE_NAMES = ['UNCONFIRMED', 'NEW', 'TRACKING', 'LOST', 'DELETED']

class Track:
    INSTANCES:list["Track"] = []
    MAX_AGE = 30
    MIN_FRAMES_TO_PREDICT = 5
    ID_COUNTER = 0
    FRAME_NUMBER = 0

    @staticmethod
    def alive_tracks() -> list["Track"]:
        return [track for track in Track.INSTANCES if track.state not in [STATE_DELETED]]

    @staticmethod
    def deleted_tracks() -> list["Track"]:
        return [track for track in Track.INSTANCES if track.state in [STATE_DELETED]]

    @staticmethod
    def predict_all() -> None:
        Track.FRAME_NUMBER += 1
        for track in Track.INSTANCES:
            if track.state not in [STATE_DELETED]:
                track.predict()
    
    @property
    def mot_format(self):
        return f"{int(Track.FRAME_NUMBER)},{int(self.id)},{round(self.bbox[0], 1)},{round(self.bbox[1], 1)},{round(self.bbox[2], 1)},{round(self.bbox[3], 1)},{round(self.score, 2)},-1,-1,-1"

    @property
    def clean_format(self):
        return textwrap.dedent(f"""
            **************************************************************************************************************
            id         -> {self.id}
            state      -> {self.state_name}
            bbox       -> {self.bbox}
            age        -> {self.age}
            score      -> {self.score}
            {f'last state -> {STATE_NAMES[self.last_state_before_deletion]}' if self.state == STATE_DELETED else ''}
            """).strip()
    
    @property
    def compressed_format(self):
        return f"{STATE_NAMES[self.state]}    {self.id}    {self.bbox}    {self.age}    {self.score}    {STATE_NAMES[self.last_state_before_deletion] if self.state == STATE_DELETED else ''}"

    @property
    def score(self):
        if len(self.scores) > 0:
            return float(np.mean(self.scores))
        else:
            return 0

    @property    
    def state_name(self):
        return STATE_NAMES[self.state]

    @property
    def tlbr(self):
        bbox = self.bbox
        return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    @property
    def xywh(self):
        bbox = self.bbox
        return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]])
    
    @property
    def valid(self):
        if self.age > Track.MAX_AGE:
            return False
        if self.state in [STATE_UNCONFIRMED, STATE_NEW] and self.age >= 2:
            return False
        if np.any(np.isnan(self.bbox)) or np.any(self.bbox[2:] <= 0):
            return False
        return True

    def __init__(self, bbox, score, state=None):
        if state == None:
            self.state = STATE_UNCONFIRMED
        else:
            self.state = state
        self.last_state_before_deletion = None
        self.bbox = np.array(bbox).copy()
        self.predict_history = []
        self.update_history = [np.array(bbox).copy()]
        self.scores = [float(score)]
        self.age = 0
        self.id = Track.ID_COUNTER
        Track.ID_COUNTER += 1
        Track.INSTANCES.append(self)

    def __str__(self):
        return self.clean_format

    def predict(self):
        self.age += 1
        if not self.valid:
            self.last_state_before_deletion = self.state
            self.state = STATE_DELETED
            return
        if self.state == STATE_TRACKING and self.age >= 2:
            self.state = STATE_LOST
        if self.state in [STATE_TRACKING, STATE_LOST]:
            delta = (self.update_history[-1] - self.update_history[-Track.MIN_FRAMES_TO_PREDICT]) / (Track.MIN_FRAMES_TO_PREDICT - 1)
            new_bbox = self.predict_history[-1] + delta
            self.predict_history.append(new_bbox)
            self.bbox = new_bbox.copy()
            
    def update(self, bbox, score):
        self.update_history.append(np.array(bbox).copy())
        self.scores.append(float(score))
        self.bbox = np.array(bbox).copy()
        self.age = 0
        if self.state == STATE_NEW and len(self.update_history) >= Track.MIN_FRAMES_TO_PREDICT:
            self.predict_history = self.update_history.copy()
            self.state = STATE_TRACKING
        if self.state == STATE_UNCONFIRMED:
            self.state = STATE_NEW
        if self.state == STATE_LOST:
            self.state = STATE_TRACKING