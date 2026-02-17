import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, detections):
        updated_objects = {}

        for box in detections:
            x1, y1, x2, y2, label = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            assigned = False

            for obj_id, (px, py, _) in self.objects.items():
                dist = np.linalg.norm([cx - px, cy - py])
                if dist < self.max_distance:
                    updated_objects[obj_id] = (cx, cy, label)
                    assigned = True
                    break

            if not assigned:
                updated_objects[self.next_object_id] = (cx, cy, label)
                self.next_object_id += 1

        self.objects = updated_objects
        return self.objects
