import pandas as pd

class StatsCollector:
    def __init__(self):
        self.data = {}

    def update(self, objects, frame_num):
        for obj_id, (_, _, label) in objects.items():
            if obj_id not in self.data:
                self.data[obj_id] = {
                    "label": label,
                    "first_frame": frame_num,
                    "last_frame": frame_num,
                    "frames_seen": 1
                }
            else:
                self.data[obj_id]["last_frame"] = frame_num
                self.data[obj_id]["frames_seen"] += 1

    def save_summary(self):
        df = pd.DataFrame(self.data).T
        df.to_csv("output/object_summary.csv", index=True)

        print("\n===== SUMMARY =====")
        print(df.groupby("label").size())
