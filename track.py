import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def process_video(video_path, model_path="best.pt", output_path="output.mp4"):
    # Load model
    model = YOLO(model_path)
    
    # Video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Counting logic
    # Define a line: (start_x, start_y), (end_x, end_y)
    # Default: Horizontal line at 70% height
    line_y = int(h * 0.7)
    line_start = (0, line_y)
    line_end = (w, line_y)
    
    # Store track history to detect crossing
    track_history = defaultdict(lambda: [])
    counts = {0: 0, 1: 0} # 0: e-bike, 1: bicycle
    class_names = {0: "e-bike", 1: "bicycle"}
    counted_ids = set()

    print(f"Processing video {video_path}...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w_box, h_box = box
                
                # Current position
                current_pos = (float(x), float(y))
                track = track_history[track_id]
                track.append(current_pos)
                if len(track) > 30:  # retain 30 frames
                    track.pop(0)
                
                # Check crossing
                # Simple logic: if previous y < line_y and current y >= line_y (moving down)
                # or vice versa
                if len(track) > 1 and track_id not in counted_ids:
                    prev_y = track[-2][1]
                    curr_y = current_pos[1]
                    
                    # Assuming camera view where vehicles move down -> up or up -> down
                    # We count both directions for now, or check vector
                    if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                        if cls in counts:
                            counts[cls] += 1
                            counted_ids.add(track_id)
                            
        # Draw Visuals
        # 1. Line
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
        
        # 2. Counts
        text = f"e-bike: {counts[0]} | bicycle: {counts[1]} | Total: {sum(counts.values())}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 3. YOLO Annotations
        annotated_frame = results[0].plot()
        
        # Combine (plot() returns a new array, but we drew line/text on original 'frame', 
        # actually plot() draws on a copy of original frame passed to track?)
        # Let's just overlay our text/line on result.plot() output
        display_frame = results[0].plot()
        cv2.line(display_frame, line_start, line_end, (0, 255, 0), 2)
        cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(display_frame)

    cap.release()
    out.release()
    print(f"Processing complete. Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--model", default="best.pt", help="Path to model weights")
    args = parser.parse_args()
    
    process_video(args.video, model_path=args.model)
