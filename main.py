import shutil
from collections import defaultdict
import cv2
import os
import numpy as np
import yt_dlp
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_wtf import FlaskForm
from ultralytics import YOLO
from wtforms import SubmitField, BooleanField

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'Stinauma@80'
# Ensure UPLOAD_FOLDER exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class WelcomeForm(FlaskForm):
    test_data = BooleanField('Test Images')
    upload_url = BooleanField('Upload Image URL')
    video_url = BooleanField('Upload VIdeo URL')
    web_cam = BooleanField('Live Stream')
    you_tube = BooleanField('You_tube_video')
    desktop = BooleanField('Desktop')
    submit = SubmitField('Submit')


# # Function to clear static/processed folder every time we run the app.
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def generate_frames():
    cap = cv2.VideoCapture('downloaded_videos/downloaded_video.mp4')

    # Store the track history
    track_history = defaultdict(lambda: [])
    prev_frame = None  # To store the previous frame for GMC tracking

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Debug print statements
            print(f"Type of frame: {type(frame)}")
            if prev_frame is not None:
                print(f"Type of prev_frame: {type(prev_frame)}")
                if isinstance(frame, np.ndarray) and isinstance(prev_frame, np.ndarray):
                    if prev_frame.shape != frame.shape:
                        print(f"Frame size mismatch! Previous: {prev_frame.shape}, Current: {frame.shape}")
                        prev_frame = None  # Reset tracker or take necessary actions here
                else:
                    print("Frame or previous frame is not a NumPy array!")

            try:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                model = YOLO('model_weights/yolov10s.pt')

                results = model.track(frame, persist=True, tracker='botsort.yaml', classes=[0], iou=0.7, conf=0.2)

                # Debug print statements
                if not results:
                    print("No results returned from model.track.")
                    continue

                # Check if results[0] is not None and has boxes
                if results[0] is not None and hasattr(results[0], 'boxes'):
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.numpy()
                        track_ids = results[0].boxes.id.numpy()

                        # Visualize the results on the frame
                        annotated_frame = results[0].plot()

                        # Plot the tracks for each object (with its unique track ID)
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # Add the (x, y) center point

                            # Limit the history to the last 30 frames to keep it manageable
                            if len(track) > 30:
                                track.pop(0)

                            # Draw the tracking lines for this specific track_id
                            if len(track) > 1:  # Only draw if there are more than 1 point in history
                                for i in range(1, len(track)):
                                    pt1 = tuple(np.hstack(track[i - 1]).astype(np.int32))  # Previous point
                                    pt2 = tuple(np.hstack(track[i]).astype(np.int32))  # Current point

                                    # Dynamic thickness: get thinner as trace gets older
                                    thickness = max(1, int(5 * (1 - i / len(track))))

                                    # Draw a line only if points belong to the same object (track_id)
                                    cv2.line(annotated_frame, pt1, pt2, color=(230, 230, 230), thickness=thickness)

                        # Encode the frame in JPEG format
                        ret, buffer = cv2.imencode('.jpg', annotated_frame)
                        frame_bytes = buffer.tobytes()

                        # Yield each frame in a stream-compatible format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        print("No bounding boxes detected in the results.")

                        # Visualize the results on the frame
                        annotated_frame = results[0].plot()
                        # Encode the frame in JPEG format
                        ret, buffer = cv2.imencode('.jpg', annotated_frame)
                        frame_bytes = buffer.tobytes()

                        # Yield each frame in a stream-compatible format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Unexpected results format or missing 'boxes' attribute.")

            except cv2.error as e:
                print(f"OpenCV error: {e}. Resetting tracker state.")
                prev_frame = None  # Reset tracker if there's an error in GMC tracking

        else:
            # Break the loop if the end of the video is reached
            break

        # Update the previous frame for tracking
        prev_frame = frame

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


# Step 1: If GET, home page, If POST redirects to Step 2(user_choice)
# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    form = WelcomeForm()
    if request.method == 'POST':
        selected_items = []
        for field in ['test_data', 'upload_url', 'video_url', 'web_cam', 'you_tube', 'desktop', 'yolo_pre',
                      'yolo_custom']:

            if request.form.get(field) == 'y':
                selected_items.append(field)

                print(selected_items)
                if 'you_tube' in selected_items:
                    return redirect(url_for('you_tube_url'))
                # if 'test_data' in selected_items:
                #     car_models_df = pd.read_csv('data files/Car+names+and+make.csv')
                #     car_models_list = car_models_df['AM General Hummer SUV 2000'].to_list()
                #     return redirect(url_for('select_pics'))
                # elif 'desktop' in selected_items:
                #     car_models_df = pd.read_csv('data files/Car+names+and+make.csv')
                #     car_models_list = car_models_df['AM General Hummer SUV 2000'].to_list()
                #     return redirect(url_for('upload_image'))

    return render_template('index.html', form=form)


# Step 2
# If GET goes back to Step 1,  If POST (option to enter YouTube url) Step 3
@app.route('/you_tube_url', methods=['GET', 'POST'])
def you_tube_url():
    if request.method == 'POST':
        clear_directory('downloaded_videos')
        video_url = request.form.get("url")
        print(video_url)
        if not video_url:
            print('Here1')
            flash('No video URL provided', 'error')
            return redirect(url_for('index'))  # Redirect to a relevant page or return an error response

        # Step: Download the YouTube video
        ydl_opts = {'outtmpl': 'downloaded_videos/downloaded_video.mp4'}
        try:
            print('Here2')
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            flash(f'Error downloading video: {str(e)}', 'error')
            return redirect(url_for('index'))  # Redirect or return an error response

        # video_path = 'downloaded_videos/downloaded_video.mp4'
        clear_directory(directory_path='y_10/processed_videos')
        return redirect(url_for('display_processed_video'))

    # Adjust for smoother playback if needed
    return render_template('you_tube_url.html')


@app.route('/processed_video', methods=['GET', 'POST'])
def processed_video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/display_processed_video', methods=['GET', 'POST'])
def display_processed_video():
    return render_template('display.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
