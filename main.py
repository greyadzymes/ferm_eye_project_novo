from flask import Flask, render_template, Response, redirect, url_for, session
import threading
import back_end

app = Flask(__name__)
app.secret_key = 'GDEA'
recording_status = False
threading_status = True

# add a threading.thread() function to continually update the frame_buffer 

# Home page
@app.route('/', methods=['GET', 'POST'])
def home_page():
    global threading_status
    if threading_status:
        threading.Thread(target=back_end.frame_buffer_update).start()
        threading_status = False
    return render_template('home_page.html')

# Video page
@app.route('/video_page', methods=['GET', 'POST'])
def video_page_render():
    global recording_status
    recording_status = session.get('recording', False)
    if recording_status:
        return render_template('YOLO_video_page.html')
    else:
        return render_template('video_page.html')

# Start recording
@app.route("/start_recording", methods=['GET', 'POST'])
def start_recording():
    global recording_status
    session['recording'] = True
    recording_status = True
    threading.Thread(target = back_end.yolo_stream).start()
    return redirect(url_for('video_page_render'))

# Stop recording
@app.route('/stop_recording', methods=['GET', 'POST'])
def stop_recording():
    global recording_status
    session['recording'] = False
    recording_status = False
    return redirect(url_for('video_page_render'))

# Video feed
@app.route('/video_feed')
def video_feed():
    return Response(back_end.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# YOLO video analysis without video
@app.route('/yolo_video_feed', methods=['GET', 'POST'])
def yolo_video_feed():
    global recording_status
    session['recording'] = True
    recording_status = session.get('recording')
    return Response(back_end.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)



# to run in virtual env 
# set FLASK_RUN_HOST=0.0.0.0
# set FLASK_RUN_PORT=9000        
# set FLASK_APP=main.py
# flask run