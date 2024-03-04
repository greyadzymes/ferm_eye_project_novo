# set up your virtual environment in cmd
## python -m venv ferm_eye_env
# activate the virtual environment 
## ferm_eye_env\Scripts\activate


# YOLOv8 Ultralytics home page Home https://docs.ultralytics.com/

# imports
from flask import Flask, render_template, Response, send_file
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
import threading
import math
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import io
import base64


yolo_model = 'C:/Users/gdea/OneDrive - Novozymes A S/Documents/python_files/Useful_work_Programs/ferm_eye_project/yolo_training/foam_model.pt'
output_folder = f'C:/Users/gdea/OneDrive - Novozymes A S/Documents/GitHub/Ferm_eye_project/pd_outputs'

recording = False # this is a global variable that will have to change when more than 1 camera is in use possibly name per camera 'recording_cam01' or something
camera_in_use = 0 # turn into a user variable with the drop down menue 
camera_lock = threading.Lock()
classNames = ["foam", "fermentation", "foam_cap", "rolling foam"]



# video feed on main video page without analysis 
def generate_frames():
    cap = cv2.VideoCapture(camera_in_use)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # dipslaying the frame 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    cap.release()        


# below is a first pass for streaming the yolo fucntional video with boxes on displayed video, next step to pull the box info out and graph over time 
def yolo_stream():
    global yolo_model
    model = YOLO(f'{yolo_model}')
    cap = cv2.VideoCapture(camera_in_use)
    df_tank1 = pd.DataFrame(columns=['date', 'foam', 'fermentation', 'foam_cap', 'rolling foam'])
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    while True:
        global recording
        if recording == True:
            ret, frame = cap.read()
            # add in the def function here and then add a button for this or something idk 
            if ret:
                current_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                df_tank1 = df_tank1.append({'date': current_time}, ignore_index=True)
                results = model(frame, show=False, conf=0.4, save=False)
                print (f'len of results: {(len(results))}')
                
                for r in results:
                    boxes=r.boxes
                    for box in boxes: 

                        # setting up the boxes in the video frames 
                        x1,y1,x2,y2=box.xyxy[0]
                        print('tensor coordinates of box: ', x1, y1, x2, y2) # prints tensors get it to print the integer box corners 
                        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                        print("int coordinates of  box: " ,x1,y1,x2,y2)  # this one should not show the tensor numbers, use this to get the area of the boxes which will then be graphed 
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
                        conf=math.ceil((box.conf[0]*100))/100
                        cls=int(box.cls[0])
                        class_name=classNames[cls]
                        label=f'{class_name}{conf}'
                        area_in_box = (x2-x1)*(y2-y1) 
                        print(f'{class_name} detected')
                        print(f'area in {class_name} box: ', area_in_box)
                        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        cv2.rectangle(frame, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                        cv2.putText(frame, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

                        # 307200 total pixles in each frame 

                        # creating the graph and a pd data frame for holding the box values and adds values 
                        # add in a def here to take the info from the pd dataframe and graph it 
                        if area_in_box:
                            time_of_box_capture = df_tank1.index[df_tank1['date'] == current_time][0]
                            df_tank1.loc[time_of_box_capture, f'{class_name}'] = (int(area_in_box)/307200)
                            #df_tank1 = df_tank1.append({class_name: int(area_in_box)}, ignore_index=True) # functional with readouts in cmd 
                        
                        

                    # actual generation of the frame on the page 
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()    
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'            )
                    
                return df_tank1      # added in as the return that will be picked up by the graphing function     

        else:
            df_tank1.fillna('0', inplace=True)    
            df_tank1.to_csv(f'{output_folder}/{start_time}_tank01.csv', index=False)     
            


            # turning chart strings into int so a graph can be made from numeric data 
            df_tank1['date'] = df_tank1['date'].str.replace('-', '').str.replace('_', '') # constant reocurring issues with this line for some reason 
            df_tank1['date'] = pd.to_numeric(df_tank1['date'])  
            df_tank1['foam'] = pd.to_numeric(df_tank1['foam'])  
            df_tank1['fermentation'] = pd.to_numeric(df_tank1['fermentation'])  
            df_tank1['foam_cap'] = pd.to_numeric(df_tank1['foam_cap'])                                                                                                     
            df_tank1['rolling foam'] = pd.to_numeric(df_tank1['rolling foam'])  


            # make the chart and save it based on end table data 
            Tank01_chart = df_tank1.plot(x = 'date', y = ['foam', 'fermentation', 'foam_cap', 'rolling foam'], kind = 'line')
            plt.xlabel("Time of Fermentation")
            plt.ylabel("Screen Percentage")
            plt.title(f"Ferm Tank01 {datetime}")
            img_bytes = io.BytesIO()
            plt.savefig(f'{output_folder}/{start_time}_tank01.jpeg')
            
            df_tank1.drop(df_tank1.index, inplace=True)
            df_tank1.drop(df_tank1.columns, axis=1, inplace=True)
            cap.release()       
            break 
       
 
    # move this into the YOLO_video_page
#def pd_chart_generation():
#    df_tank1 = pd.DataFrame(columns=['date', 'foam', 'fermentation', 'foam_cap', 'rolling foam'])
#   while recording_for_chart is True:

def global_recording_false():
    global recording
    recording = False

def global_recording_true():
    global recording
    recording = True    


# THINGS TO DO 
# 3. MAKE A PAGE WHERE ALL 6 VIDEOS CAN BE WATCHED AT THE SAME TIME POSSIBLY EXCLUDING ANY GRAPHS 
# 4. GET IMAGES OF THE FERM TANKS SO THE STOCK IMAGES CAN BE REPLACED 
# 5. understand why the live video integration with YOLO here works and why the last attempt was getting the index error
# 6. get the video recording working again 
# 7. get the charts made based off of the box sizes 
# 8. get a video of a foaming tank to use for test case scenarios   
# 9. use the pandas data frame as storage for the values created from the boxes produced by YOLO    



# create an instance of flask 
app = Flask(__name__)

# home page 
@app.route('/', methods = ['GET', 'POST'])
def home_page():
    return render_template('home_page.html')


# page with the video feed and future analytics feed
@app.route('/video_page', methods = ['GET', 'POST'])
# this works but the global variable is not properly being changed when the button is clicked 
def video_page_render():   # this brings you to the straight video page if recording is False and to the YOLO page if recording is TRUE
    global recording
    if recording == True:
        return render_template('YOLO_video_page.html')
    else:
        return render_template('video_page.html')   

# changes global variable when 'start recording' button is pressed 
@app.route("/start_recording", methods =['GET', 'POST'])
def start_recording():
    global_recording_true()
    return render_template('YOLO_video_page.html')

# changes the global variable when the 'stop recording' button is pressed 
@app.route('/stop_recording', methods = ['GET', 'POST'])
def stop_recording():
    global_recording_false()
    time.sleep(3)
    return render_template('video_page.html') 


# called by the .js file should show the live updating chart 
# this regularly breaks the program, everything works fine without it 
 
"""@app.route('/plot', methods = ['GET', 'POST'])
def plot():
    time.sleep(1)
    df_tank1 = yolo_stream()  # this variable might be bad
    # add in a while loop that causes the graph to be updated with the frames 
    def generate():
        while True: # change to global recording == true possibly  
            df_tank_for_graph = df_tank1
            # code to clean and process data omitted for brevity

            # preparing the dataframe for charting
            print (df_tank_for_graph)
            df_tank_for_graph.fillna('0', inplace=True)    # this is an issue at the moment possibly not getting a pd dataframe from the return of def yolo_strem()


            df_tank_for_graph['date'] = df_tank_for_graph['date'].str.replace('-', '').str.replace('_', '') # constant reocurring issues with this line for some reason 

            
            df_tank_for_graph['date'] = pd.to_numeric(df_tank_for_graph['date'])  
            df_tank_for_graph['foam'] = pd.to_numeric(df_tank_for_graph['foam'])  
            df_tank_for_graph['fermentation'] = pd.to_numeric(df_tank_for_graph['fermentation'])  
            df_tank_for_graph['foam_cap'] = pd.to_numeric(df_tank_for_graph['foam_cap'])                                                                                                     
            df_tank_for_graph['rolling foam'] = pd.to_numeric(df_tank_for_graph['rolling foam'])  

            #actually making the chart 
            Tank01_chart = df_tank_for_graph.plot(x = 'date', y = ['foam', 'fermentation', 'foam_cap', 'rolling foam'], kind = 'line')
            plt.xlabel("Time of Fermentation")
            plt.ylabel("Screen Percentage")
            plt.title(f"Ferm Tank01 {datetime}")

            # Convert the plot to PNG image data
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')

            # Encode the image data as a Base64 string
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            # Send the image data as a server-sent event
            yield 'data: data:image/png;base64,' + img_base64 + '\n\n'

            # Wait for 1 second before generating the next plot
            time.sleep(1)
    
    # Set the response headers to indicate that this is an SSE endpoint
    return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})"""





# makes the web page that will display the video feed from the camera 
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# the feed with the video and Yolo analysis with boxes 
@app.route('/yolo_video_feed')
def yolo_video_feed():
    return Response(yolo_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# since code is run sequentally you want the server to activate 
# after all dependencies and decorators have been defined 
# so run the 'app.run()' in the last line of the code 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

# to run in virtual env 
# set FLASK_APP=main.py
# flask run
