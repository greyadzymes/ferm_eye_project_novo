import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import threading
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import session
from io import BytesIO
import time
import main

yolo_model = 'C:/Users/gdea/OneDrive - Novozymes A S/Documents/python_files/Useful_work_Programs/ferm_eye_project/yolo_training/foam_model.pt'
output_folder = '//ko/datasets/ds_mdev_fermentation/antifoam/'
camera_in_use = 0
classNames = ["foam", "fermentation", "foam_cap", "rolling foam"]
df_tank1 = pd.DataFrame(columns=['date', 'foam', 'fermentation', 'foam_cap', 'rolling foam'])

# Define a buffer to store frames
frame_buffer = []
max_buffer_size = 5

def frame_buffer_update():
    global frame_buffer
    # this is the threading.thread() function to continually update the frame_buffer 
    cap = cv2.VideoCapture(camera_in_use)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_buffer.append(frame)
        if len(frame_buffer) > 10:
            frame_buffer.pop(0)

def generate_frames():
    time.sleep(1)
    while True:
        global frame_buffer

        # frame processing 
        ret, buffer = cv2.imencode('.jpg', frame_buffer[-1])
        frame_data = buffer.tobytes()

        # adding this to reduce network requirments 
        time.sleep(0.2)
                
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')             

# threading.thread() for processing and then replace the video element on the Yolo video page with the generate_frames() display 
def yolo_stream():
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    global frame_buffer, df_tank1
    model = YOLO(f'{yolo_model}')

    while True:
        recording_status = main.recording_status
        if recording_status:
            frame = frame_buffer[-1]
            ret = True
            if ret:
                current_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                new_row = pd.DataFrame({'date': [current_time]})
                df_tank1 = pd.concat([df_tank1, new_row], ignore_index=True)
                results = model(frame, show=False, conf=0.4, save=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        class_name = classNames[cls]
                        label = f'{class_name}{conf}'
                        area_in_box = (x2 - x1) * (y2 - y1)
                        # need to add picture capture per unit of time or per percentage of foam or something 
                        if area_in_box:
                            time_of_box_capture = df_tank1.index[df_tank1['date'] == current_time][0]
                            df_tank1.loc[time_of_box_capture, f'{class_name}'] = (int(area_in_box) / 307200) 

                # can increase this to once every minute
                time.sleep(3)
                    
        else:
            pd.set_option('future.no_silent_downcasting', True)
            df_tank1.fillna(0, inplace=True)   
            
            

            df_tank1.to_csv(f'{output_folder}/{start_time}_tank01.csv', index=False)     
            df_tank1['date'] = df_tank1['date'].astype(str)
            df_tank1['date'] = df_tank1['date'].str.replace('-', '').str.replace('_', '') 
            df_tank1['date'] = pd.to_numeric(df_tank1['date'])  
            
            df_tank1['foam'] = pd.to_numeric(df_tank1['foam'])  
            df_tank1['fermentation'] = pd.to_numeric(df_tank1['fermentation'])  
            df_tank1['foam_cap'] = pd.to_numeric(df_tank1['foam_cap'])                                                                                                     
            df_tank1['rolling foam'] = pd.to_numeric(df_tank1['rolling foam'])  
      
            # graphing the chart elements against date 
            plt.plot(df_tank1['date'], df_tank1['foam'], label='foam')
            plt.plot(df_tank1['date'], df_tank1['fermentation'], label='fermentation')
            plt.plot(df_tank1['date'], df_tank1['foam_cap'], label='foam_cap')
            plt.plot(df_tank1['date'], df_tank1['rolling foam'], label='rolling foam')

            # chart titles and saving 
            plt.xlabel("(x3) Seconds of Fermentation")
            plt.ylabel("Screen Percentage")
            plt.title(f"Ferm Tank01 {current_time}")
            plt.legend()
            plt.savefig(f'{output_folder}/{start_time}_tank01.jpeg')
            plt.close()
            df_tank1.drop(df_tank1.index, inplace=True)
            df_tank1.drop(df_tank1.columns, axis=1, inplace=True)
            # might have to add this to reset the dataframe for multiple recordings 
            df_tank1 = pd.DataFrame(columns=['date', 'foam', 'fermentation', 'foam_cap', 'rolling foam'])

            break 


def chart_display(): 
    while True:
        recording_status = main.recording_status
        global df_tank1
        working_chart = df_tank1.copy()
        if recording_status:
            # can be increased to once every minute 
            time.sleep(3)
            img = BytesIO()
            
            pd.set_option('future.no_silent_downcasting', True)
            working_chart.fillna(0, inplace=True)
            
            # replaces the date time with a sequence of numbers for the chart 
            working_chart['date'] = range(1, len(working_chart) + 1)
            
            working_chart['foam'] = pd.to_numeric(working_chart['foam'])  
            working_chart['fermentation'] = pd.to_numeric(working_chart['fermentation'])  
            working_chart['foam_cap'] = pd.to_numeric(working_chart['foam_cap'])                                                                                                     
            working_chart['rolling foam'] = pd.to_numeric(working_chart['rolling foam']) 

            # chart generation  
            working_chart.plot(x='date', y=['foam', 'fermentation', 'foam_cap', 'rolling foam'], kind='line')
            plt.savefig(img, format='jpg')
            plt.close()
            img.seek(0)

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img.getvalue() + b'\r\n')
        else: 
            break    

    


       
if __name__ == '__main__':
    yolo_stream()


