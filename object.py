import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as npqq
import cv2
import matplotlib.pyplot as plt
import urllib.request
import tarfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3
import RPi.GPIO as GPIO
import time
import pyttsx3
import os 
import threading
engine = pyttsx3.init()
GPIO.setmode(GPIO.BCM)
import time


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

TRIG = 21
ECHO = 20
maxTime = 0.04



engine = pyttsx3.init()
engine.setProperty('rate',180)


engine.say("Hello")
engine.say("i am")
engine.say("Jarvis")
engine.say("i will guide")
engine.say("you ")
engine.say("please wait a moment")


engine.runAndWait()



# Load model from .h5 and save as Saved Model:
import tensorflow as tf
model = tf.keras.models.load_model(r"/home/pi/Project/objectdetector.h5")
tf.saved_model.save(model, r"/home/pi/Project/tmp_model")


# 
engine.setProperty('rate',190)
engine.say("I can Detect now")
engine.runAndWait()


net = cv2.dnn.readNetFromONNX(r'/home/pi/Project/objectdetector.onnx')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.43.224:81/stream")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
cap.set(cv2.CAP_PROP_FPS, 2);
cap.set(cv2.CAP_PROP_POS_FRAMES , 1);




font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
# Define class names and sort them alphabatically as this is how tf.keras remembers them
label_names = ['Keyboard', 'Mouse','Laptop']
label_names.sort()
#mouse
alarm_sound = pyttsx3.init()
alarm_sound.setProperty('rate', 170)
#keyboard




def voice_alarm(alarm_sound):
    alarm_sound.say("I Think the Mouse Device! is Detected in!")
    alarm_sound.say(dis)
    alarm_sound.say("inch")        
    alarm_sound.runAndWait()
    if dis <=100:
        alarm_sound.say("Near You")
        alarm_sound.runAndWait()
    else:
        alarm_sound.say("Far You")
        alarm_sound.runAndWait()
def voice_alarm2(alarm_sound):
    alarm_sound.say("I Think the Keyboard Device! is Detected in!")
    alarm_sound.say(dis)
    alarm_sound.say("inch")        
    alarm_sound.runAndWait()
    if dis <=100:
        alarm_sound.say("Near You")
        alarm_sound.runAndWait()
    else:
        alarm_sound.say("Far You")
        alarm_sound.runAndWait()
def voice_alarm3(alarm_sound):
    alarm_sound.say("I Think the Laptop Device! is Detected in!")
    alarm_sound.say(dis)
    alarm_sound.say("inch")
    if dis <=100:
        alarm_sound.say("Near You")
        alarm_sound.runAndWait()
        
    else:
        alarm_sound.say("Far You")
        alarm_sound.runAndWait() 
    
def voice_alarm4(alarm_sound):
    alarm_sound.setProperty('rate',180) 
    alarm_sound.say('Alertttttt! you are close in')
    alarm_sound.say(dis)
    alarm_sound.say('inch in object! please be aware')
    alarm_sound.runAndWait()



def voice_alarm44(alarm_sound):
    alarm_sound.setProperty('rate',200)
    alarm_sound.say('Stay back! you are in')
    alarm_sound.say(dis)
    alarm_sound.say('inch in object!')
    
    
    alarm_sound.runAndWait()

    
def voice_alarm5(alarm_sound):
    alarm_sound.setProperty('rate',150)
    alarm_sound.say("Shutting Down")
    alarm_sound.runAndWait()
    
while True:
    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)

    GPIO.output(TRIG,False)

    time.sleep(0.0001)

    GPIO.output(TRIG,True)

    time.sleep(0.0001)

    GPIO.output(TRIG,False)

    pulse_start = time.time()
    timeout = pulse_start + maxTime
    while GPIO.input(ECHO) == 0 and pulse_start < timeout:
        pulse_start = time.time()
    pulse_end = time.time()
    timeout = pulse_end + maxTime
    while GPIO.input(ECHO) == 1 and pulse_end < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17000/2.54 #inch
    distance = round(distance, 2)

    print(distance)
    dis = int(distance)

 #### IMAGE PROCESSING

    ret, frame = cap.read()
    if frame is None:
        break
    img = frame.copy()
# Resize Image
    img = cv2.resize(frame,(128,128))

# Convert BGR TO RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize the image and format it
    img = np.array([img]).astype('float64') / 255.0

# Input Image to the network
    net.setInput(img)

# Perform a Forward pass
    Out = net.forward()
    
# Get the top predicted index
    index = np.argmax(Out[0])

# Get the probability of the class.
    prob = np.max(Out[0])
    
    label =  label_names[index]
    str1 = str(label)
    text2 = "Predicted: {} {:.2f}%".format(label, prob*100)
    text = "Predicted: NO DETECT"
    if prob <= 0.75:
        cv2.putText(frame, text, (5, 4*26),  cv2.FONT_HERSHEY_COMPLEX, 1, (50, 20, 255), 2)
        cv2.imshow('Sample video',frame)
    

## VOICE ACTIVATED

####
    
    else:
        cv2.putText(frame, text2, (5, 4*26),  cv2.FONT_HERSHEY_COMPLEX, 1, (50, 20, 255), 2)
        vidcap= cv2.imshow('Sample video',frame)
        
    
    if prob >= 0.93 and prob <= 0.97 and label == 'Laptop' and dis >= 1 and dis <=100:
        alarm = threading.Thread(target=voice_alarm3, args=(alarm_sound,))
        alarm.start()
    
        
    elif prob >= 0.93 and prob <= 0.97 and label == 'Laptop' and dis >= 101 and dis <=200:
        alarm = threading.Thread(target=voice_alarm3, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)    

        
    elif prob >= 0.93 and prob <= 0.97 and label == 'Laptop' and dis >= 201:
    
        alarm = threading.Thread(target=voice_alarm3, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)

    elif prob >= 0.93 and prob <= 0.97 and label == 'Mouse' and dis >= 1 and dis <=100:
    
        alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)
        
    elif prob >= 0.93 and prob <= 0.97 and label == 'Mouse' and dis >= 101 and dis <=200:
    
        alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)
        
    elif prob >= 0.93 and prob <= 0.97 and label == 'Mouse' and dis >= 201:
    
        alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)
        
    elif prob >= 0.90 and prob <= 0.97 and label == 'Keyboard' and dis >= 1 and dis <=100:
    
        alarm = threading.Thread(target=voice_alarm2, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)
        
    elif prob >= 0.90 and prob <= 0.97 and label == 'Keyboard' and dis >= 101 and dis <=200:
    
        alarm = threading.Thread(target=voice_alarm2, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)
        
    elif prob >= 0.90 and prob <= 0.97 and label == 'Keyboard' and dis >= 201:
    
        alarm = threading.Thread(target=voice_alarm2, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.001)

#####
#
    
#         
    if distance >= 5 and distance <= 10:
        alarm = threading.Thread(target=voice_alarm4, args=(alarm_sound,))
        alarm.start()

    elif distance <= 4:
        alarm = threading.Thread(target=voice_alarm44, args=(alarm_sound,))
        alarm.start()
        
        time.sleep(0.1)
        
#

#      
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        alarm = threading.Thread(target=voice_alarm5, args=(alarm_sound,))
        alarm.start()
        time.sleep(0.1)
        break
    
GPIO.cleanup()

cv2.destroyAllWindows()

#
#         