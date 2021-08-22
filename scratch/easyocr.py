import easyocr
import datetime
import time
x = 1 # displays the frame rate every 1 second
counter = 0
start_time = time.time()
reader = easyocr.Reader(['en'], gpu=True) # need to run only once to load model into memory
while True:
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    #print (datetime.datetime.now())
    result = reader.readtext('pet_black3.jpg')
    
    print (result)




