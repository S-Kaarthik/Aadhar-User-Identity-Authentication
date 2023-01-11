import tkinter
from tkinter import *
import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import tkinter.ttk as ttk
import tkinter.font as font
import numpy as np
from PIL import Image, ImageTk
import csv
import json
import pandas as pd


#import shutil
#import pandas as pd
#import datetime
#import time



window=tk.Tk()
window.title("Registration Page")
window.geometry('1366x768')

####Background image
image1 = Image.open("face.png")
test = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(window,image=test)
label1.image = test
label1.pack(fill=BOTH,expand=YES)



#####Loading Data from txt file
def loading_data():
    file=open('data.txt','r',encoding='utf-8')
    data=json.load(file)
    file.close()
    return data
##Saving Data to .txt file
def saving_data(data):
    file=open('data.txt','w',encoding='utf-8')
    json.dump(data,file,ensure_ascii=False)
    file.close()

##saving data in a variable
#data=dict()
data=loading_data()
print(data)







#HEADINGS

login_label= tk.Label(window, text="REGISTER YOURSELF!" ,bg="#000f1e"  ,fg="white"  ,width=25 ,height=1,font=('times', 30, 'bold')) 
login_label.place(x=340, y=10)



msg_label=tk.Label(window,text="AUTHENTICATION MESSAGE: ",bg='#001020' ,fg='white',width=26,height=1,font=('times',15,'bold'))
msg_label.place(x=150,y=500)
message = tk.Label(window,text="",fg="black", width=30, height=1, font=('times',15,'bold'))
message.place(x=480,y=500)



#Login FIEDLS





#Register Fields

lb3 = tk.Label(window, text="UNIQUE ID :",width=10,height=1  ,fg="#00192d" ,font=('times', 15, ' bold ') ) 
lb3.place(x=150, y=250)

txt3 = tk.Entry(window,width=25  ,bg="#003a62" ,show='*',fg="white",font=('times', 15, ' bold '))
txt3.place(x=320, y=250)

lbl4= tk.Label(window, text="Enter Name :",width=11  ,height=1  ,fg="white"  ,bg="#001020" ,font=('times', 15, ' bold ') ) 
lbl4.place(x=150, y=300)

txt4 = tk.Entry(window,width=25  ,fg="#00192d",font=('times', 15, ' bold '))
txt4.place(x=320, y=300)


#Main


    #res = ""
    #txt.configure(text= res)
    #txt2.configure(text=res)
def reg_clear():
    txt3.delete(0,'end')
    txt4.delete(0,'end')
    #res = ""
    #txt3.configure(text= res)
    #txt4.configure(text=res)



################## Login_Functions   ######################

#When Submit button is clicked We have to Track the Person from our data through web cam
    
def TrackImages(UserId):
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("Details\Details.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX          
    run_count=0;run=True
    while run:
        
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id, conf)
            if(conf < 50):
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                if (str(Id)==UserId):
                    message.configure(text="Face Recognised Successfully")
                    run=False
            else:
                Id='Unknown'                
                tt=str(Id)            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        run_count += 1    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q') or run_count==150):
            message.configure(text="Unable to Recognise Face")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    

################## Register_Functions   ######################

def TakeImages():        
    Id=(txt3.get())
    name=(txt4.get())
    ret=0
    if (Id not in data):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +'.'+Id+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>100:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('Details\Details.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
        ret=1
    else:
        res = "User name Already Exists! Try another one!!"
        message.configure(text= res)
    return ret

# Training Images

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    #print(faces,np.array(Id))
    #res = "Image Trained"#+",".join(str(f) for f in Id)
    res="Registration Successful"
    message.configure(text= res)
    return True
    

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image path  s and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids


def reg_submit():
    Userid=txt3.get()
    if Userid.isdigit():
        if TakeImages()==1:
            if TrainImages():
                data[txt3.get()] = txt4.get()
                saving_data(data)
            else:
                pass
    
    else:
        message.configure(text="User Id Should contain number only!!!")
    reg_clear()
    print(data)



#Register Actions
submit2 = tk.Button(window, text="Submit",fg="white",command= reg_submit, bg="#001020"  ,width=25  ,height=1 ,relief="groove" ,activebackground = "white" ,font=('times', 10, ' bold '))
submit2.place(x=250, y=350)

clearButton2 = tk.Button(window, text="Clear",fg="#001020", command=reg_clear,bg="white"  ,width=25  ,height=1, relief='groove', activebackground = "blue" ,activeforeground='white',font=('times', 10, ' bold '))
clearButton2.place(x=450, y=350)


#final Actions
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="red"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1050, y=550)
window.mainloop()
