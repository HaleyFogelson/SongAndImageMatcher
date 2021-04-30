import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def getModel():
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model


def train_model(model):
    #numbers to adjust during training
    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    
    # Define data generators
    train_dir = 'data/train'
    val_dir = 'data/test'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')
    return model




def predictLiveStream(model,emotion_dict):
    cap = cv2.VideoCapture(0)
    print("Type q when you are at the emotion you want the playlist for")
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Since you're facial emotion is %s you will get a %s song",prediction, prediction)
            return prediction
            break
    cap.release()
    cv2.destroyAllWindows()



def predictUploadedPhoto(model,emotion_dict, img_path):
    frame = cv2.imread(img_path)
            #ret, frame = cap.read()
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    except:
        print("the image can not detect faces for path:", img_path)

    emotions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        emotions.append(emotion_dict[maxindex])
        # print(emotion_dict[maxindex])
    return emotions
    #cv2.imshow(cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))


def runAlgorithm(mode,inputType, img_path=""):
    #create the model
    model = getModel()

    # If you want to train the same model or try other models, go for this
    if mode == "train":
        model = train_model(model)

    # emotions will be displayed 
    else:
        model.load_weights('model.h5')

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

       #upload an image
        if inputType == "upload":
            return predictUploadedPhoto(model,emotion_dict, img_path)
        else:
             # start the webcam feed to get emotion off live webcab
             return predictLiveStream(model,emotion_dict)

def labelImages():
    cvsData=[]

    for filename in os.listdir('../photosForAI'):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            print("running algorthm for file",filename)
            img_path = './photosForAI/' + filename
            result = runAlgorithm("display","upload",img_path)
            emotions=" "
            if result:
                result.sort()
                for emotion in result:
                    emotions = emotions + emotion + ", " 
                #print("just appended: ", emotions)
            cvsData.append([filename,emotions])
    df = pd.DataFrame(np.asarray(cvsData), columns=['file name','emotions']) 
    df.to_csv('result.csv',index=False)        

def commandLineAlgorthmsMain():
    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help="Image Path")
    ap.add_argument("--mode",help="train or display")
    ap.add_argument('-it','--inputType',help="live or upload")
    args = vars(ap.parse_args())
    img_path = args['image']
    mode = args['mode']
    inputType = args['inputType']
    return (runAlgorithm(mode,inputType, img_path))
       

# if __name__ == "__main__":
#     #commandLineAlgorthmsMain()
#     print(commandLineAlgorthmsMain())
#     #labelImages()
    
    


