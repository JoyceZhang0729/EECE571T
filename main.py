from model import Model
from tensorflow import keras
import cv2 as cv

model = Model()
names = ["age", "gender", "ethnicity"]
classifier = cv.CascadeClassifier('pretrained_model/haarcascade_frontalface_default.xml')



def captureImage():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # perform face detection
        bboxes = classifier.detectMultiScale(gray, 1.05, 8)
        # print bounding box for each detected face
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            cv.rectangle(gray, (x, y), (x2, y2), (255, 0, 0), 2)
            crop_img = gray[y:y2, x:x2]
            text = model.predict(crop_img)
            frame = cv.putText(frame, text, (x-400, y), font, fontScale, color, thickness, cv.LINE_AA)
        # Create a basic model instance
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    for name in names:
        if name == "age":
            model.age_model = keras.models.load_model(f'pretrained_model/{name}_model')
        elif name == "gender":
            model.gender_model = keras.models.load_model(f'pretrained_model/{name}_model')
        else:
            model.ethnicity_model = keras.models.load_model(f'pretrained_model/{name}_model')
    captureImage()