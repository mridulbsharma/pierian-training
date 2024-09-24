import numpy as np
import cv2
import pickle

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def get_class_name(class_no):
    classes = {
        0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
        3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
        6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
        9:'No passing', 10:'No passing for vehicles over 3.5 metric tons', 
        11:'Right-of-way at the next intersection', 12:'Priority road', 13:'Yield', 
        14:'Stop', 15:'No vehicles', 16:'Vehicles over 3.5 metric tons prohibited', 
        17:'No entry', 18:'General caution', 19:'Dangerous curve to the left', 
        20:'Dangerous curve to the right', 21:'Double curve', 22:'Bumpy road', 
        23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 
        26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
        29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
        32:'End of all speed and passing limits', 33:'Turn right ahead', 
        34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 
        37:'Go straight or left', 38:'Keep right', 39:'Keep left', 
        40:'Roundabout mandatory', 41:'End of no passing', 
        42:'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes[class_no]

def main():
    # Load the pickled model
    with open("traffic_sign_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Setup the video capture
    cap = cv2.VideoCapture(0)
    frameWidth = 640
    frameHeight = 480
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 180)  # Brightness

    while True:
        success, img_original = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        img = np.asarray(img_original)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        # Predict
        predictions = model.predict(img, verbose=0)
        class_index = np.argmax(predictions)
        probability_value = np.amax(predictions)

        if probability_value > 0.75:
            class_name = get_class_name(class_index)
            cv2.putText(img_original, f"Class: {class_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img_original, f"Probability: {probability_value:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Result", img_original)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()