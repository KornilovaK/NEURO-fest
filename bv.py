import cv2
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
import numpy as np
from model import Model


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    model = Model.model
    actions = Model.actions
    model.load_weights('new.h5')
    sequence = []
    sentence = []
    predictions = set()
    threshold = 0.8
    delete = ['До свидания', 'Привет']
    add = ['дела', 'зовут']

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([lh, rh])

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

    def run(self):

        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:
                ret, frame = cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]

                    if results.right_hand_landmarks:
                        result_action = []
                        result_action.append(self.actions[np.argmax(res)])
                        self.predictions.update(result_action)

                        if result_action[0] in self.add:
                            self.predictions.update(['?'])

                        cv2.putText(image, ' '.join(self.predictions), (3, 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                        if result_action[0] in self.delete or result_action[0] in self.add:
                            self.predictions.clear()


                if ret:
                    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.left = 200
        self.top = 200
        self.width = 800
        self.height = 400
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.title = 'Автоматический сурдопереводчик'
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1200, 800)
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())