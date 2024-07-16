import cv2 as cv

class FancyDrawn:
    def __init__(self, frame):
        self.__frame = frame

    def draw(self, bbox, l=30, t=10):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        #Rectangle
        cv.rectangle(self.__frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)# Draw the bounding box
        # Top Left
        cv.line(self.__frame, (x, y), (x + l, y), (0, 255, 0), t)
        cv.line(self.__frame, (x, y), (x, y + l), (0, 255, 0), t)
        # Top Right
        cv.line(self.__frame, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv.line(self.__frame, (x1, y), (x1, y + l), (0, 255, 0), t)
        # Bottom Left
        cv.line(self.__frame, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv.line(self.__frame, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # Bottom Right
        cv.line(self.__frame, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv.line(self.__frame, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        return self.__frame