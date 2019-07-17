import cv2

def main():
     print("hello world!")
     cap = cv2.VideoCapture(0)
     fourcc = cv2.VideoWriter_fourcc(*"DIVX")
     out = cv2.VideoWriter('output.avi', fourcc, 20, (640,480))
     while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

if __name__ == "__main__":
    main()