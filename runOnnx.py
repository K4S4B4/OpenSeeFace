import onnxruntime
import cv2
import numpy as np

mean = (0.485 + 0.456 + 0.406) / 3 # np.float32(np.array([0.485, 0.456, 0.406]))
std = (0.229 + 0.224 + 0.225) / 3 # np.float32(np.array([0.229, 0.224, 0.225]))
mean = mean / std
std = std * 255.0

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad

onnx_file_name = 'models/lm_model2_opt.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name


WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(2)
if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)
    img3 = cv2.resize(img1, (224,224))

    ##################################################

    im = np.float32(img3[:, :,::-1]) #BGR to RGB
    im = im * std + mean
    im = np.expand_dims(im, 0)

    ##################################################

    #im = np.expand_dims(im, axis=0).astype(np.uint8)

    im = np.transpose(im, (0,3,1,2))

    ort_inputs = {input_name: im}

    ort_outs = ort_session.run(None, ort_inputs)

    #frame = cv2.rectangle(img3, (30,30), (162,162), (255, 255, 255), -1)
    #img3 = cv2.resize(img3, (960,960))

    for i in range(len(ort_outs[0])):
        landmark, flag = ort_outs[0][i], ort_outs[1][i]
        if flag>.5:
            draw_landmarks(img3, landmark[:,:2], FACE_CONNECTIONS, size=1)

    cv2.imshow(WINDOW, img3)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

