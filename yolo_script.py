import torch
import utils
import time
import detect
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# --Parameters:
imsize = 416 # Image dimensions
thresh = 0.3 # Confidence threshold for bounding
source = './run_detections/' # Path to image to run detections on
weight = './final-weights.pt' # Pretrained weights for deteections
data = 'data/custom_data.yaml'
outpath = 'runs/detect'
# --img 416 
# --source ../accuracy_test/5-3-2022-test/tdata 352.jpg 
# --weights ../final-best.pt 
# --conf-thres 0.51


# --Loading model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
model.conf = thresh
model.cpu()  # CPU
# model.cuda()  # GPU

# Run Detections
start = time.time()
detect.run(
        weights=weight,  # model.pt path(s)
        source=source,  # file/dir/URL/glob, 0 for webcam
        data=data,  # dataset.yaml path
        imgsz=(imsize, imsize),  # inference size (height, width)
        conf_thres=thresh,  # confidence threshold
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=outpath,  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
)
print("inference completed in {}ms".format((time.time() - start)*1000))