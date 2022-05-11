import torch
import utils
import time
import detect
import ssl
import package_data

ssl._create_default_https_context = ssl._create_unverified_context
# --Parameters:
imsize = 416 # Image dimensions
thresh = 0.3 # Confidence threshold for bounding
source = './run_detections/full_dataset/' # Path to image to run detections on
weight = './final-weights.pt' # Pretrained weights for deteections
data = 'data/custom_data.yaml'
outpath = 'runs/detect'
filename = 'test.jpg'

# --Loading model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
model.conf = thresh
model.cpu()  # CPU
# model.cuda()  # GPU

# Run Detections
start = time.time()
prediction = package_data.run_detections(model=model,
                                         image_filename=filename,
                                         image_in_folder=source,
                                         model_train_image_size=imsize,
                                         image_out_folder=outpath,
                                         pred_type='all',
                                         save_images=True)
print("Image inference completed in {}ms".format((time.time() - start)*1000))
print(prediction)

#detect.run(
#        weights=weight,  # model.pt path(s)
#        source=source,  # file/dir/URL/glob, 0 for webcam
#        data=data,  # dataset.yaml path
#        imgsz=(imsize, imsize),  # inference size (height, width)
#        conf_thres=thresh,  # confidence threshold
#        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#        name='exp',  # save results to project/name
#        line_thickness=1,  # bounding box thickness (pixels)
#)
#print("inference completed in {}ms".format((time.time() - start)*1000))