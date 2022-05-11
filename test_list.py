import glob
source = './run_detections/full_dataset/' # Path to image to run detections on

imgs = [f for f in glob.glob(source+'*.jpg')]
imgs.append([f for f in glob.glob(source+'*.png')])

print(imgs)