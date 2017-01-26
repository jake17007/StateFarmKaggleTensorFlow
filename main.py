from preprocess import preprocess_for_imageflow
from preprocess import preprocess_for_csvreader

preprocess_for_imageflow('../imgsTest/train', 'imagesOutfile.csv', '../img_labels.csv', 'labelsOutfile.csv')
#preprocess_for_csvreader('../imgsTest/train', '../img_labels.csv', 'outfile.csv'))
