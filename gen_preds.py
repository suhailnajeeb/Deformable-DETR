"""
Usage: python gen_preds.py --with_box_refine --two_stage --num_classes 5
"""


import os
import json
import random
from PIL import Image
import argparse
from test_utils import *
from models import build_model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Deformable DETR prediction script', parents=[get_args_parser()])
args = parser.parse_args()

CLASSES = ["D00", "D10", "D20", "D40"]

#random.seed()

data_root = '/data/gpfs/projects/punim1800/RDD-2022/holdout/coco_holdout/'
set_name = 'test2017'
model_dir = 'out/iter_2s/'

annotations_path = os.path.join(
    data_root, 'annotations/instances_{}.json'.format(set_name))

thresh = 0.8
sample = 1

subplots = True
show = False

output_path = 'preds/'

file_name_list = [
    'Japan_009600.jpg', # no crack present in ground truth, but found by model
    'Norway_001506.jpg', # no crack present in ground truth, but found by model
    'Czech_001028.jpg', # regular, better detection than ground truth
    'Japan_001183.jpg', # regular
    'United_States_001598.jpg', # regular
    'United_States_001609.jpg', # regular
]

#sample = random.choice(list(range(len(file_name_list))))
sample = None

if sample is not None:
    file_name = file_name_list[sample]
else:
    file_name = None

# load annotations

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

images = annotations['images']
image = random.choice(images)
if not file_name:
    file_name = image['file_name']
    image_id = image['id']
    print('file_name: ', file_name)
else:
    image_id = get_image_id(file_name)

# get annotations for the image_id
annotations = [a for a in annotations['annotations'] if a['image_id'] == image_id]

# load image and plot annotations in image
image_path = os.path.join(data_root, set_name, file_name)

im = Image.open(image_path)

# load model

#model_path = 'outputs/checkpoint.pth'
model_path = model_dir + 'checkpoint0084.pth'
model, criterion, postprocessors = load_model_from_ckp(model_path, args)
model = model.cuda()

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)
img = img.cuda()

# propagate through the model
outputs = model(img)

# keep only predictions with thresh+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
probas = probas
keep = probas.max(-1).values > thresh
#keep = keep.cpu()

# convert boxes from [0; 1] to image scales
#bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# convert all boxes to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, :], im.size)
bboxes_scaled = bboxes_scaled.cpu().detach().numpy()

# plot predictions

#fig = plot_gt_preds(im, annotations, probas, bboxes_scaled)

# save figure

#fig.savefig(model_dir + '{}.png'.format(file_name))

# Plot the predicted bounding boxes along with the probability of the predicted class
file_name = file_name.split('.')[0]

import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = COLORS * 100

#ax = plt.gca()

if subplots:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(im)
    ax[0].set_title('Ground Truth')

    for a in annotations:
        x, y, w, h = a['bbox']
        #print(a['category_id'])
        #class_name = CLASS_NAMES[a['category_id']]
        class_code = CLASSES[a['category_id']]
        color = colors[a['category_id']]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(x, y, class_code, bbox={'facecolor': color, 'alpha': 0.5})
    ax[0].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
    #plt.axis('off')
    # turn off axis
    ax[0].axis('off')

    ax[1].imshow(im)
    ax[1].set_title('Predictions (Def-DETR)')

    for i in range(len(keep)):
        if keep[i]:
            bbox = bboxes_scaled[i]
            prob = probas[i]
            class_id = prob.argmax()
            class_name = CLASSES[class_id]
            prob = prob[class_id]*100
            color = colors[class_id]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
            ax[1].add_patch(rect)
            ax[1].text(bbox[0], bbox[1], '{}: {:.0f} %'.format(class_name, prob), color=color, bbox=dict(facecolor='white', alpha=0.5))

    # plot legend including CLASS_NAMES

    #ax[1].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
    plt.axis('off')
    plt.savefig(output_path + '{}_dfdetr_subplot.png'.format(file_name))
    if show:
        plt.show()
else:
    # plot ground truth

    plt.imshow(im)
    plt.title('Ground Truth')

    for a in annotations:
        x, y, w, h = a['bbox']
        #print(a['category_id'])
        #class_name = CLASS_NAMES[a['category_id']]
        class_code = CLASSES[a['category_id']]
        color = colors[a['category_id']]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x, y, class_code, bbox={'facecolor': color, 'alpha': 0.5})
    plt.legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
    plt.axis('off')
    plt.savefig(output_path + '{}_gt.png'.format(file_name))
    if show:
        plt.show()

    # plot predictions

    plt.imshow(im)
    plt.title('Predictions (Def-DETR)')

    for i in range(len(keep)):
        if keep[i]:
            bbox = bboxes_scaled[i]
            prob = probas[i]
            class_id = prob.argmax()
            class_name = CLASSES[class_id]
            prob = prob[class_id]*100
            color = colors[class_id]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1], '{}: {:.0f} %'.format(class_name, prob), color=color, bbox=dict(facecolor='white', alpha=0.5))

    # plot legend including CLASS_NAMES

    #ax[1].legend(handles=[patches.Patch(color=color, label=class_name) for class_name, color in zip(CLASS_NAMES.values(), colors)])    
    plt.axis('off')
    plt.savefig(output_path + '{}_dfdetr.png'.format(file_name))
    if show:
        plt.show()