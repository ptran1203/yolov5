import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_boxes(label):
    if 'opacity' not in label:
        return []
    
    label = label.replace('opacity 1', '').replace('  ', ' ').strip()
    boxes = []
    coors = label.split(' ')
    num_boxes = len(coors) // 4
    for i in range(num_boxes):
        start = i * 4
        end = start + 4
        boxes.append([float(x) for x in coors[start:end]])
        
    return boxes


def to_yolo_boxes(bboxes, img_w, img_h):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0] # xmax - xmin
        h = bbox[3] - bbox[1] # ymax - ymin
        xc = bbox[0] + int(np.round(w/2)) # xmin + width/2
        yc = bbox[1] + int(np.round(h/2)) # ymin + height/2
        # x_center y_center width height
        yolo_boxes.append([xc / img_w, yc / img_h, w / img_w, h / img_h])
    
    return yolo_boxes

def scale_boxes(boxes, dim0, dim1, img_w, img_h, yolo_format=True):
    scale_x = img_w / dim0
    scale_y = img_h / dim1
    scaled_boxes = []
    for box in boxes:
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
        # xmin, ymin, xmax, ymax
        scaled_boxes.append([x1, y1, x2, y2])

    if yolo_format:
        scaled_boxes = to_yolo_boxes(scaled_boxes, img_w, img_h)
        
    return scaled_boxes

def visualize(image, boxes, scores=None, figsize=(6, 6), linewidth=1):
    '''
    Expect boxes to have format: x1, y1, x2, y2
    '''
    if scores is None:
        scores = [0] * len(boxes)

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(boxes)):
        box, score = boxes[i], scores[i]
        color = [0, 0, 1]
        text = "{}: {:.2f}".format('Opacity', score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(x1, y1, text,
            bbox={"facecolor": color, "alpha": 0.5},
            clip_box=ax.clipbox,
            clip_on=True,
        )

def random_visualize(train_df, img_dir='/content/dataset/images',
                     img_w=640, img_h=640):
    def v(item):
        return item.values[0]

    row = train_df[train_df.label_y == 1.0].sample(n=1)
    boxes = scale_boxes(
        get_boxes(v(row.label)),
        v(row.dim0), v(row.dim1),
        img_w, img_h, yolo_format=False)

    img_path = os.path.join(img_dir, f'{v(row.id)}.jpg')
    img = cv2.imread(img_path)
    assert img is not None, f'Could not read image from {img_path}'
    visualize(img, boxes)