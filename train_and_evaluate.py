import math

import torch
import os
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
import torchvision
from matplotlib import patches
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.transforms import RandomRotation
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import precision_recall_fscore_support

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU를 사용합니다:', torch.cuda.get_device_name(0))
else:
    print('사용 가능한 GPU가 없습니다. CPU를 사용합니다.')
    device = torch.device("cpu")

# 객체의 bounding box 생성
def generate_box(obj):
    xmin = float(obj.find('x_min').text)
    ymin = float(obj.find('y_min').text)
    xmax = float(obj.find('x_max').text)
    ymax = float(obj.find('y_max').text)
    return [xmin, ymin, xmax, ymax]

# 객체의 라벨 생성
adjust_label = 1
default_label = 0
def generate_label(obj):
    if obj.find('state').text == "1":
        return 0 + adjust_label
    elif obj.find('state').text == "2":
        return 1 + adjust_label
    elif obj.find('state').text == "3":
        return 2 + adjust_label
    elif obj.find('state').text == "4":
        return 3 + adjust_label
    elif obj.find('state').text == "5":
        return 4 + adjust_label
    return default_label  # 기본 라벨 값 반환

# XML 파일로부터 타겟 생성
def generate_target(file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("bndbox")

        num_objs = len(objects)

        if num_objs == 0:
            print(f"{file}에 유효한 주석이 없습니다.")
            # 주석이 없는 경우 빈 바운딩 박스와 기본 라벨 반환
            boxes = torch.as_tensor([], dtype=torch.float32)
            labels = torch.as_tensor([default_label], dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
            return target

        boxes = []
        labels = []

        for i in objects:
            box = generate_box(i)
            label = generate_label(i)

            # 유효한 바운딩 박스만 추가
            if box is not None:
                boxes.append(box)
                labels.append(label)

        if not boxes or not labels:
            print(f"{file}에 유효한 주석이 없습니다.")
            # 유효한 주석이 없는 경우 빈 바운딩 박스와 기본 라벨 반환
            boxes = torch.as_tensor([], dtype=torch.float32)
            labels = torch.as_tensor([default_label], dtype=torch.int64)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return target

# 계란 데이터셋 클래스 정의
class EggDataset(object):
    def __init__(self, transforms, path):
        self.transforms = transforms
        self.path = path
        self.imgs = list(sorted(os.listdir(os.path.join(self.path, '01.원천데이터\\TS_02. COLOR'))))
        self.num_classes = 6

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, '01.원천데이터\\TS_02. COLOR', file_image)
        label_path = os.path.join(self.path, '02.라벨링데이터\\TL_02. COLOR', file_label)
        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path)

        if target is None or 'boxes' not in target or 'labels' not in target or target['boxes'].numel() == 0:
            print(f"인덱스 {idx}는 모든 주석이 무시되었으므로 건너뜁니다.")
            return None, None  # 이미지와 타겟 모두 None으로 반환

        img = transforms.ToTensor()(img)
        target_boxes = target["boxes"]
        target_labels = target["labels"]

        target = [{"boxes": target_boxes, "labels": target_labels, "image_id": torch.tensor([idx])}]
        return img, target

    def __len__(self):
        return len(self.imgs)

# 데이터 변환 정의
data_transform = transforms.Compose([
    transforms.Lambda(lambda x: {
        'image': transforms.functional.to_tensor(x['image']),
        'boxes': x['boxes'],
        'labels': x['labels']
    } if x['boxes'].size(0) > 0 else {
        'image': torch.empty((0, 0, 0), dtype=torch.float32),
        'boxes': torch.empty((0, 4), dtype=torch.float32),
        'labels': torch.empty((0,), dtype=torch.int64),
    }),
    RandomRotation(degrees=(-45, 45), fill=(0,)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
])

# 모델 가중치 저장 디렉토리 생성
save_dir = 'path/to/save/weights'
os.makedirs(save_dir, exist_ok=True)

# 모델 가중치 저장 함수 정의
def save_model_weights(model, epoch, optimizer, loss, save_dir):
    save_path = os.path.join(save_dir, f'1.0, 1.5, 1.0, 1.0, 1.0, 1.0{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

    print(f'모델 가중치가 저장되었습니다: {save_path}')

# 데이터 로더 생성
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch)) if batch else list(batch)

dataset = EggDataset(data_transform, 'D:\\248.계란 데이터\\01.데이터\\Training')
validation_dataset = EggDataset(data_transform, 'D:\\248.계란 데이터\\01.데이터\\Validation')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

# 모델 인스턴스 생성 함수 정의
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 평가용 데이터셋에 대한 예측 및 평가
def evaluate_model(model, data_loader, device):
    all_predictions = []
    all_targets = []
    coordinate_threshold = 100
    with torch.no_grad():
        for imgs, targets_list in data_loader:
            if None in imgs or None in targets_list:
                continue

            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in target.items()} for targets in targets_list for target in targets]

            predictions = model(imgs)

            for img, target, prediction in zip(imgs, targets, predictions):
                img = to_pil_image(img)

                # NMS 적용
                keep_indices = apply_nms(prediction['boxes'], prediction['scores'], iou_threshold=0.7)

                # 예측된 바운딩 박스 중에서 라벨링 데이터와 일정 기준 이하로 차이 나는 것만 남기기
                keep_indices = filter_predictions_by_difference(keep_indices, prediction['boxes'], target['boxes'],
                                                                coordinate_threshold)

                prediction['boxes'] = prediction['boxes'][keep_indices]
                prediction['labels'] = prediction['labels'][keep_indices]
                prediction['scores'] = prediction['scores'][keep_indices]

                num_predictions = len(prediction['labels'])
                num_targets = len(target['labels'])

                # 부족한 부분을 0으로 채우기
                if num_predictions < num_targets:
                    prediction['labels'] = torch.cat(
                        [prediction['labels'], torch.zeros(num_targets - num_predictions).to(device)])
                elif num_predictions > num_targets:
                    target['labels'] = torch.cat(
                        [target['labels'], torch.zeros(num_predictions - num_targets).to(device)])

                all_predictions.extend(prediction['labels'].cpu().numpy())
                all_targets.extend(target['labels'].cpu().numpy())

                # plot_image_with_labels(img, target, prediction)

    return all_predictions, all_targets


def plot_image_with_labels(img, annotation, predictions):
    img_np = np.array(img)
    annotation_cpu = {k: v.cpu().numpy() for k, v in annotation.items()}
    predictions_cpu = {k: v.cpu().numpy() for k, v in predictions.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title('Data')

    for idx in range(len(annotation_cpu["boxes"])):
        xmin, ymin, xmax, ymax = annotation_cpu["boxes"][idx]
        label_value = annotation_cpu['labels'][idx]
        if annotation_cpu['labels'][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g', facecolor='none')
        elif annotation_cpu['labels'][idx] == 2:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none')
        elif annotation_cpu['labels'][idx] <= 5:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange', facecolor='none')

        axes[0].add_patch(rect)
        axes[0].text(xmin, ymin, f'Label: {label_value}', color='white', fontsize=8, verticalalignment='top')

    axes[1].imshow(img)
    axes[1].set_title('Predictions')

    # NMS가 적용된 예측 결과만 사용
    keep_indices = apply_nms(predictions_cpu["boxes"], predictions_cpu["scores"], iou_threshold=0.6)
    pred_boxes = predictions_cpu["boxes"][keep_indices]
    pred_labels = predictions_cpu["labels"][keep_indices]
    pred_scores = predictions_cpu["scores"][keep_indices]

    if len(pred_boxes) == 0:
        # NMS 후에 예측이 없는 경우
        print("No predictions after NMS.")
        return

    if len(pred_boxes.shape) == 1:
        # NMS 후에 하나의 예측만 있는 경우 (배열이 아닌 값)
        pred_boxes = pred_boxes.reshape(1, -1)
        pred_labels = np.array([pred_labels])
        pred_scores = np.array([pred_scores])

    for idx in range(len(pred_boxes)):
        xmin, ymin, xmax, ymax = pred_boxes[idx]

        if pred_labels[idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g',
                                     facecolor='none')
        elif pred_labels[idx] == 2:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')
        elif pred_labels[idx] <= 5:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')

        axes[1].add_patch(rect)

        pred_label = int(pred_labels[idx])
        pred_score = round(float(pred_scores[idx]), 2)

        axes[1].text(xmin, ymin, f'Label: {pred_label}\nScore: {pred_score}', color='white', fontsize=8,
                     verticalalignment='top')

    img_pil = transforms.ToPILImage()(img_np)
    plt.imshow(img_pil)
    plt.show()

def apply_nms(boxes, scores, iou_threshold, score_threshold=0.1):
    # NumPy 배열을 torch.Tensor로 변환
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)

    # score_threshold보다 높은 점수를 가진 예측에 대해서만 NMS 적용
    high_score_indices = torch.where(scores >= score_threshold)[0]
    boxes = boxes[high_score_indices]
    scores = scores[high_score_indices]

    # torchvision의 NMS 함수 사용
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)

    # score_threshold 이하의 점수를 가진 예측은 제외
    keep = keep[scores[keep] >= score_threshold]

    # 정수로 변환
    keep = keep.long()

    return keep



# 두 바운딩 박스 간의 차이를 계산하는 함수
def calculate_box_difference(box1, box2):
    return torch.abs(box1 - box2)

# 예측된 바운딩 박스 중에서 라벨링 데이터와 일정 기준 이하로 차이 나는 것만 남기기
def filter_predictions_by_difference(indices, predicted_boxes, target_boxes, threshold):
    filtered_indices = []

    for idx in indices:
        # 예측된 바운딩 박스
        predicted_box = predicted_boxes[idx]

        # 모든 타겟 바운딩 박스와의 차이 확인
        differences = [torch.all(calculate_box_difference(predicted_box, target_box) <= threshold) for target_box in
                       target_boxes]

        # 어떤 타겟과도 차이가 threshold 이하인 경우에만 해당 예측을 유효한 예측으로 간주
        if any(differences):
            filtered_indices.append(idx.item())  # .item()을 추가하여 long tensor를 int로 변환

    return filtered_indices

# 모델 인스턴스 생성
model = get_model_instance_segmentation(6)

# CUDA 사용 가능 여부에 따라 모델을 적절한 디바이스로 이동
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 모델 가중치 파일 경로
model_weights_path = 'path/to/save/weights/1.0, 1.5, 1.0, 1.0, 1.0, 1.02.pth'

# 모델 로딩
model.load_state_dict(torch.load(model_weights_path)['model_state_dict'])
model.eval()

# 데이터 로더 생성
validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
all_predictions, all_targets = evaluate_model(model, validation_data_loader, device)

# 두 변수의 길이 확인
len_all_targets = len(all_targets)
len_all_predictions = len(all_predictions)


print(f"Length of all_targets: {len_all_targets}")
print(f"Length of all_predictions: {len_all_predictions}")


# 정밀도, 재현율, F1 점수 계산 및 출력
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

'''
# 손실 함수 및 옵티마이저 정의
class_weights = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
num_epochs = 1
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# 학습 
print('----------------------train start--------------------------')
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    epoch_loss = 0

    for imgs, targets_list in data_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in target.items()} for targets in targets_list for target in targets]

        optimizer.zero_grad()

        # 모델에 이미지와 타겟 모두 전달
        loss_dict = model(imgs, targets)
        losses_list = [loss for loss in loss_dict.values()]
        losses = sum(losses_list)
        # 역전파 및 최적화
        losses.backward()
        optimizer.step()

        epoch_loss += sum(loss.item() for loss in losses_list)

    print(f'Training Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Time: {time.time() - start}')

# 모델 가중치 저장
save_model_weights(model, num_epochs, optimizer, epoch_loss, save_dir)
'''
