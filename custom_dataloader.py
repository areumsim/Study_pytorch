from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
import urllib.request
import zipfile
import glob
import os
import random
from PIL import Image, UnidentifiedImageError, ImageFile

################################################################
##### 데이터 준비 #####

SEED = 123

# 데이터셋을 다운로드 합니다.
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
urllib.request.urlretrieve(url, "cats_and_dogs.zip")

#  다운로드 후 data 폴더에 압축을 해제 합니다.
local_zip = "cats_and_dogs.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./data/")
zip_ref.close()

# 잘린 이미지 Load 시 경고 출력 안함
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 이미지 Validation을 수행하고 Validate 여부를 return 합니다.
def validate_image(filepath):
    try:
        # PIL.Image로 이미지 데이터를 로드하려고 시도합니다.
        img = Image.open(filepath).convert("RGB")
        img.load()
    except UnidentifiedImageError:  # corrupt 된 이미지는 해당 에러를 출력합니다.
        print(f"Corrupted Image is found at: {filepath}")
        return False
    except (IOError, OSError):  # Truncated (잘린) 이미지에 대한 에러를 출력합니다.
        print(f"Truncated Image is found at: {filepath}")
        return False
    else:
        return True


# image 데이터셋 root 폴더
root = "./data/PetImages"

dirs = os.listdir(root)

for dir_ in dirs:
    folder_path = os.path.join(root, dir_)
    files = os.listdir(folder_path)

    images = [os.path.join(folder_path, f) for f in files]
    for img in images:
        valid = validate_image(img)
        if not valid:
            # corrupted 된 이미지 제거
            os.remove(img)
            # TODO 몇개 지웠는지 cnt - code 추가
folders = glob.glob("./data/PetImages/*")
print(folders)

# Train / Test 셋 분할
# image 데이터셋 root 폴더
# train: test ratio. 0.2로 설정시 test set의 비율은 20%로 설정
test_size = 0.2

# train / test 셋의 파일을 나눕니다.
train_images = []
test_images = []

for folder in folders:
    label = os.path.basename(folder)
    files = sorted(glob.glob(folder + "/*"))

    # 각 Label별 이미지 데이터셋 셔플
    random.seed(SEED)
    random.shuffle(files)

    idx = int(len(files) * test_size)
    train = files[:-idx]
    test = files[-idx:]

    train_images.extend(train)
    test_images.extend(test)

# train, test 전체 이미지 셔플
random.shuffle(train_images)
random.shuffle(test_images)

# Class to Index 생성. {'Dog': 0, 'Cat': 1}
class_to_idx = {os.path.basename(f): idx for idx, f in enumerate(folders)}

# Label 생성
train_labels = [f.split("/")[-2] for f in train_images]
test_labels = [f.split("/")[-2] for f in test_images]

print("===" * 10)
print(f"train images: {len(train_images)}")
print(f"train labels: {len(train_labels)}")
print(f"test images: {len(test_images)}")
print(f"test labels: {len(test_labels)}")


################################################################
#####  Dataset, loader 선언 ###


# Dataset을 상속받아 Customdataset 구성
class CustomImageDataset(Dataset):
    def __init__(self, files, labels, class_to_idx, transform):
        super(CustomImageDataset, self).__init__()
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # file 경로
        file = self.files[idx]
        # PIL.Image로 이미지 로드
        img = Image.open(file).convert("RGB")
        # transform 적용
        img = self.transform(img)
        # label 생성
        lbl = self.class_to_idx[self.labels[idx]]
        # image, label return
        return img, lbl


train_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256)
        ),  # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
        transforms.CenterCrop((224, 224)),  # 중앙 Crop
        transforms.RandomHorizontalFlip(0.5),  # 50% 확률로 Horizontal Flip
        transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
        transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ]
)

# train, test 데이터셋 생성
train_dataset = CustomImageDataset(
    train_images, train_labels, class_to_idx, train_transform
)
test_dataset = CustomImageDataset(
    test_images, test_labels, class_to_idx, test_transform
)

###########
# train, test 데이터 로더 생성 => 모델 학습시 입력하는 데이터셋
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

################################################################
##### 사용예시 #####

# 1개의 배치를 추출합니다.
images, labels = next(iter(train_loader))

# 이미지의 shape을 확인합니다. 224 X 224 RGB 이미지 임을 확인합니다.
images[0].shape

####
# 개와 고양이 데이터셋 시각화
# 총 2개의 class(강아지/고양이)로 구성된 사진 파일입니다.
import matplotlib.pyplot as plt

# ImageFolder의 속성 값인 class_to_idx를 할당
labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 4

# 이미지를 출력합니다. RGB 이미지로 구성되어 있습니다.
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # 본래 이미지의 shape은 (3, 300, 300) 입니다.
    # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (300, 300, 3)으로 shape 변경을 한 후 시각화합니다.
    plt.imshow(torch.permute(img, (1, 2, 0)))
plt.show()
