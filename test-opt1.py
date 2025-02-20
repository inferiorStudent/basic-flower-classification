import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './data'

with open(path + '/classname.txt', 'r', encoding='UTF-8') as file:
    class_name = file.read().splitlines()

num_classes = len(class_name)

# construct the model
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# load model.pth
model.load_state_dict(torch.load('./model/flower_cls.pth')['model'])
model = model.to(device)

# test image
test_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model.eval()

with torch.no_grad():
    # batch_size = 1
    test_img = test_transform(Image.open('./test.jpg')).unsqueeze(0).to(device)
    output = model(test_img)
    _, pred = torch.max(output, dim=1)

pred_name = class_name[pred]
print(pred_name)