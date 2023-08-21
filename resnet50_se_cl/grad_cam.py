import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from gradcam_utils import GradCAM, show_cam_on_image, center_crop_img
from model import resnet50


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # create model
    model = resnet50(num_classes=4)  # Instantiate the final optimized model

    # load model weights
    model_weight_path = "./resnet50.pth" 

    model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    model.eval()

    target_layers = [model.layer4]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image

    # load image
    img_path = "../0.jpg"
    # img_path = "../1.jpg"
    # img_path = "../2.jpg"
    # img_path = "../3.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img_shape = img.size
    print(img_shape)
    img_np = np.array(img, dtype=np.uint8)
    img_np = center_crop_img(img_np, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0


    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    print(type(visualization))
    visualization = Image.fromarray(visualization.astype('uint8')).convert('RGB')
    visualization = visualization.resize(img_shape)
    visualization.save("grad_cam.jpg")


if __name__ == '__main__':
    main()
