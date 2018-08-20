import torch
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import PIL.Image as Image
import numpy as np
from model import *
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--image_path', type=str, help="image path")
parser.add_argument('--output_path', type=str, help="output path")
parser.add_argument('--model', type=str, help="model path")
args = parser.parse_args()


class Demo:
    def __init__(self):
        self.args = args
        self.net = model().cuda()
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        self.net.load_state_dict(torch.load(self.args.model))
        self.input_transform = Compose([Resize((512, 512)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.image_path = self.args.image_path

    def group_demo(self):
        self.net.eval()
        attentions=[]
        images_path=glob(self.image_path+"/*.jpg")
        print(images_path)
        for image_path in images_path:
            image=Image.open(image_path).convert('RGB')
            image = self.input_transform(image)
            image = image.unsqueeze(0).cuda()
            feature,attention =self.net.module.generate_attention(image)
            attentions.append(attention)

        group_mean_attentions=torch.stack(attentions)
        group_mean_attention=torch.mean(torch.stack(attentions),dim=0)

        for index,image_path in enumerate(images_path):
            image=Image.open(image_path).convert('RGB')
            image = self.input_transform(image)
            image = image.unsqueeze(0).cuda()
            feature,attention =self.net.module.generate_attention(image)
            mask=self.net.module.dec(feature*group_mean_attention)
            mask = torch.argmax(mask, dim=1)
            image = (image - image.min()) / image.max()
            mask = torch.cat([torch.zeros(1, 512, 512).long().cuda(
                    ), mask, torch.zeros(1, 512, 512).long().cuda()]).unsqueeze(0)
            save_image(mask.float().data * 0.8 + image.data,
                   self.args.output_path+"co_%d.jpg"%(index), normalize=True)


if __name__ == "__main__":
    with torch.no_grad():
        demo = Demo()
        demo.group_demo()

print("Finish!!!")
