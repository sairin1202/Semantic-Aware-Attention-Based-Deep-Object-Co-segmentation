import torch
import PIL.Image as Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from model import *
import numpy as np
from torchvision.utils import save_image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--image1', type=str, help="image1 path")
parser.add_argument('--image2', type=str, help="image2 path")
parser.add_argument('--output1', type=str, help="output1")
parser.add_argument('--output2', type=str, help="output2")
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
        self.image1_path = self.args.image1
        self.image2_path = self.args.image2

    def single_demo(self):
        self.net.eval()
        image1 = Image.open(self.image1_path).convert('RGB')
        image2 = Image.open(self.image2_path).convert('RGB')
        image1 = self.input_transform(image1)
        image2 = self.input_transform(image2)
        image1, image2 = image1.unsqueeze(0).cuda(), image2.unsqueeze(0).cuda()

        output1, output2 = self.net(image1, image2)

        output1 = torch.argmax(output1, dim=1)
        output2 = torch.argmax(output2, dim=1)

        image1 = (image1 - image1.min()) / image1.max()
        image2 = (image2 - image2.min()) / image2.max()

        output1 = torch.cat([torch.zeros(1, 512, 512).long().cuda(
        ), output1, torch.zeros(1, 512, 512).long().cuda()]).unsqueeze(0)
        output2 = torch.cat([torch.zeros(1, 512, 512).long().cuda(
        ), output2, torch.zeros(1, 512, 512).long().cuda()]).unsqueeze(0)

        save_image(output1.float().data * 0.8 + image1.data,
                   self.args.output1, normalize=True)
        save_image(output2.float().data * 0.8 + image2.data,
                   self.args.output2, normalize=True)


if __name__ == "__main__":
    demo = Demo()
    demo.single_demo()

print("Finish!!!")
