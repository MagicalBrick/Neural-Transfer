import PIL.Image as Image
import torch
import torchvision.transforms as transforms


img_size: int = 512 if torch.cuda.is_available() else 128


def load_img(img_path):
    # load the image and turn it to RGB
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size=(img_size, img_size))
    # turn it to tensor
    img = transforms.ToTensor()(img)
    # add a dimension
    img = img.unsqueeze(0)
    return img


def show_img(img):
    # delete the dimension we added before
    img = img.squeeze(0)
    # turn the image from tensor to RGN
    img = transforms.ToPILImage()(img)
    img.show()
