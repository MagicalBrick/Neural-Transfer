from torch.autograd import Variable
from torchvision import transforms
import torch
from utils.run_code import run_style_transfer
from utils.load_img import load_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select the style you want.
style_img = load_img('./pictures/3.jpg')
style_img = Variable(style_img).to(device)

# select the content, the result will use the style you decided.
content_img = load_img('./pictures/test_2.jpg')
content_img = Variable(content_img).to(device)
input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=400)
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./pictures/result.png')
save_pic.show()
