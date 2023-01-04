import torch.nn as nn
import torch


class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # It is necessary to use detach to separate the target. At this time, the target is no longer a Variable.
        # This is to dynamically calculate the gradient, otherwise the forward will make an error and cannot be
        # propagated forward.
        self.criterion = nn.MSELoss()  # get MSE loss

    def forward(self, input):  # forward calculate for loss
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):  # backward calculate
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):  # forward  calculate of Gram matrix
        a, b, c, d = input.size()  # a:batch size, b:number of feature map, c*d:size of feature map
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram


class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss