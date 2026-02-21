import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaClassifierModel(torch.nn.Module):
    def __init__(self):
        super(PneumoniaClassifierModel,self).__init__()
        self.vision_branch = torch.nn.Sequential(
            self._conv_block(3,32),
            self._conv_block(32,64),
            self._conv_block(64,128),
            self._conv_block(128,256),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        self.audio_branch = torch.nn.Sequential(
            self._conv_block(1,32),
            self._conv_block(32,64),
            self._conv_block(64,128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        self.classifier = torch.nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )
    
    def _conv_block(self,in_f,out_f):
        return nn.Sequential(
            nn.Conv2d(in_f,out_f,kernel_size=3,padding=2),
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self,xray,audio):
        x_feat=self.vision_branch(xray)
        a_feat=self.audio_branch(audio)
        
        combined=torch.cat((x_feat,a_feat),dim=1)
        return self.classifier(combined)