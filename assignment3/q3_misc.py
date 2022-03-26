import os 
import torch 

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def load_checkpoints(resume,model,optimizer, gpu):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if gpu is None:
            checkpoint = torch.load(resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))
        return model, optimizer, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(resume))

def load_pretrained_checkpoints(pretrained,model,optimizer, gpu):
    # load from pre-trained 
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrained))
            return model, optimizer, start_epoch
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))
