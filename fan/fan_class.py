# Modified from: https://github.com/1adrianb/face-alignment
#
# BSD 3-Clause License
#
# Copyright (c) 2017, Adrian Bulat
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import torch

from torch.utils.model_zoo import load_url
from PIL import Image
import numpy as np


from fan_models import FAN, ResNetDepth
from fan_utils import *

models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}

class FaceLandmarks:
    def __init__(self, image, box, device='cuda', verbose=False):
        self.box = box
        self.image = image
        self.device = device
        self.verbose = verbose
        network_size = 4 #large

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # load large 2D FAN network
        self.face_alignment_net = FAN(network_size)
        network_name = '2DFAN-' + str(network_size)
        fan_weights = load_url(models_urls[network_name],
                                map_location=lambda storage, loc: storage)
        self.face_alignment_net.load_state_dict(fan_weights)
        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

    @torch.no_grad()
    def get_landmarks(self):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        """
        image = self.image
        box = self.box

        landmarks = []

        # crop image to detected face and convert to tensor
        center = torch.FloatTensor(
                [box[2] - (box[2] - box[0]) / 2.0,
                box[3] - (box[3] - box[1]) / 2.0])
        center[1] = center[1] - (box[3] - box[1]) * 0.12
        scale = (box[2] - box[0] + box[3] - box[1]) / 195
        crp_img = crop(image, center, scale)

        inp = np.array(crp_img)
        inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()

        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        out = self.face_alignment_net(inp)[-1].detach()
        out = out.cpu()

        pts, pts_img = get_preds_fromhm(out, center, scale)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

        landmarks.append(pts_img.numpy())

        return landmarks
