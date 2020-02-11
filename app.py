

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import fastai

from fastai.vision import *
from fastai.vision.learner import create_head, cnn_config, num_features_model, create_head
from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook, _hook_inner
from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock
from skimage import measure
from skimage.transform import resize
import csv
import random
# from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from gevent.pywsgi import WSGIServer
import cv2
from werkzeug.utils import secure_filename

from jinja2 import Environment, FileSystemLoader

from sanic import Sanic, response
app = Sanic(__name__)

app.static('/static', './static')


@app.route('/', methods=['GET'])
def index(request):
    data = {'name': 'name'}
    template = env.get_template('index.html')
    html_content = template.render(name=data["name"])
    # Main page
    return response.html(html_content)


class Hcolumns(nn.Module):
    def __init__(self, hooks:Collection[Hook], nc:Collection[int]=None):
        super(Hcolumns,self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None 
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                self.factorization.append(nn.Sequential(
                    conv2d(nc[i],nc[-1],3,padding=1,bias=True),
                    conv2d(nc[-1],nc[-1],3,padding=1,bias=True)))
                #self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))
        
    def forward(self, x:Tensor):
        n = len(self.hooks)
        out = [F.interpolate(self.hooks[i].stored if self.factorization is None
            else self.factorization[i](self.hooks[i].stored), scale_factor=2**(self.n-i),
            mode='bilinear',align_corners=False) for i in range(self.n)] + [x]
        return torch.cat(out, dim=1)

class DynamicUnet_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, 
                 self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None,
                 last_cross:bool=True, bottle:bool=False, **kwargs):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        hc_c = [x.shape[1]]
        
        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, 
                blur=blur, self_attention=sa, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)
            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        hc_c.append(ni)
        layers.append(Hcolumns(self.hc_hooks, hc_c))
        layers += [conv_layer(ni*len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


path = Path("")
export_file_name = 'export.pkl'
path = Path(__file__).parent
env = Environment(loader=FileSystemLoader(['./templates']))
learn = load_learner(path, export_file_name)


@app.route('/predict', methods=['GET', 'POST'])
def predict(request):
    f = request.files.get('file')

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.name))
    write = open(file_path, 'wb')
    write.write(f.body)
    img = open_image(file_path)
    im = learn.predict(img)[2].numpy()
    im = im[1:]*255
    im = np.moveaxis(im, -1,0).squeeze()
    im=im.transpose()
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    image =cv2.imread(file_path)
    dst = cv2.addWeighted(image/255, 0.5, im, 0.5, 0,  dtype = cv2.CV_32F)
    cv2.imwrite(file_path,dst*255)
    status = 'detected'
    if  learn.predict(img)[0].data.sum()> 0 :            
        print('Pneumothorax detected')
        
    else:
        status = 'not detected.'

    return response.json({
        'file_name': file_path,
        'status': status,
        # 'confidence': confidence
    })

# Callback to grab an image given a local path
@app.route('/get_image')
def get_image(request):
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return response.file(path, mime_type='image/' + ext[1:])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, access_log=False, workers=1)

# In[ ]:




