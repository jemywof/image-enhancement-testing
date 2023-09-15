import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN

img = Image.open('./')
lr_img = np.array(img)


#available models ('weights') are as follows:
# RDN: psnr-large, psnr-small, noise-cancel
# RRDN: gans
# example usage: model = RRDN(weights='gans')
rdn = RDN(weights='psnr-large')
sr_image = rdn.predict(lr_img)
Image.fromarray(sr_img)

