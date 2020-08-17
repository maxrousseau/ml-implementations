import numpy as np
import matplotlib.pyplot as plt
import logging
import pprint

def info(MSG, VALUE):
        logging.basicConfig(format='%(levelname)s: %(message)s',
        level=logging.INFO)
        pp = pprint.PrettyPrinter(indent=4)
        logging.info(MSG)
        if VALUE != None:
	        pp.pprint(VALUE)
        else:
	        None
                print('-'*80)

def imsave(name, img):
        img = img/2+ 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.savefig('%s%s' % ('./fig/', name))
