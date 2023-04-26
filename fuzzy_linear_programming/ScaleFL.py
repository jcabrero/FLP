from RealScaleFL import *
from DecScaleFL import *

class ScaleFL(object):
    def __init__(self):
        
        self.rsfl = RealScaleFL()
        self.rsfl2 = RealScaleFL2()
        
        self.dsfl = DecScaleFL()
        self.dsfl2 = DecScaleFL2()
    
    def simulate(self, precision, performance, security, mult_depth, plot=False):
        
        dec_scale = self.dsfl.simulate(precision, performance, security, plot)
        real_scale = self.rsfl.simulate(precision, performance, security, plot)
        real_scale_final = self.rsfl2.simulate(real_scale, mult_depth)
        dec_scale_final = self.dsfl2.simulate(dec_scale, mult_depth)
        mid = int(dec_scale_final - real_scale_final)
        res = [dec_scale_final] + [mid] * (mult_depth -2) + [dec_scale_final]
        print("RES: ", [dec_scale_final] + [mid] * (mult_depth -2) + [dec_scale_final], np.array(res).sum())
        if plot == True:
            self.rsfl.plot_meshgrid(precision, performance, security)
            self.dsfl.plot_meshgrid(precision, performance, security)
            self.rsfl2.plot_meshgrid(real_scale, mult_depth)
            self.dsfl2.plot_meshgrid(dec_scale, mult_depth)
        
        return dec_scale_final, real_scale_final
