from BaseFL import * 
class LogQFL(BaseFL):
    """
    The _Sca scale is the amount of bits we reserve for the real scale.
    This is influenced by two factors:
        - The precision we want to achieve, because it involves the real precision.
        - The mult depth, since, the bigger the mult depth, the lower the bound can be set.
        - The performance is also influenced by the real scale. The better performance, the lower lambda.
        - The security is also influenced. The larger lambda, the lower security.
    """
    def __init__(self):
        super().__init__()
        
    def generate_antecedent(self):
        self.precision = BaseFL.get_precision()
        self.perfsec = BaseFL.get_perfsec()
    
    def generate_consequent(self):
        self.lambd = BaseFL.generate_lambda('consequent')
        
    def generate_rules(self):
        self.rules = []
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low precision we can consider a very_low lambda
        self.rules.append(ctrl.Rule(antecedent=(self.precision['very_low'] | 
                                                self.perfsec['very_high']), 
                                    consequent=self.lambd['very_low'], 
                                    label='lambd very_low'))
        # if either mult_depth or performance or security are high or performance is low, we can have low lambd
        self.rules.append(ctrl.Rule(antecedent=(self.precision['low'] | 
                                                self.perfsec['high']), 
                                    consequent=self.lambd['low'], 
                                    label='lambd low'))
        # if either mult_depth or performance or security performance are medium, we can have medium lambd
        self.rules.append(ctrl.Rule(antecedent=(self.precision['medium'] |
                                                self.perfsec['medium']), 
                                    consequent=self.lambd['medium'], 
                                    label='lambd medium'))
        # if precision is low and mult depth is low then we get lambd low
        self.rules.append(ctrl.Rule(antecedent=(self.precision['high'] |
                                                self.perfsec['low']), 
                                    consequent=self.lambd['high'], 
                                    label='lambd high'))
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low precision we can consider a very_low lambd
        self.rules.append(ctrl.Rule(antecedent=(self.precision['very_high'] | 
                                                self.perfsec['very_low']), 
                                    consequent=self.lambd['very_high'], 
                                    label='lambd very_high'))
    
    def generate_ctrl_system(self):
        self.ctrl_sys = ctrl.ControlSystem(self.rules)
        return self.ctrl_sys
    
    def simulate(self, precision, performance, security, plot=False):
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys) # , flush_after_run=21 * 21 + 1)
        #perfsec = (performance + security) * 0.5
        perfsec = max(performance, security)
        sim.input['perfsec'] = perfsec
        sim.input['precision'] = precision
        sim.compute()
    

        if plot:
            self.precision.view(sim=sim)
            self.perfsec.view(sim=sim)
            self.lambd.view(sim=sim)
        print("For Precission[%d], Performance[%d] and Security[%d] we get: %0.2f" %(precision, performance, security, sim.output['lambda']))
        return sim.output['lambda']

        
        
    def plot_meshgrid(self, precision=None, performance=None, security=None):
        
        # We can simulate at higher resolution with full accuracy
        size_x, size_y = int(self.perfsec.universe[-1]), int(self.precision.universe[-1])
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys, flush_after_run= (size_x * size_y) + 1)
        upsampled_x = self.perfsec.universe
        upsampled_y = self.precision.universe
        x, y = np.meshgrid(upsampled_y, upsampled_x)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(0, len(upsampled_x)):
            for j in range(0, len(upsampled_y)):
                    sim.input['perfsec'] = x[i, j]
                    sim.input['precision'] = y[i, j]
                    sim.compute()
                    z[i, j] = sim.output['lambda']
                #dec_scale.view(sim=sim)
                #print(x[i, j], y[i, j], z[i, j])

        # Plot the result in pretty 3D with alpha blending
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        
        
        perfsec = None
        
        if precision is None or performance is None or security is None:
            precision = 10
            performance = 4
            security = 4
            perfsec = 4
        else:
            #perfsec = (performance + security) * 0.5
            perfsec = max(performance, security)
        
    
        res = self.simulate(precision, performance, security)
        
        lines = np.arange(0, min(size_x, size_y), 1)
        
        lines_x = np.linspace(0, upsampled_y[-1], num=50)
        lines_y = np.linspace(0, upsampled_x[-1], num=50)
        lines_z = np.linspace(0, self.lambd.universe[-1], num=50)
        
        ones = np.ones_like(lines)
        ones_x = np.ones_like(lines_x)
        ones_y = np.ones_like(lines_y)
        ones_z = np.ones_like(lines_z)
        
        end = ones * lines[-1]
        end_x = ones_x * lines_x[-1]
        end_y = ones_y * lines_y[-1]
        end_z = ones_z * lines_z[-1]
        
        zeros = np.zeros_like(lines)
        zeros_x = np.zeros_like(lines_x)
        zeros_y = np.zeros_like(lines_y)
        zeros_z = np.zeros_like(lines_z)
        
        ax.scatter(perfsec, precision, res, color='red', linewidth=4) 
        ax.scatter(perfsec, precision, res, color='red', linewidth=40)
        ax.plot(perfsec * ones_x, precision * ones_y, lines_z, color='red', linewidth=3)
        ax.plot(perfsec * ones_x, lines_y, res * ones_z, color='red', linewidth=3)
        ax.plot(lines_x, precision * ones_y, res * ones_z, color='red', linewidth=3)

        ax.set_xlabel('Perf.Sec.')#, fontsize=20)
        ax.set_ylabel('Precission')
        ax.set_zlabel('Real Scale')
        cset = ax.contourf(x, y, z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='x', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='y', offset=20, cmap='viridis', alpha=0.5)
        ax.view_init(20, 340)
        
lqfl = LogQFL()
lqfl.simulate(precision=10, performance=4, security=5, plot=True)
# precision, mult_depth, performance, security
#rsfl.simulate(1, 16, 9, 9)
lqfl.plot_meshgrid()