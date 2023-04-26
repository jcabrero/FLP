from BaseFL import * 
class LogNFL(BaseFL):
    """
    The real scale is the amount of bits we reserve for the real scale.
    This is influenced by two factors:
        - The security we want to achieve, because it involves the real security.
        - The mult depth, since, the bigger the mult depth, the lower the bound can be set.
        - The performance is also influenced by the real scale. The better performance, the lower logN.
        - The security is also influenced. The larger logN, the lower security.
    """
    def __init__(self):
        super().__init__()
        
    def generate_antecedent(self):
        self.performance = BaseFL.get_performance()
        self.security = BaseFL.get_security()
    
    def generate_consequent(self):
        self.logN = BaseFL.generate_logN('consequent')
        #self.lambda = BaseFL.generate_lambda('consequent')
        
    def generate_rules(self):
        self.rules = []
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low security we can consider a very_low logN
        self.rules.append(ctrl.Rule(antecedent=(self.security['very_low'] | 
                                                self.performance['very_high']), 
                                    consequent=self.logN['very_low'], 
                                    label='logN very_low'))
        # if either mult_depth or performance or security are high or performance is low, we can have low logN
        self.rules.append(ctrl.Rule(antecedent=(self.security['low'] | 
                                                self.performance['high']), 
                                    consequent=self.logN['low'], 
                                    label='logN low'))
        # if either mult_depth or performance or security performance are medium, we can have medium logN
        self.rules.append(ctrl.Rule(antecedent=(self.security['medium'] |
                                                self.performance['medium']), 
                                    consequent=self.logN['medium'], 
                                    label='logN medium'))
        # if security is low and mult depth is low then we get logN low
        self.rules.append(ctrl.Rule(antecedent=(self.security['high'] |
                                                self.performance['low']), 
                                    consequent=self.logN['high'], 
                                    label='logN high'))
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low security we can consider a very_low logN
        self.rules.append(ctrl.Rule(antecedent=(self.security['very_high'] | 
                                                self.performance['very_low']), 
                                    consequent=self.logN['very_high'], 
                                    label='logN very_high'))
      
    
    def generate_ctrl_system(self):
        self.ctrl_sys = ctrl.ControlSystem(self.rules)
        return self.ctrl_sys
        
    def simulate(self, performance, security, plot=False):
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys) # , flush_after_run=21 * 21 + 1)
        #performance = (performance + security) * 0.5
        sim.input['security'] = security
        sim.input['performance'] = performance
        sim.compute()
    

        if plot:
            self.security.view(sim=sim)
            self.performance.view(sim=sim)
            self.logN.view(sim=sim)
        print("For Performance[%d] and Security[%d] we get: %0.2f" %(performance, security, sim.output['logN']))
        return sim.output['logN']

        
        
    def plot_meshgrid(self, security=None, performance=None):
        
        # We can simulate at higher resolution with full accuracy
        size_x, size_y = int(self.performance.universe[-1]), int(self.security.universe[-1])
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys, flush_after_run= (size_x * size_y) + 1)
        upsampled_x = self.performance.universe
        upsampled_y = self.security.universe
        x, y = np.meshgrid(upsampled_y, upsampled_x)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(0, len(upsampled_x)):
            for j in range(0, len(upsampled_y)):
                    sim.input['performance'] = x[i, j]
                    sim.input['security'] = y[i, j]
                    sim.compute()
                    z[i, j] = sim.output['logN']
                #dec_scale.view(sim=sim)
                #print(x[i, j], y[i, j], z[i, j])

        # Plot the result in pretty 3D with alpha blending
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        
        
        
        if security is None or performance is None:
            security = 10
            performance = 4
        else:
            #performance = (performance + security) * 0.5
            performance = max(performance, security)
        
    
        res = self.simulate(performance=performance, security=security)
        
        lines = np.arange(0, min(size_x, size_y), 1)
        
        lines_x = np.linspace(0, upsampled_y[-1], num=50)
        lines_y = np.linspace(0, upsampled_x[-1], num=50)
        lines_z = np.linspace(0, self.logN.universe[-1], num=50)
        
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
        
        print(res)
        ax.scatter(performance, security, res, color='red', linewidth=4) 
        ax.scatter(performance, security, res, color='red', linewidth=40)
        ax.plot(performance * ones_x, security * ones_y, lines_z, color='red', linewidth=3)
        ax.plot(performance * ones_x, lines_y, res * ones_z, color='red', linewidth=3)
        ax.plot(lines_x, security * ones_y, res * ones_z, color='red', linewidth=3)
        
        
        
        ax.set_xlabel('Performance')#, fontsize=20)
        ax.set_ylabel('Security')
        ax.set_zlabel('log N Score')
        cset = ax.contourf(x, y, z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='x', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='y', offset=20, cmap='viridis', alpha=0.5)
        ax.view_init(20, 340)
        