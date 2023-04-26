from BaseFL import * 

class DecScaleFL(BaseFL):
    """
    The decimal scale is the amount of bits we reserve for the real scale.
    This is influenced by two factors:
        - The precision we want to achieve, because it involves the real precision.
        - The mult depth, since, the bigger the mult depth, the lower the bound can be set.
        - The performance is also influenced by the real scale. The better performance, the lower dec_scale.
        - The security is also influenced. The larger dec_scale, the lower security.
    """
    def __init__(self):
        super().__init__()
        
    def generate_antecedent(self):
        self.precision = BaseFL.get_precision()
        self.perfsec = BaseFL.get_perfsec()
    
    def generate_consequent(self):
        self.dec_scale = BaseFL.generate_dec_scale('consequent')
        
    def generate_rules(self):
        self.rules = []
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low precision we can consider a very_low dec_scale
        self.rules.append(ctrl.Rule(antecedent=(self.precision['very_low'] | 
                                                self.perfsec['very_high']), 
                                    consequent=self.dec_scale['very_low'], 
                                    label='dec_scale very_low'))
        # if either mult_depth or performance or security are high or performance is low, we can have low dec_scale
        self.rules.append(ctrl.Rule(antecedent=(self.precision['low'] | 
                                                self.perfsec['high']), 
                                    consequent=self.dec_scale['low'], 
                                    label='dec_scale low'))
        # if either mult_depth or performance or security performance are medium, we can have medium dec_scale
        self.rules.append(ctrl.Rule(antecedent=(self.precision['medium'] |
                                                self.perfsec['medium']), 
                                    consequent=self.dec_scale['medium'], 
                                    label='dec_scale medium'))
        # if precision is low and mult depth is low then we get dec_scale low
        self.rules.append(ctrl.Rule(antecedent=(self.precision['high'] |
                                                self.perfsec['low']), 
                                    consequent=self.dec_scale['high'], 
                                    label='dec_scale high'))
        # Only in cases with all conditions, we consider feasible to have very_low real scale.
        # So having security, performance and mult_depth very_high abd very_low precision we can consider a very_low dec_scale
        self.rules.append(ctrl.Rule(antecedent=(self.precision['very_high'] | 
                                                self.perfsec['very_low']), 
                                    consequent=self.dec_scale['very_high'], 
                                    label='dec_scale very_high'))
    def generate_rules(self):
        self.rules = []
        
        rules = [
            ('very_high', 'very_high', 'medium'),
            ('very_high', 'high', 'medium'),
            ('very_high', 'medium', 'high'),
            ('very_high', 'low', 'very_high'),
            ('very_high', 'very_low', 'very_high'),
            ('high', 'very_high', 'medium'),
            ('high', 'high', 'medium'),
            ('high', 'medium', 'medium'),
            ('high', 'low', 'high'),
            ('high', 'very_low', 'high'),
            ('medium', 'very_high', 'medium'),
            ('medium', 'high', 'medium'),
            ('medium', 'medium', 'medium'),
            ('medium', 'low', 'medium'),
            ('medium', 'very_low', 'medium'),
            ('low', 'very_high', 'low'),
            ('low', 'high', 'low'),
            ('low', 'medium', 'medium'),
            ('low', 'low', 'medium'),
            ('low', 'very_low', 'medium'),
            ('very_low', 'very_high', 'low'),
            ('very_low', 'high', 'low'),
            ('very_low', 'medium', 'medium'),
            ('very_low', 'low', 'medium'),
            ('very_low', 'very_low', 'medium'),
            ]
        for a, b, c in rules:
            self.rules.append(ctrl.Rule(antecedent=(self.precision[a] & self.perfsec[b]), 
                                                    consequent=self.dec_scale[c], 
                                                    label='[Rule] Precission %s Perfsec %s -> Dec Scale %s'%(a, b, c)))
        
      
    
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
            self.dec_scale.view(sim=sim)
        print("For Precission[%d], Performance[%d] and Security[%d] we get: %0.2f" %(precision, performance, security, sim.output['dec_scale']))
        return np.round(sim.output['dec_scale'])

        
        
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
                    z[i, j] = sim.output['dec_scale']
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
        lines_z = np.linspace(0, self.dec_scale.universe[-1], num=50)
        
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
        ax.set_zlabel('Decimal Scale')
        cset = ax.contourf(x, y, z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='x', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='y', offset=20, cmap='viridis', alpha=0.5)
        ax.view_init(20, 320)
        

class DecScaleFL2(BaseFL):
    """
    The real scale is the amount of bits we reserve for the real scale.
    This is influenced by two factors:
        - The precision we want to achieve, because it involves the real precision.
        - The mult depth, since, the bigger the mult depth, the lower the bound can be set.
        - The performance is also influenced by the real scale. The better performance, the lower real_scale.
        - The security is also influenced. The larger real_scale, the lower security.
    """
    def __init__(self):
        super().__init__()
        
    def generate_antecedent(self):
        self.dec_scale = BaseFL.generate_dec_scale('antecedent')
        self.mult_depth = BaseFL.get_mult_depth()
    
    def generate_consequent(self):
        self.dec_scale_final = BaseFL.generate_dec_scale('consequent')
        
    def generate_rules(self):
        self.rules = []
        

        # For mult_depths of low and very_low we preserve the choice made in the last section, we don't modify them
        for mult_depth_condition in ['very_low', 'low']:
            for dec_scale_condition in ['very_low', 'low', 'medium', 'high', 'very_high']:
                self.rules.append(ctrl.Rule(antecedent=(self.dec_scale[dec_scale_condition] & self.mult_depth[mult_depth_condition]), 
                                            consequent=self.dec_scale_final[dec_scale_condition], 
                                            label='if dec_scale %s and mult_depth is %s we preserve choice' % (dec_scale_condition, mult_depth_condition)))
        
        # Only in cases with all conditions, we consider feasible to have very_low dec scale.
        # So having security, performance and mult_depth very_high abd very_low precision we can consider a very_low dec_scale
        for mult_depth_condition in ['medium']:
            for dec_scale_condition in ['very_low', 'low', 'medium', 'high', 'very_high']:
                self.rules.append(ctrl.Rule(antecedent=(self.dec_scale[dec_scale_condition] & self.mult_depth[mult_depth_condition]), 
                                            consequent=self.dec_scale_final[dec_scale_condition], 
                                            label='if dec_scale %s and mult_depth is %s we preserve choice' % (dec_scale_condition, mult_depth_condition)))
        

        ## HIGH RULES

        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['very_low'] & self.mult_depth['high']), 
                                    consequent=self.dec_scale_final['very_low'], 
                                    label='if dec_scale very_low and mult_depth is high we preserve choice'))
   
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['low'] & self.mult_depth['high']), 
                                    consequent=self.dec_scale_final['low'], 
                                    label='if dec_scale low and mult_depth is high we preserve choice'))
        
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['medium'] & self.mult_depth['high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale medium and mult_depth is high we preserve choice'))
   
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['high'] & self.mult_depth['high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale high and mult_depth is high we preserve choice'))
    
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['very_high'] & self.mult_depth['high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale very_high and mult_depth is high we preserve choice'))
       
        ## VERY HIGH RULES
    
    
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['very_low'] & self.mult_depth['very_high']), 
                                    consequent=self.dec_scale_final['very_low'], 
                                    label='if dec_scale very_low and mult_depth is very_high we preserve choice'))
   
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['low'] & self.mult_depth['very_high']), 
                                    consequent=self.dec_scale_final['low'], 
                                    label='if dec_scale low and mult_depth is very_high we preserve choice'))
        
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['medium'] & self.mult_depth['very_high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale medium and mult_depth is very_high we preserve choice'))
   
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['high'] & self.mult_depth['very_high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale high and mult_depth is very_high we preserve choice'))
    
        self.rules.append(ctrl.Rule(antecedent=(self.dec_scale['very_high'] & self.mult_depth['very_high']), 
                                    consequent=self.dec_scale_final['medium'], 
                                    label='if dec_scale very_high and mult_depth is very_high we preserve choice'))
        
                               
    def generate_ctrl_system(self):
        self.ctrl_sys = ctrl.ControlSystem(self.rules)
        return self.ctrl_sys
    
    def simulate(self, dec_scale, mult_depth, plot=False):
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys , flush_after_run=21 * 21 + 1)
        sim.input['dec_scale'] = dec_scale
        sim.input['mult_depth'] = mult_depth
        sim.compute()
    

        if plot:
            self.mult_depth.view(sim=sim)
            self.dec_scale.view(sim=sim)
            self.dec_scale_final.view(sim=sim)
        print("For Dec Scale[%d] and Mult. Depth[%d] we get: %0.2f" %(dec_scale, mult_depth, sim.output['dec_scale']))
        return np.round(sim.output['dec_scale'])

        
        
    def plot_meshgrid(self, dec_scale=None, mult_depth=None):
        size_x, size_y = int(self.dec_scale.universe[-1]), int(self.mult_depth.universe[-1])
        sim = ctrl.ControlSystemSimulation(self.ctrl_sys, flush_after_run= (size_x * size_y + 1))
        # We can simulate at higher resolution with full accuracy
        upsampled_x = self.dec_scale.universe
        upsampled_y = self.mult_depth.universe
        x, y = np.meshgrid(upsampled_y, upsampled_x)
        
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(0, len(upsampled_x)):
            for j in range(0, len(upsampled_y)):
                    sim.input['mult_depth'] = x[i, j]
                    sim.input['dec_scale'] = y[i, j]
                    
                    
                    
                    
                    sim.compute()
                    z[i, j] = sim.output['dec_scale']
                    #print(x[j, i], y[j, i], z[j, i])
                #dec_scale.view(sim=sim)
                #print(x[i, j], y[i, j], z[i, j])

        # Plot the result in pretty 3D with alpha blending
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=8, antialiased=True)

        cset = ax.contourf(x, y, z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        #cset = ax.contourf(x, y, z, zdir='x', offset=0, cmap='viridis', alpha=0.5)
        cset = ax.contourf(x, y, z, zdir='y', offset=upsampled_x[-1], cmap='viridis', alpha=0.5)
        
        
        
        
        ax.set_xlabel('Mult. Depth(X)')#, fontsize=20)
        ax.set_ylabel('Decimal Scale (Y)')
        ax.set_zlabel('Final Decimal Scale(Z)')
        
        if dec_scale is None or mult_depth is None:
            md, sc= 16, 16
        else:
            md, sc = mult_depth, dec_scale
        
        res = self.simulate(dec_scale=sc, mult_depth=md)
        
        
        lines = np.arange(0, min(size_x, size_y), 1)
        
        lines_x = np.linspace(0, upsampled_y[-1], num=50)
        lines_y = np.linspace(0, upsampled_x[-1], num=50)
        lines_z = np.linspace(0, self.dec_scale_final.universe[-1], num=50)
        
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
            
        ax.scatter(md, sc, res , color='red', linewidth=40)
        ax.plot(md * ones_x, sc * ones_y, lines_z, color='red', linewidth=3)
        ax.plot(md * ones_x, lines_y, res * ones_z, color='red', linewidth=3)
        ax.plot(lines_x, sc * ones_y, res * ones_z, color='red', linewidth=3)

        
        ax.scatter(md, zeros_y, 0, color='purple', linewidth=4)
        ax.plot(lines_x, zeros_y, zeros_z, color='purple', linewidth=3)
        ax.plot(md * ones_x, lines_y, zeros_z, color='purple', linewidth=3)
        #ax.plot(end, lines, zeros, color='purple', linewidth=3)
        #ax.plot(end, end, lines, color='purple', linewidth=3)
        
        ax.scatter(end_x, sc, 0, color='blue', linewidth=4)
        ax.plot(end_x, lines_y, zeros_z, color='blue', linewidth=3)
        ax.plot(lines_x, sc * ones_y, zeros_z, color='blue', linewidth=3)
        #ax.plot(zeros, lines, zeros, color='blue', linewidth=3)
        #ax.plot(zeros, lines, zeros, color='blue', linewidth=3)
        
        ax.scatter(end_x, end_y, res, color='orange', linewidth=4)
        ax.plot(end_x, end_y, lines_z, color='orange', linewidth=3)
        ax.plot(end_x, lines_y, res, color='orange', linewidth=3)
        ax.plot(lines_x, end_y, res, color='orange', linewidth=3)
        #ax.scatter(md, sc, res, color='black', linewidth=4)
        #ax.scatter(40, sc, z[md, sc], color='red')
        #ax.scatter(md, 40, z[md, sc], color='red') 
        #ax.scatter(md, sc, 40, color='red') 
        ax.view_init(20, 320)
        #ax.view_init(0, 90)

