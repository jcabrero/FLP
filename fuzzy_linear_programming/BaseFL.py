import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

class BaseFL(object):
    Inf = 1 << 32
    def __init__(self):
        self.initialize()
        self.sim = None
    
    def initialize(self):
        self.generate_antecedent()
        self.generate_consequent()
        self.generate_rules()
        self.generate_ctrl_system()
    @staticmethod
    def get_uniform(var):
        var['very_low'] = fuzz.trimf(var.universe, [-BaseFL.Inf, 0, 2.5])
        var['low'] = fuzz.trimf(var.universe, [0, 2.5, 5])
        var['medium'] = fuzz.trimf(var.universe, [2.5, 5, 7.5])
        var['high'] = fuzz.trimf(var.universe, [5, 7.5, 10])
        var['very_high'] = fuzz.trimf(var.universe,[7.5, 10, BaseFL.Inf])
        return var
    
    @staticmethod
    def get_mult_depth():
        mult_depth = ctrl.Antecedent(np.arange(0, 34.1, 1), 'mult_depth')
        mult_depth['very_low'] = fuzz.trimf(mult_depth.universe, [-BaseFL.Inf, 0, 7])
        mult_depth['low'] = fuzz.trimf(mult_depth.universe, [0, 7, 14])
        mult_depth['medium'] = fuzz.trimf(mult_depth.universe, [7, 14, 21])
        mult_depth['high'] = fuzz.trimf(mult_depth.universe, [14, 21, 28])
        mult_depth['very_high'] = fuzz.trimf(mult_depth.universe,[21, 28, BaseFL.Inf])
        #mult_depth.view()
        return mult_depth
    
    @staticmethod
    def get_security():
        security = ctrl.Antecedent(np.arange(0, 10.5, .5), 'security')
        security = BaseFL.get_uniform(security)
        #security.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #security.view()
        return security     
    
    @staticmethod
    def get_precision():
        precision = ctrl.Antecedent(np.arange(0, 10.5, .5), 'precision')
        precision = BaseFL.get_uniform(precision)
        #precision.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #precision.view()
        return precision
    
    @staticmethod
    def get_performance():
        performance = ctrl.Antecedent(np.arange(0, 10.5, .5), 'performance')
        performance = BaseFL.get_uniform(performance)
        #performance.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #performance.view()
        return performance
    
    @staticmethod
    def get_perfsec():
        perfsec = ctrl.Antecedent(np.arange(0, 10.5, .5), 'perfsec')
        perfsec = BaseFL.get_uniform(perfsec)
        #perfsec.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #perfsec.view()
        return perfsec
    
    @staticmethod
    def generate_real_scale(fz_type='consequent'):
        real_scale = None
        universe = np.arange(0, 30.1, 0.5)
        var_name = 'real_scale'
        
        if fz_type == 'antecedent':
            real_scale = ctrl.Antecedent(universe, var_name)
        elif fz_type == 'consequent':
            real_scale = ctrl.Consequent(universe, var_name, defuzzify_method='centroid')
        else:
            raise Exception("Options available are antecedent and consequent")
        real_scale['very_low'] = fuzz.trimf(real_scale.universe, [-BaseFL.Inf, 0, 7.5])
        real_scale['low'] = fuzz.trimf(real_scale.universe, [0, 7.5, 15])
        real_scale['medium'] = fuzz.trimf(real_scale.universe, [7.5, 15, 22.5])
        real_scale['high'] = fuzz.trimf(real_scale.universe, [15, 22.5, 30])
        real_scale['very_high'] = fuzz.trimf(real_scale.universe,[22.5, 30, BaseFL.Inf])
        #real_scale.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #real_scale.view()
        return real_scale
    
    @staticmethod
    def generate_dec_scale(fz_type='consequent'):
        dec_scale = None
        universe = np.arange(0, 60.1, 1)
        var_name = 'dec_scale'
        if fz_type == 'antecedent':
            dec_scale = ctrl.Antecedent(universe, var_name)
        elif fz_type == 'consequent':
            dec_scale = ctrl.Consequent(universe, var_name, defuzzify_method='centroid')
        else:
            raise Exception("Options available are antecedent and consequent")
            
        dec_scale['very_low'] = fuzz.trimf(dec_scale.universe, [-BaseFL.Inf, 12, 24])
        dec_scale['low'] = fuzz.trimf(dec_scale.universe, [12, 24, 36])
        dec_scale['medium'] = fuzz.trimf(dec_scale.universe, [24, 36, 48])
        dec_scale['high'] = fuzz.trimf(dec_scale.universe, [36, 48, 60])
        dec_scale['very_high'] = fuzz.trimf(dec_scale.universe,[48, 60, 60])
        #dec_scale.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #dec_scale.view()
        return dec_scale 
    
    @staticmethod
    def generate_logN(fz_type='consequent'):
        logN = None
        universe = np.arange(0.0, 1.01, 0.05)
        var_name = 'logN'
        if fz_type == 'antecedent':
            logN = ctrl.Antecedent(universe, var_name)
        elif fz_type == 'consequent':
            logN = ctrl.Consequent(universe, var_name, defuzzify_method='centroid')
        else:
            raise Exception("Options available are antecedent and consequent")
            
        logN['very_low'] = fuzz.trimf(logN.universe, [-BaseFL.Inf, 0.0, 0.25])
        logN['low'] = fuzz.trimf(logN.universe, [0.0, 0.25, 0.5])
        logN['medium'] = fuzz.trimf(logN.universe, [0.25, 0.5, 0.75])
        logN['high'] = fuzz.trimf(logN.universe, [0.5, 0.75, 1.0])
        logN['very_high'] = fuzz.trimf(logN.universe,[0.75, 1.0, BaseFL.Inf])
        #logN.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #logN.view()
        return logN 

    @staticmethod
    def generate_lambda(fz_type='consequent'):
        lambd = None
        universe = np.arange(0.0, 1.01, 0.05)
        var_name = 'lambda'
        if fz_type == 'antecedent':
            lambd = ctrl.Antecedent(universe, var_name)
        elif fz_type == 'consequent':
            lambd = ctrl.Consequent(universe, var_name, defuzzify_method='centroid')
        else:
            raise Exception("Options available are antecedent and consequent")
            
        lambd['very_low'] = fuzz.trimf(lambd.universe, [-BaseFL.Inf, 0.0, 0.25])
        lambd['low'] = fuzz.trimf(lambd.universe, [0.0, 0.25, 0.5])
        lambd['medium'] = fuzz.trimf(lambd.universe, [0.25, 0.5, 0.75])
        lambd['high'] = fuzz.trimf(lambd.universe, [0.5, 0.75, 1.0])
        lambd['very_high'] = fuzz.trimf(lambd.universe,[0.75, 1.0, BaseFL.Inf])
        #lambd.automf(names = ['very_low', 'low', 'medium', 'high', 'very_high'])
        #lambd.view()
        return lambd 
    
    def generate_antecedent(self):
        self.precision = BaseFL.get_precision()
        self.performance = BaseFL.get_performance()
        self.security = BaseFL.get_security()
        self.mult_depth = BaseFL.get_mult_depth()
        self.perfsec = BaseFL.get_perfsec()
        # self.real_scale = BaseFL.generate_real_scale('antecedent')
        # self.dec_scale = BaseFL.generate_dec_scale('antecedent')
        # self.logN = BaseFL.generate_logN('antecedent')
        # self.lambd = BaseFL.generate_lambda('antecedent')
    
    def generate_consequent(self):
        self.real_scale = BaseFL.generate_real_scale('consequent')
        self.dec_scale = BaseFL.generate_dec_scale('consequent')
        self.logN = BaseFL.generate_logN('consequent')
        self.lambd = BaseFL.generate_lambda('consequent')
        
        
    def generate_rules(self):
        self.rules = None
    
    def generate_ctrl_system(self):
        self.ctrl_system = None


bfl = BaseFL()