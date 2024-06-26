


class Metrics(object):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.acc = 0.0
        self.sen = 0.0
        self.pre = 0.0
        self.F1 = 0.0
        self.CM = []
        self.spe = 0.0
        self.auc = 0.0
        self.a =[]
        self.b=[]
        self.c=[]

    
    def update(self, a,b,c, acc,sen,pre,F1,spe,auc,CM):
        self.acc = acc
        self.sen = sen
        self.pre = pre
        self.F1 = F1
        self.CM = CM
        self.spe = spe
        self.auc = auc
        self.a =a
        self.b=b
        self.c=c


