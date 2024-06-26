


class AverageMeter(object):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.list = []

    
    def update(self, value, n):
        self.val = value
        self.sum += value
        self.count += n
        self.avg = self.sum/ self.count
        self.list.append(value)

