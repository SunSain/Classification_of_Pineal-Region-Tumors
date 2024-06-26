import torch.nn as nn

class one_Binary_LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(one_Binary_LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x): 
        print("x: ",x)
        out = self.linear(x) 
        out = nn.functional.softmax(out, dim=1) 
        return out 
    
class three_layer_Binary_LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(three_layer_Binary_LogisticRegression, self).__init__() 
        self.linear_1 = nn.Linear(input_size, input_size) 
        self.softmax=nn.Softmax(dim=-1)
        self.linear_2 = nn.Linear(input_size, input_size) 
        self.linear_3 = nn.Linear(input_size, num_classes) 
        
  
    def forward(self, x): 
        print("x: ",x)
        out = self.linear_1(x) 
        out = self.softmax(out) 
        out = self.linear_2(out) 
        out = self.softmax(out) 
        out = self.linear_3(out) 
        out = self.softmax(out) 
        return out 

class Seven_layer_Binary_LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(Seven_layer_Binary_LogisticRegression, self).__init__() 
        self.linear_1 = nn.Linear(input_size, input_size) 
        self.softmax=nn.Softmax(dim=-1)
        self.linear_2 = nn.Linear(input_size, input_size) 
        self.linear_3 = nn.Linear(input_size, input_size) 
        self.linear_4 = nn.Linear(input_size, input_size) 
        self.linear_5 = nn.Linear(input_size, input_size) 
        self.linear_6 = nn.Linear(input_size, input_size)
        self.linear_7 = nn.Linear(input_size, num_classes) 
        
  
    def forward(self, x): 
        print("x: ",x)
        out = self.linear_1(x) 
        out = self.softmax(out) 
        out = self.linear_2(out) 
        out = self.softmax(out) 
        out = self.linear_3(out) 
        out = self.softmax(out) 
        out = self.linear_4(out) 
        out = self.softmax(out) 
        out = self.linear_5(out) 
        out = self.softmax(out) 
        out = self.linear_6(out) 
        out = self.softmax(out) 
        out = self.linear_7(out) 
        out = self.softmax(out) 
        return out 
    

class Binary_LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(Binary_LogisticRegression, self).__init__() 
        self.linear_1 = nn.Linear(input_size, input_size) 
        self.softmax=nn.Softmax(dim=-1)
        self.linear_2 = nn.Linear(input_size, input_size) 
        self.linear_3 = nn.Linear(input_size, input_size) 
        self.linear_4 = nn.Linear(input_size, input_size) 
        self.linear_5 = nn.Linear(input_size, num_classes) 
        
  
    def forward(self, x): 
        print("x: ",x)
        out = self.linear_1(x) 
        out = self.softmax(out) 
        out = self.linear_2(out) 
        out = self.softmax(out) 
        out = self.linear_3(out) 
        out = self.softmax(out) 
        out = self.linear_4(out) 
        out = self.softmax(out) 
        out = self.linear_5(out) 
        return out 