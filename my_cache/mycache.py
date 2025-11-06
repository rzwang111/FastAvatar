from collections import namedtuple
import torch.nn.functional as F
# from sortedcontainers import OrderedDict
import bisect

RowItem = namedtuple('RowItem', ['audio', 'image'])

class MyCache():
    def __init__(self,size, strategy='nearest'):
        self.size = size
        self.full = False
        self.similiraty_history = []
        self.table = []
        self.smoothed_threshold = 0.95
        # self.window_size = window_size
        self.query_idx = 0
        # self.window_size = window_size

        self.strategy = strategy  # 新增参数，支持替换策略，'nearest', 'fifo', 'lru'
        self.usage_order = []  # 用于跟踪 FIFO 和 LRU 的顺序  
        self.integral = 0
        self.previous_error = 0

        self.power = 0
    def query(self,sample):
        idx, sim = self.get_most_similar(sample)
        self.query_idx = idx
        bisect.insort(self.similiraty_history,sim)
        if sim > self.smoothed_threshold:
            if self.strategy == 'lru' and self.query_idx in self.usage_order:
                self.usage_order.remove(self.query_idx)
                self.usage_order.append(self.query_idx)
            return self.table[idx][1]
        return None

    def update(self,audio, image):
        if self.is_full(): 
            self.row_replace(audio, image)
        else:
            self.table.append([audio, image])
            if self.length() == self.size:
                self.full = True

    def get_most_similar(self, sample):
        most_similar_idx = 0
        max_similarity = -float('inf')

        for idx, item in enumerate(self.table):
            sim = self.get_similarity(item[0],sample)
            if sim > max_similarity:
                max_similarity = sim
                most_similar_idx = idx
        return most_similar_idx, max_similarity

    def row_replace(self,audio, image):
        if self.strategy in ['fifo','lru']:
            idx = self.usage_order.pop(0)  # FIFO，移除最早的 ID
            self.usage_order.append(idx)  # 更新顺序
        # if self.strategy == 'lru':
        #     idx = self.usage_order.pop(0)  # lru，移除最早的 ID
        #     self.usage_order.append(idx)
        if self.strategy == 'nearest':
            idx = self.query_idx 

        self.table[idx] = [audio, image]
    
    def get_similarity(self,s1,s2):
        audio1 = s1.reshape(-1)
        audio2 = s2.reshape(-1)

        # 归一化音频张量
        audio1 = audio1 / audio1.norm(dim=0)
        audio2 = audio2 / audio2.norm(dim=0)

        # 计算余弦相似度
        return float(F.cosine_similarity(audio1, audio2, dim=0))
        
    def set_threshold(self,skip_rate,sr):
        # alpha = 0.8
        # idx = int(skip_rate * len(self.similiraty_history))
        # self.smoothed_threshold = (1-alpha) * self.smoothed_threshold + alpha * self.similiraty_history[-idx]
        
        error = sr - skip_rate
        alpha = 1.05

        if error>0:
            delta = 0.001
        else :
            delta = -0.001

        if error * self.previous_error >= 0:
            self.power += 1
        else:
            self.power = 0

        if self.power > 10:
            self.power = 10
        
        self.smoothed_threshold += pow(alpha, self.power) * delta 
        

        # error =  sr - skip_rate
        # # PID coefficients
        # Kp = 0.2
        # Ki = 0.0000
        # Kd = 0.5
        
        # # Initialize PID terms
        # self.integral = getattr(self, 'integral', 0) + error
        # self.integral = max(min(self.integral, 10), -10)
        # self.previous_error = getattr(self, 'previous_error', 0)
        
        # # PID calculations
        # P = Kp * error
        # I = Ki * self.integral
        # D = Kd * (error - self.previous_error)
        # # Update smoothed_threshold using PID output
        # delta = P + I + D

        # self.smoothed_threshold += delta

        # # # Ensure smoothed_threshold does not exceed 0.99
        if self.smoothed_threshold > 0.99:
            self.smoothed_threshold = 0.99
        if self.smoothed_threshold < 0.8:
            self.smoothed_threshold = 0.8
        # Update previous error
        self.previous_error = error

    def length(self):
        return len(self.table)
    def is_full(self):
        return self.full