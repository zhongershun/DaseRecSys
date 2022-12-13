import pandas as pd
import numpy as np
class Read_Dataset():
    
    def __init__(self,label1,label2,scores):
        self.train_dataset,self.dev_dataset,self.test_dataset = return_dataset(label1,label2,scores)
    
    def get_set(self,train):
        if train == "train":
            return self.train_dataset,self.dev_dataset
        elif train == "test":
            return self.test_dataset

def return_dataset(label1,label2,scores):
    dtype = [(label1, np.int32), (label2, np.int32), (scores, np.float32)]
    print("正在读取数据......")
    print("正在读取训练集数据......")
    train_dataset = pd.read_csv("data/train.csv", usecols=range(3), dtype=dict(dtype))
    print("正在读取验证集数据......")
    dev_dataset = pd.read_csv("data/dev.csv", usecols=range(3), dtype=dict(dtype))
    print("正在读取测试集数据......")
    dtype = [("userId", np.int32), ("movieId", np.int32)]
    test_dataset = pd.read_csv("data/test.csv", usecols=range(1,3), dtype=dict(dtype))
    print("数据载入完毕")
    return train_dataset,dev_dataset,test_dataset