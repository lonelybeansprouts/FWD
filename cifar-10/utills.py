import numpy as np


def load_cifar():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    file = './cifar-10-batches-py/'
    train_batch_1=unpickle(file+'data_batch_1')
    train_batch_2=unpickle(file+'data_batch_2')
    train_batch_3=unpickle(file+'data_batch_3')
    train_batch_4=unpickle(file+'data_batch_4')
    train_batch_5=unpickle(file+'data_batch_5')
    test_batch=unpickle(file+'data_batch_6')
    
    # for key in train_batch_1:
    #     print(key,len(train_batch_1[b'data']))
    
    def convert(data):
        if isinstance(data, bytes):  return data.decode('ascii')
        if isinstance(data, dict):   return dict(map(convert, data.items()))
        if isinstance(data, tuple):  return map(convert, data)
        return data    
    
    train_batch_1=convert(train_batch_1)
    train_batch_2=convert(train_batch_2)
    train_batch_3=convert(train_batch_3)
    train_batch_4=convert(train_batch_4)
    train_batch_5=convert(train_batch_5)
    test_batch=convert(test_batch)
    
    
    train_Data=np.concatenate((train_batch_1['data'],train_batch_2['data'],train_batch_3['data'],train_batch_4['data'],train_batch_5['data']),axis=0).astype(np.float32)
    train_Labels=np.concatenate((train_batch_1['labels'],train_batch_2['labels'],train_batch_3['labels'],train_batch_4['labels'],train_batch_5['labels']),axis=0).astype(np.float32)
    test_data=np.array(test_batch['data']).astype(np.float32)
    test_label=np.array(test_batch['labels']).astype(np.float32)
    
    
    def label_format_changer(labels):
        label_changed=np.zeros((labels.shape[0],10))    
        for i in range(labels.shape[0]):
               label_changed[i][int(labels[i])]=1.0
        return label_changed   
    
    train_Labels=label_format_changer(train_Labels)
    test_label=label_format_changer(test_label)
    
    train_Data = train_Data.reshape(50000,3,32,32).transpose(0,2,3,1)
    test_data = test_data.reshape(10000,3,32,32).transpose(0,2,3,1)
    
    return [[train_Data,train_Labels],[test_data,test_label]]

