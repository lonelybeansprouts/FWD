import numpy as np



def load_cifar_100():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    file = './cifar_100/cifar-100-python/'
    train_batch=unpickle(file+'train')
    test_batch=unpickle(file+'test')
    
    # for key in train_batch_1:
    #     print(key,len(train_batch_1[b'data']))
    
    def convert(data):
        if isinstance(data, bytes):  return data.decode('ascii')
        if isinstance(data, dict):   return dict(map(convert, data.items()))
        if isinstance(data, tuple):  return map(convert, data)
        return data    
    
    train_batch=convert(train_batch)
    test_batch=convert(test_batch)
    
    print(train_batch.keys())
    
    train_Data=np.array(train_batch['data']).astype(np.float32)
    train_Labels=np.array(train_batch['fine_labels']).astype(np.float32)
    test_data=np.array(test_batch['data']).astype(np.float32)
    test_label=np.array(test_batch['fine_labels']).astype(np.float32)
    
    def label_format_changer(labels):
        label_changed=np.zeros((labels.shape[0],100))    
        for i in range(labels.shape[0]):
               label_changed[i][int(labels[i])]=1.0
        return label_changed   
    
    train_Labels=label_format_changer(train_Labels)
    test_label=label_format_changer(test_label)
    
    train_Data = train_Data.reshape(50000,3,32,32).transpose(0,2,3,1)
    test_data = test_data.reshape(10000,3,32,32).transpose(0,2,3,1)
    
    return [[train_Data,train_Labels],[test_data,test_label]]

#[[X1,Y1],[X2,Y2]] = load_cifar_100()

# print(X1.shape)
# print(Y1.shape)
# print(X2.shape)
# print(Y2.shape)

# load_cifar()    