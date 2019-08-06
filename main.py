# coding=utf-8

from config import args 
from model import BiLSTMCL
from dataset import CLDataset
from teacher import TeacherNet
print(args)


model = BiLSTMCL(args)
dataset = CLDataset(args)

if args.mode == 'train':
    train_set = dataset.getTrainDatas()
    eval_set = dataset.getEvalDatas()
    model.train(train_set, eval_set)

elif args.mode == 'eval':
    eval_set = dataset.getEvalDatas()
    print('evaluation')
    model.eval(eval_set)
elif args.mode == 'freeze':
    print('evaluation')
    model.freeze()
elif args.mode == 'preinfer':
    teacher = TeacherNet(args)
    train_set = dataset.getTrainDatas()
    teacher.preInfer(train_set, 'train')

#else args.mode == 'predict':
#    pass

