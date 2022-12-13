import argparse

from metrics import RMSE
from read_dataset import Read_Dataset
from model import MatrixRecSysModel


if __name__ == '__main__':

    ## 设置训练参数
    parser = argparse.ArgumentParser(description="Command")
    parser.add_argument('--learning_rate', default=0.02, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--reg_p', default=0.01, type=float)
    parser.add_argument('--reg_q', default=0.01, type=float)
    parser.add_argument('--hidden_size', default=16, type=int)
    parser.add_argument('--optimizer_type', default="SGD", type=str, help="SGD or BGD")
    parser.add_argument('--train', default=False, action='store_true', help='is train')
    parser.add_argument('--test', default=False, action='store_true', help='is test')
    args = parser.parse_args()

    ## 获取数据集
    RDset = Read_Dataset("userID","movieID","rating")
    model = MatrixRecSysModel(
            lr=args.learning_rate, 
            batch_size=args.batch_size,
            reg_p=args.reg_p, 
            reg_q=args.reg_q, 
            hidden_size=args.hidden_size,
            epoch=args.epoch,
            columns=["userId", "movieId", "rating"],
            metric=RMSE
            )

    if args.train:
        train_dataset,dev_dataset = RDset.get_set("train")
        model.load_dataset(train_dataset,dev_dataset)
        print("Starting training ...")
        model.train(optimizer_type=args.optimizer_type)
        print("Finish training.")
    
    if args.test:
        test_dataset = RDset.get_set("test")
        train_dataset,dev_dataset = RDset.get_set("train")
        model.load_dataset(train_dataset,dev_dataset)
        print("load finisd")
        print("Starting predicting ...")
        model.test(test_dataset)
        print("Finish predicting, you can submit your results on the leaderboard.")

