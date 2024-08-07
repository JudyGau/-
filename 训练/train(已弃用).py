import numpy as np
import torch
import torch.nn as nn
from dataset import process_data
from transformer import vit_base_patch16_224_in21k

num_classes = 20
data_path = "dataset"
pre_weights_path = "model/pre_train/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
output_path = "model/output/transformer_zqg.pt"

# 训练轮数
num_epoch = 30
label_list = []


def show_predictResult(pre_label, truth_label):
    for pre, truth in pre_label, truth_label:
        print(
            f"预测类别为{label_list[int(pre_label)]}, 而真实类别为{label_list[int(truth_label)]}"
        )


if __name__ == "__main__":
    model = vit_base_patch16_224_in21k(num_classes=num_classes, has_logits=False)

    print("start get data...")
    train_loader, train_label, label_list = process_data(data_path + "/train")
    print("already process data...")

    print("start prepared model...")
    print("if GPU can be used?:", torch.cuda.is_available())  # true 查看GPU是否可用
    model.cuda()  # 利用GPU加速
    # 加载预训练权重参数
    weights = torch.load(pre_weights_path)
    # 应用权重参数到模型实例
    model.load_state_dict(weights)
    print("already prepared model...")

    print("start training...")
    loss = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0
        model.train()  # 模型置于训练状态
        for i, data in enumerate(train_loader):  # train_loader = (inputs, label)
            data[0] = data[0].type(torch.FloatTensor)
            data[1] = data[1].type(torch.FloatTensor)
            optimizer.zero_grad()  # 梯度清零
            pred_label = model(data[0].cuda())  # 预测的输出
            batch_loss = loss(pred_label, data[1].cuda().long())  # 计算误差
            batch_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_acc += np.sum(
                np.argmax(pred_label.cpu().data.numpy(), axis=1) == data[1].numpy()
            )  # 计算准确度
            show_predictResult(
                np.argmax(pred_label.cpu().data.numpy(), axis=1),
                data[1].numpy(),
            )
            train_loss += batch_loss.item()  # 计算误差
        print("epoch:{}".format(epoch + 1))
        print(
            "model's accuracy in train_data is:{}".format(train_acc / len(train_label))
        )
        print("model's loss in train_data is:{}".format(train_loss / len(train_label)))
        print(f"开始保存第{epoch}轮的训练结果")
        torch.save(model.state_dict(), output_path)  # 导出.pt文件
        print(f"第{epoch}轮的训练结果已保存")
        if train_acc / len(train_label) > 0.98:
            break
    print("finish training...")

    print("start testing...")
    test_loader, test_label = process_data(data_path + "/test")
    model.eval()  # 将模型置于评估状态
    test_acc = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data[0] = data[0].type(torch.FloatTensor)
            data[1] = data[1].type(torch.FloatTensor)
            pred_label = model(data[0].cuda())  # 预测输出
            test_acc += np.sum(
                np.argmax(pred_label.cpu().data.numpy(), axis=1) == data[1].numpy()
            )  # 计算准确度
        print("model's accuracy in test_data is:{}".format(test_acc / len(test_label)))
    print("finish testing...")

    # data = torch.rand(1, 3, 224, 224)
    # out = model(data)
    # print(out)
