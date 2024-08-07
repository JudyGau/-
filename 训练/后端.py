from flask import Flask, jsonify, request
from PIL import Image
import base64
import io
import torchvision.transforms as transforms
import numpy as np
import torch
from transformer import vit_base_patch16_224_in21k

# 创建 Flask 应用程序实例
app = Flask(__name__)

# 在这里定义其他的代码和路由处理函数

# 定义预处理函数，将图片进行resize和归一化等操作
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # 转换为张量
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化处理
    ]
)

# 创建模型实例
model = vit_base_patch16_224_in21k(num_classes=20, has_logits=False)

# 加载模型权重
weights_path = "model/output/transformer_zqg.pt"
weights = torch.load(weights_path)
model.load_state_dict(weights)

# 设置模型为评估模式
model.eval()

# 将模型转移到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 类别名称列表
class_names = [
    "土豆",
    "茄子",
    "香菜",
    "玉米",
    "青椒",
    "韭菜",
    "豌豆",
    "梨子",
    "苹果",
    "西兰花",
    "蒜苗",
    "水稻",
    "小麦",
    "胡萝卜",
    "葡萄",
    "南瓜",
    "黄瓜",
    "香蕉",
    "西红柿",
    "西瓜",
]


# 定义预测函数
@app.route("/predict", methods=["POST"])
def predict():
    # 从POST请求中获取base64编码的图片数据
    img_base64 = request.form.get("img_data")

    # 解码base64数据并转换为PIL Image对象
    image = Image.open(io.BytesIO(base64.b64decode(img_base64)))

    # 图片预处理，并转换为模型输入的张量
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        pre = model(image_tensor)
        pre = torch.softmax(pre, dim=1).cpu().numpy()[0]  # 计算softmax并转换为numpy数组

    # 获取预测结果中概率最高的前5个类别索引
    y_hat = np.argsort(pre)[::-1][:5]

    # 构建预测结果列表
    result = []
    for y in y_hat:
        name = class_names[y]
        score = str(pre[y])
        result.append({"name": name, "score": score})

    # 返回JSON响应
    return jsonify(
        {"error_code": "undefined", "error_msg": "未知错误", "results": result}
    )


# 启动服务器
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
