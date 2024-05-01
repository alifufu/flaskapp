from flask import Flask, request, jsonify,render_template
from io import BytesIO

#导入模型
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class DenseNet_MNIST(nn.Module):
    """DenseNet for MNIST dataset"""
    def __init__(self, growth_rate=12, block_config=(6, 6, 6), num_init_features=16,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=10):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 2 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet_MNIST, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features * compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


app = Flask(__name__)


model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=61)
model.classifier = nn.Linear(1024, 61)

# 加载模型训练文件
model.load_state_dict(torch.load('E:\edge download\densenet_plant_classification.pth', map_location=torch.device('cpu')))
# 假设你的模型已经被预加载
model.eval()

@app.route('/')
def index():
    # 渲染上传页面
    return render_template('upload.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' in request.files:
        image = request.files['file']
        img = Image.open(BytesIO(image.read()))
        transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_t = transformations(img).unsqueeze(0)

        # 使用模型进行预测
        outputs = model(img_t)
        _, predictions = outputs.topk(5, dim=1)

        # 将预测的张量转换为 NumPy 数组
        predicted_indices = predictions.numpy()[0]

        #with torch.no_grad():
        #predictions = model(img_t)
        # 处理预测结果，返回 top-1 或其他你需要的格式
        class_map = {0: '苹果健康', 1: '苹果黑星病一般', 2: '苹果黑星病严重', 3: '苹果灰斑病', 4: '苹果雪松锈病一般', 5: '苹果雪松锈病严重',
                     6: '樱桃健康', 7: '樱桃白粉病一般', 8: '樱桃白粉病严重',
                     9: '玉米健康', 10: '玉米灰斑病一般', 11: '玉米灰斑病严重', 12: '玉米灰斑病严重', 13: '玉米锈病一般', 14: '玉米叶斑病一般',
                     15: '玉米叶斑病严重', 16: '玉米花叶病毒病', 17: '葡萄健康', 18: '葡萄黑腐病一般', 19: '葡萄黑腐病严重', 20: '葡萄轮斑病一般',
                     21: '葡萄轮斑病严重', 22: '葡萄褐斑病一般', 23: '葡萄褐斑病严重', 24: '柑桔健康', 25: '柑桔黄龙病一般', 26: '柑桔黄龙病严重',
                     27: '桃健康', 28: '桃疮痂病一般', 29: '桃疮痂病严重', 30: '辣椒健康', 31: '辣椒疮痂病一般', 32: '辣椒疮痂病严重', 33: '马铃薯健康',
                     34: '马铃薯早疫病一般', 35: '马铃薯早疫病严重', 36: '马铃薯晚疫病一般', 37: '马铃薯晚疫病严重', 38: '草莓健康',
                     39: '草莓叶枯病一般', 40: '草莓叶枯病严重', 41: '番茄健康', 42: '番茄白粉病一般', 43: '番茄白粉病严重', 44: '番茄疮痂病一般',
                     45: '番茄疮痂病严重', 46: '番茄早疫病一般', 47: '番茄早疫病严重', 48: '番茄晚疫病菌一般', 49: '番茄晚疫病菌严重', 50: '番茄叶霉病一般',
                     51: '番茄叶霉病严重', 52: '番茄斑点病一般', 53: '番茄斑点病严重', 54: '番茄斑枯病一般', 55: '番茄斑枯病严重', 56: '番茄红蜘蛛损伤一般',
                     57: '番茄红蜘蛛损伤严重', 58: '番茄黄化曲叶病毒病一般', 59: '番茄黄化曲叶病毒病严重', 60: '番茄花叶病毒病'}
        # 根据索引从 class_map 获取对应的类别名称
        predicted_classes = [class_map[index] for index in predicted_indices]


        return jsonify({'prediction': predicted_classes })
    return jsonify({'error': 'Image not provided'}), 400



if __name__ == '__main__':
    app.run(debug=True,ssl_context='adhoc')