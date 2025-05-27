# Tensorflow.js-Py与Js模型互转

## Python模型
通过Python版本的Tensorflow/Keras生成的模型。
包括：Tensorflow Saved Model、Keras HDF5 Model等
## Javascript模型
可以在Tensorflow.js中运行的模型。
包括：tfjs_layers_model、tfjs_graph_model等
## 模型互转
### Py模型转Js模型
使用[Tensorflow.js-converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)工具转换
#### 安装Python环境
使用conda创建一个Python环境
```
conda create --name tfjs python=3.6.8
```
#### 安装Tensorflow.js-converter包
```
pip install tensorflowjs==2.8.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
#### 转换模型
##### tf_saved_model->tfjs_graph_model
[下载tf_saved_model模型](https://www.kaggle.com/models/google/mobilenet-v2)
```
// 转换命令
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve D:\jinux\github\tensorflow.js-tutorial\code\Py与Js模型互转\source\model\Py\saveModel D:\jinux\github\tensorflow.js-tutorial\code\Py与Js模型互转\source\model\Js\graphModel
```
见`source/example`目录下引用转换后模型的案例。

>目前其他格式转换报错，还在研究中......

### Js模型转Py模型
很少会用到这种转换
### Js模型转jS模型
分片、量化、加速



