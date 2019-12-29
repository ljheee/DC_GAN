
### GAN  生成对抗网络
- 生成卡通人脸

##### 工程结构
- cartoon_img 真实数据集
- detect_img 最终要伪造的“仿真”图片
- params 保存pytorch训练的模型文件

- NetD.py 判别网络
- NetG.py 生成网络 “造假”的人
- Samling.py 采样
- Train.py 训练
- Detect.py 检测&运行，Windows cpu环境可直接运行；gpu时开启cuda


##### 环境
- py3.7
- torch-1.0.0-cp37-cp37m-win_amd64.whl
- torchvision0.1.6
