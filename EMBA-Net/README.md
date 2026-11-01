# EMBA-Net 论文图结构修正版 v0.4.1（8.73G）

本版落实图2的GLM串联与中间拼接、图4右侧的多级门控Conv_GN，以及BGSA式(1)–(5)。它替代之前只补齐主要算子的v0.3草案。原论文图文有矛盾和一处通道标注歧义，采用的尺寸解释在STRUCTURE_MAP.md逐项列出。

相对v0.4，本版保持结构连接，仅更新GLM隐藏宽度与活动重参数分支配置；实际计算量是8.72752256G，不是将8.684G四舍五入。

## 预算实测

单类别、n规模、640×640、batch=1：

| 指标 | 本版 |
|---|---:|
| 训练图参数总数 | 2,313,702（约2.31M） |
| 相对用户原版2,314,132 | 减少430，约0.019% |
| 部署图参数总数 | 2,096,350（约2.10M） |
| 部署图计算量 | 8.72752256 GFLOPs（约8.73G） |

训练图包含参与前向和反向的重参数分支；部署图将RepConv和DEConv等价合并。参数与FLOPs保持的是约2.31M/8.7G的显示量级，并非逐个参数或逐次操作与旧数值严格相等。

FLOPs采用每MAC=2 FLOPs，统计卷积和矩阵乘法，包含函数式DEConv；不计归一化、激活、池化、插值、逐元素运算、解码及NMS。计数已与PyTorch算子profiler的同口径小计核对。原包普通THOP得到的约8.7386G漏计DEConv，在本口径下原版为13.5314G，比较时必须统一口径。

## 这次落实的结构

**GLM：** RepConv→EMA→等分。一路旁路，另一路经过串行Bottleneck；旁路与每个Bottleneck输出进入Concat，再由第二个RepConv融合。RepConv激活改为图中的ReLU。n规模每处一个Bottleneck，其他深度由构建器传入内部串联次数。

**LSD输入Conv_GN：** Proj,2C→分出C/4 seed与7C/4 context→context经过真正的DW/PW深度可分离卷积→分出C/4、C/2、C三路gate→三次Mul和Proj,C/2→Proj,C→Proj,C/2→GN。图中中间gate重复标C/4，不能与Proj,C/2结果逐元素相乘；本版明确使用C/2，详见对应表。

**LSD主体：** 各尺度保留自己的原分辨率，经上述Conv_GN投影，再GAP并对尺度维做逐通道Softmax；同一套Conv_GN→DEConv_GN→Conv_GN共享编码器处理三尺度；分类与回归分别经过独立DW/PW解码和最终投影。上一版的P3公共潜空间与跨尺度残差已移除。

**BGSA：** 先对齐到目标网格，投影后由同一个psi处理两种有序拼接来生成门控，再用原始对齐特征同时更新两路，Concat后3×3融合。不是旧代码的各自Sigmoid和顺序覆写。

主干和检测输出保留n规模实际步长4/8/16。论文中的P3/P4/P5标号、未缩放通道数字及与图2相冲突的GLM文字/公式，应在修订稿中同步说明。STRUCTURE_MAP.md区分了已实现的连接和为解决图文歧义所采用的解释。

## 文件选择

使用默认的 **configs/EMBA-Net.yaml**。另一份EMBA-Net-baseline.yaml保留原始实现供对照。

主要文件：

- emba_net/glm_paper.py：图2的EMA先行、串行Bottleneck及Concat。
- emba_net/gated_projection.py：图4右侧的Proj/DSconv/三次Mul。
- emba_net/lsd_paper.py：三尺度原分辨率共享编码和DW/PW任务解码。
- emba_net/figure_bgsa.py：论文公式的BGSA。
- emba_net/repconv.py：训练多分支与部署等价合并。
- STRUCTURE_MAP.md：逐项对应论文结构的说明。

GLM单路隐藏通道（第11/13/15/17/19层）为24/4/8/4/36；每个RepConv的3×3训练分支数为1/1/1/2/2，另有1×1分支和形状兼容时的恒等分支；所有分支参与计算，部署统一合并为一个3×3。LSD共享输出通道为16。上述选择只依据结构与计算预算，没有用测试精度筛选。

## 检查与训练

在解压后的EMBA-Net-Run目录打开PowerShell：

```powershell
& 'D:\Anaconda\envs\yolov13_trt\python.exe' tools/check_model.py
& 'D:\Anaconda\envs\yolov13_trt\python.exe' tools/profile_model.py
& 'D:\Anaconda\envs\yolov13_trt\python.exe' -m unittest discover -s tests -v
```

将--data-root换成自己的已有数据根目录，该目录下应有images/train、images/val、images/test及对应labels目录：

```powershell
python run.py check --data-root 'D:\你的数据\data'
python run.py smoke --batch 2 --data-root 'D:\你的数据\data'
python run.py train --data-root 'D:\你的数据\data'
```

也可把已有完整data目录放入本项目，使用默认路径和bat入口。训练设置在configs/train.yaml。试跑入口已与正式训练入口保持相同优化器选择、weight_decay和梯度范数裁剪（10）；高阶门控训练时应保留该稳定措施。当前训练器仍是原包的最小流程，按验证损失保存best，尚未加入完整AP评估。

## 部署

使用本版实际训练产生的兼容权重：

```powershell
python tools/convert_deploy.py --weights runs/你的训练/best.pt --output runs/你的训练/best_deploy.pt
python tools/export_onnx.py --weights runs/你的训练/best.pt --output EMBA-Net-figure.onnx --imgsz 640
```

ONNX导出会合并RepConv和DEConv。训练checkpoint和部署checkpoint均有严格加载及回读测试。新结构应重新训练，不能沿用旧模型的90.42%精度或旧设备FPS。

## 验证记录

15项测试全部通过，含具体拓扑、三次Mul有效性、BGSA同时更新、原分辨率编码、正样本反向、裁剪后训练更新、融合和两种checkpoint回读。GPU真实图像batch=2、640输入的一次训练更新与一次验证前向通过。

ONNX checker和640输入ONNX Runtime检查通过；重参数化融合前后最大输出差0.0001984，ONNX坐标最大差0.0018921输入像素、概率最大差约1.74e-7。导出容限分别为0.01像素和1e-5概率；所有输出都检查了有限性。详见verification.json。

本次没有完成全量训练、收敛性或精度验证，也没有重新测试Jetson FPS。此前数据包发现的重复与标注问题未在本次模型修改中自动处理，正式比较应使用同一份经过核对的独立划分。

来源与许可证见LICENSE、NOTICE.md。原始实现通过baseline配置保留，正文/图的矛盾及所有尺寸解释详见STRUCTURE_MAP.md。
