# 论文结构与代码对应表（v0.4.1）

依据为原稿图2、图4和BGSA式(1)–(5)。原图存在通道标注歧义，图与正文也不完全一致，本版采用的解释在最后列明。默认模型为configs/EMBA-Net.yaml。

| 结构要求 | 本版实现 | 代码位置 | 专项验证 |
|---|---|---|---|
| 图2：RepConv→EMA，再分流 | EMA作用于RepConv的全部输出，再chunk为两路 | glm_paper.GLM_P_B.forward | 检查EMA输出及最终Concat实际输入 |
| 图2：Bottleneck串联 | 分流后的局部分支依次通过各Bottleneck | glm_paper.GLM_P_B.m | 使用3个Bottleneck检验顺序与中间输出 |
| 图2：旁路和中间特征进入C | Concat输入为EMA旁路与每一个Bottleneck输出 | glm_paper.GLM_P_B.forward | 逐项核对Concat张量；不额外拼接局部初始seed |
| 图2：第二个RepConv融合 | 输入通道为(1+N)×隐藏通道 | glm_paper.GLM_P_B.cv2 | 多深度Concat通道检查 |
| 图2重参数化示意的ReLU | RepConv使用ReLU，训练分支融合为单3×3 | repconv.RepConv | 非平凡BN统计下的融合一致性 |
| 图4右侧：Proj,2C | 输入映射至2C，再分C/4 seed与7C/4 context | gated_projection.GatedProjectionGN.proj_in | 检查后续门控宽度 |
| 图4右侧：DSconv,2C−C/4 | context经过真实DW3×3/BN/ReLU/PW1×1/BN/ReLU | gated_projection.GatedProjectionGN.dsconv | 检查groups=通道数 |
| 图4右侧：三次Mul | seed×gate0→Proj,C/2→×gate1→Proj,C→×gate2→Proj,C/2 | gated_projection.GatedProjectionGN.forward | 依次置零三路gate，验证三次乘法均生效 |
| 图4：GN | 最终投影后GroupNorm、非线性 | gated_projection.GatedProjectionGN.norm | 真实批次反向与有限值检查 |
| 图4：Avgpool、Softmax | 每尺度GAP，对尺度维作逐通道Softmax | lsd_paper.LSDheadreplacement.forward | 全模型前向/反向；图中未规定权重维度，本文明确为逐通道 |
| 图4：共享Conv/DEConv/Conv | 同一套1×1 Conv_GN→DEConv_GN→1×1 Conv_GN依次处理三尺度 | lsd_paper.LSDheadreplacement.share_conv | 检查编码器三次输入保持各自原分辨率 |
| 图4：分类/回归预测 | 各有独立DW/PW解码与最终输出投影；各任务模块在尺度间共享 | lsd_paper.LSDheadreplacement | 检查实际depthwise模块及全部可训练参数的梯度连接 |
| BGSA式(3) | 同一个psi处理两种有序拼接，生成两个门控 | figure_bgsa.BGSA.psi | 同时更新专项数值用例 |
| BGSA式(4) | 两个输出均以原始对齐特征计算，不顺序覆写输入 | figure_bgsa.BGSA.forward | 用常数特征检验输出7和5，排除顺序更新的5.5 |
| BGSA式(5) | 两路精炼特征Concat后3×3融合 | figure_bgsa.BGSA.conv | 全模型形状与预算验证 |

## 本版明确处理的歧义

1. **GLM依据选择。** 原图2画EMA先行，正文实现段落写成EMA与Bottleneck并行，式(10)又写为加法。本版按照图2的EMA先行、旁路/中间特征Concat实现。正文和式(10)应改为与此一致，不能说三个互相矛盾的表达同时被逐字复现。
2. **图4的C/4标注。** 图中中间gate也标成C/4，但它与Proj,C/2的输出相乘时维度不相容。本版使用C/4、C/2、C三个gate；这是保证逐元素Mul成立的通道解释。令C=2×共享输出通道，末端Proj,C/2正好回到共享输出通道。
3. **GN/非线性细节。** 图右侧标题为Conv_GN，正文明确要求投影后GN。本版在门控链最终投影后使用GN/SiLU；DSconv内部按图中的BN/ReLU实现。图没有明确每个Proj的独立归一化或激活，本版Proj为1×1线性卷积。
4. **BGSA依据选择。** 本版按式(1)–(5)实现对齐、联合门控与同时更新。各融合节点保留原来两路输入及外部输出通道；common hidden width取输出宽度的一半。上/下行节点都按目标特征大小对齐。
5. **检测尺度与宽度。** 保留n规模实际步长4/8/16及外部通道32/64/128/256。论文的P3/P4/P5标签与图中的未缩放通道数应另行注明缩放/步长；本版没有把实际检测头移到8/16/32，否则会改变原模型的小目标路径。

## 与上一版的区别

v0.3的EMA/Bottleneck并行连接已改为图2的EMA先行及串行中间拼接；简单ProjectionGN已改为完整多级门控链；P3公共潜空间、统一空间融合和跨尺度残差已移除；BGSA已改为论文公式的联合门控及同时更新。原始baseline仍通过独立配置保留。


## v0.4.1预算配置

相对v0.4，结构连接不变；第11/13/15/17/19层的单路隐藏通道为24/4/8/4/36，RepConv的活动3×3训练分支数为1/1/1/2/2。训练图参数2,313,702，融合部署图卷积/GEMM计算量8.72752256G。所有预算配置基于实际运算计数，没有使用验证或测试精度选择。
