# 继承的基础配置文件
_base_ = ['coco_transformer.py']

# coco类别数，加上背景一共有91个编号
num_classes=91

# 学习率
lr = 0.0001
param_dict_type = 'default'
# backbone学习率
lr_backbone = 1e-05
# 主干网络中需要单独设置学习率的层
lr_backbone_names = ['backbone.0']
# 指定线性投影层的学习率
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
# 线性投影层学习率的缩放因子
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
# batch_size
batch_size = 2
# 权重衰减
weight_decay = 0.0001
# 训练轮数
epochs = 12
# 在第11个epoch开始降低学习率
lr_drop = 11
save_checkpoint_interval = 1
# 随机裁剪的最大范数，防止梯度爆炸
clip_max_norm = 0.1
# 是否使用 OneCycle学习率策略（一种动态调整学习率的方法，能加速收敛）
onecyclelr = False
# 是否使用多步学习率衰减（在指定epoch按固定比例降低学习率，与lr_drop_list = [33, 45]相关联）
multi_step_lr = False
# 多步学习率衰减的epoch列表
lr_drop_list = [33, 45]

# 模型名称
modelname = 'dino'
# 是否冻结某些层的权重，此处表示不冻结权重，所有参数均用来训练
frozen_weights = None
# 使用哪个backbone，此处使用的是resnet50
backbone = 'resnet50'
# 是否使用梯度检查点
use_checkpoint = False

dilation = False
# 位置编码模式，sine位置编码
position_embedding = 'sine'
# 控制位置编码的温度参数（缩放因子）
pe_temperatureH = 20
pe_temperatureW = 20
# 指定从主干网络的哪些中间层返回特征图
return_interm_indices = [1, 2, 3]
# 指定冻结主干网络中包含特定关键词的层，在这里不冻结
backbone_freeze_keywords = None
# encoder的层数
enc_layers = 6
# decoder的层数
dec_layers = 6
# 这个参数没搞懂是啥
unic_layers = 0
# 是否在Transformer层中使用预归一化（将LayerNorm放在注意力机制前，而非之后）
pre_norm = False
# transformer中前馈网络的维度
dim_feedforward = 2048
# transformer隐藏层的维度
hidden_dim = 256
# 不启用dropout
dropout = 0.0
# 多头注意力的头数
nheads = 8
# 解码器的查询数（即每张图像预测的最大目标数）
num_queries = 900
# 查询的维度，为4，对应边界框的坐标 (x, y, w, h)
query_dim = 4
# 是否使用动态模式
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
# 是否随机初始化参考点的 (x, y) 坐标
random_refpoints_xy = False
# 固定参考点的高度和宽度（-1表示不固定）
fix_refpoints_hw = -1

dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False

# 是否在边界框预测中使用可变形注意力
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
# 多尺度特征图的层数
num_feature_levels = 4
# 编码器和解码器中每个可变形注意力头的参考点数
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
# 激活函数类型
transformer_activation = 'relu'
# 批量归一化类型，冻结的批量归一化？
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
# 是否使用辅助损失（在每个解码层都计算损失）
aux_loss = True
# 匈牙利匹配的损失权重
# 分类损失的权重
set_cost_class = 2.0
# 边界框回归损失的权重
set_cost_bbox = 5.0
# GIoU损失的权重
set_cost_giou = 2.0

# 损失函数的系数
# 分类损失的系数
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
# 边界框回归损失的系数
bbox_loss_coef = 5.0
# giou损失的系数
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
# Focal Loss的alpha参数，用于解决类别不平衡的问题
focal_alpha = 0.25

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
# 是否使用去噪训练
use_dn = True
# 去噪训练中的噪声样本数
dn_number = 100
# 边界框噪声的缩放比例
dn_box_noise_scale = 0.4
# 标签噪声比例
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 91

match_unstable_error = True

# for ema
# 是否使用EMA更新模型权重
use_ema = False
# EMA衰减率
ema_decay = 0.9997
# 从第几个epoch开始使用EMA
ema_epoch = 0

use_detached_boxes_dec_out = False

