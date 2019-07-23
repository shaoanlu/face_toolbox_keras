from keras.layers import *
from keras.models import Model

def conv_block(x, num_filter, kernel=3, stride=1, pad=0, depth_wise=False, block_name=""):
    if pad > 0:
        x = ZeroPadding2D(pad)(x)
    if depth_wise:
        x = DepthwiseConv2D(kernel_size=kernel, strides=stride, use_bias=False, name=block_name+"_conv2d")(x)
    else:
        x = Conv2D(num_filter, kernel_size=kernel, strides=stride, use_bias=False, name=block_name+"_conv2d")(x)
    x = BatchNormalization(momentum=0.9, epsilon=0.001, name=block_name+"_batchnorm")(x) # epsilon=2e-5?
    out = Activation("relu", name=block_name+"_relu")(x)
    return out

def fmobilenet(num_classes=202, input_shape=(112,112,3)):
    inp = Input(input_shape)
    minusscalar0 = Lambda(lambda x: x - 127.5, name="minusscalar0")(inp) # "name": "_minusscalar0"
    mulscalar0 = Lambda(lambda x: x / 128, name="mulscalar0")(minusscalar0) # "name": "_mulscalar0"
    
    bf = 8
    conv_1 = conv_block(mulscalar0, num_filter=bf, kernel=3, pad=1, stride=1, block_name="conv_1") # 112/112    
    conv_2_dw = conv_block(conv_1, depth_wise=True, num_filter=bf, kernel=3, pad=1, stride=1, block_name="conv_2_dw") # 112/112
    conv_2 = conv_block(conv_2_dw, num_filter=bf*2, kernel=1, pad=0, stride=1, block_name="conv_2") # 112/112
    
    conv_3_dw = conv_block(conv_2, depth_wise=True, num_filter=bf*2, kernel=3, pad=1, stride=2, block_name="conv_3_dw") # 112/56    
    conv_3 = conv_block(conv_3_dw, num_filter=bf*4, kernel=1, pad=0, stride=1, block_name="conv_3") # 56/56    
    conv_4_dw = conv_block(conv_3, depth_wise=True, num_filter=bf*4, kernel=3, pad=1, stride=1, block_name="conv_4_dw") # 56/56
    conv_4 = conv_block(conv_4_dw, num_filter=bf*4, kernel=1, pad=0, stride=1, block_name="conv_4") # 56/56
    
    conv_5_dw = conv_block(conv_4, depth_wise=True, num_filter=bf*4, kernel=3, pad=1, stride=2, block_name="conv_5_dw") # 56/28    
    conv_5 = conv_block(conv_5_dw, num_filter=bf*8, kernel=1, pad=0, stride=1, block_name="conv_5") # 28/28    
    conv_6_dw = conv_block(conv_5, depth_wise=True, num_filter=bf*8, kernel=3, pad=1, stride=1, block_name="conv_6_dw") # 28/28
    conv_6 = conv_block(conv_6_dw, num_filter=bf*8, kernel=1, pad=0, stride=1, block_name="conv_6") # 28/28
    
    conv_7_dw = conv_block(conv_6, depth_wise=True, num_filter=bf*8, kernel=3, pad=1, stride=2, block_name="conv_7_dw") # 28/14    
    conv_7 = conv_block(conv_7_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_7") # 14/14
    conv_8_dw = conv_block(conv_7, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=1, block_name="conv_8_dw") # 14/14
    conv_8 = conv_block(conv_8_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_8") # 14/14
    conv_9_dw = conv_block(conv_8, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=1, block_name="conv_9_dw") # 14/14
    conv_9 = conv_block(conv_9_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_9") # 14/14
    conv_10_dw = conv_block(conv_9, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=1, block_name="conv_10_dw") # 14/14
    conv_10 = conv_block(conv_10_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_10") # 14/14
    conv_11_dw = conv_block(conv_10, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=1, block_name="conv_11_dw") # 14/14
    conv_11 = conv_block(conv_11_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_11") # 14/14
    conv_12_dw = conv_block(conv_11, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=1, block_name="conv_12_dw") # 14/14
    conv_12 = conv_block(conv_12_dw, num_filter=bf*16, kernel=1, pad=0, stride=1, block_name="conv_12") # 14/14

    conv_13_dw = conv_block(conv_12, depth_wise=True, num_filter=bf*16, kernel=3, pad=1, stride=2, block_name="conv_13_dw") # 14/7
    conv_13 = conv_block(conv_13_dw, num_filter=bf*32, kernel=1, pad=0, stride=1, block_name="conv_13") # 7/7
    conv_14_dw = conv_block(conv_13, depth_wise=True, num_filter=bf*32, kernel=3, pad=1, stride=1, block_name="conv_14_dw") # 7/7
    conv_14 = conv_block(conv_14_dw, num_filter=bf*32, kernel=1, pad=0, stride=1, block_name="conv_14") # 7/7
    body = conv_14
    
    bn1 = BatchNormalization(momentum=0.9, epsilon=2e-5, name="bn1")(body)
    relu1 = PReLU(shared_axes=[1, 2], name="relu1")(bn1)
    pool1 = AveragePooling2D((7,7), name="pool1")(relu1)
    permute = Permute((3,1,2))(pool1)
    flat = Flatten()(permute)
    pre_fc1 = Dense(num_classes, name="pre_fc1")(flat)
    fc1 = BatchNormalization(momentum=0.9, epsilon=2e-5, name="fc1")(pre_fc1)
    return Model(inp, fc1)
