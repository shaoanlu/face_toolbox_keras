from keras.layers import *
from keras.models import Model

def batchnorm(x, name):
    return BatchNormalization(momentum=0.9, epsilon=2e-5, name=name)(x)

def res_block(x, f, block_name, downscale=False):
    s = 2 if downscale else 1
    shortcut = x
    if downscale:
        shortcut = Conv2D(f, 1, strides=s, use_bias=False, padding="valid", name=block_name+"_conv1sc")(shortcut)
        shortcut = BatchNormalization(momentum=0.9, epsilon=2e-5, name=block_name+"_sc")(shortcut)
    
    x = batchnorm(x, name=block_name+"_bn1")
    x = ZeroPadding2D(1)(x)
    x = Conv2D(f, 3, use_bias=False, padding="valid", name=block_name+"_conv1")(x)
    x = batchnorm(x, name=block_name+"_bn2")
    x = PReLU(shared_axes=[1, 2], name=block_name+"_relu1")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(f, 3, strides=s, use_bias=False, padding="valid", name=block_name+"_conv2")(x)
    x = batchnorm(x, name=block_name+"_bn3")
    
    out = Add()([x, shortcut])
    return out

def LResNet100E_IR(weights_path=None):
    inp = Input((112,112,3))
    minusscalar0 = Lambda(lambda x: x - 127.5, name="minusscalar0")(inp) # "name": "_minusscalar0"
    mulscalar0 = Lambda(lambda x: x / 128, name="mulscalar0")(minusscalar0) # "name": "_mulscalar0"
    
    conv0 = ZeroPadding2D(1)(mulscalar0)
    conv0 = Conv2D(64, 3, padding="valid", use_bias=False, name="conv0")(conv0) #conv0_weight
    bn0 = batchnorm(conv0, name="bn0") # bn0_gamma, bn0_bate, bn0_moving_mean, bn0_moving_var
    relu0 = PReLU(shared_axes=[1, 2], name="relu0")(bn0) # relu0_gamma
    
    stage1 = res_block(relu0, 64, "stage1_unit1", True)
    for i in range(2, 4):
        stage1 = res_block(stage1, 64, f"stage1_unit{str(i)}", False)
      
    stage2 = res_block(stage1, 128, "stage2_unit1", True)
    for i in range(2, 14):
        stage2 = res_block(stage2, 128, f"stage2_unit{str(i)}", False)
      
    stage3 = res_block(stage2, 256, "stage3_unit1", True)
    for i in range(2, 31):
        stage3 = res_block(stage3, 256, f"stage3_unit{str(i)}", False)
      
    stage4 = res_block(stage3, 512, "stage4_unit1", True)
    for i in range(2, 4):
        stage4 = res_block(stage4, 512, f"stage4_unit{str(i)}", False)
      
    bn1 = batchnorm(stage4, name="bn1")
    dropout0 = Dropout(0.4, name="dropout0")(bn1)
    permute = Permute((3,1,2))(dropout0)
    flat = Flatten()(permute)
    pre_fc1 = Dense(512, name="pre_fc1")(flat)
    fc1 = batchnorm(pre_fc1, name="fc1")
    
    model = Model(inp, fc1, name="LResNet100E_IR")
    if weights_path is not None:
        model.load_weights(weights_path)
    return model
