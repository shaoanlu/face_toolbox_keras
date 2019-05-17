from keras.layers import *
from keras.models import Model
import tensorflow as tf

def conv_block(x, f, k, s=1, block_name="", layer_id="", use_activ=True):
    if k != 1:
        x = ZeroPadding2D(1)(x)
    x = Conv2D(f, k, strides=s, padding='valid', use_bias=False, name=block_name+".conv"+layer_id)(x)
    x = BatchNormalization(epsilon=1e-5, name=block_name+".bn"+layer_id)(x)
    x = Activation("relu")(x) if use_activ else x
    return x

def res_block_with_downsampling(x, f, block_name="cp.resnet.layerN"):
    skip = Conv2D(f, 1, strides=2, use_bias=False, name=block_name+".0.downsample.0")(x)
    skip = BatchNormalization(epsilon=1e-5, name=block_name+".0.downsample.1")(skip)
    x = conv_block(x, f, 3, s=2, block_name=block_name+".0", layer_id="1")
    x = conv_block(x, f, 3, block_name=block_name+".0", layer_id="2", use_activ=False)
    x = Add()([x, skip])
    x = Activation("relu")(x)
    
    skip = x
    x = conv_block(x, f, 3, block_name=block_name+".1", layer_id="1")
    x = conv_block(x, f, 3, block_name=block_name+".1", layer_id="2", use_activ=False)
    x = Add()([x, skip])
    x = Activation("relu")(x)
    return x

def attention_refinment_block(x, f, block_name="cp.arm16"):
    x = Conv2D(f, 3, padding='same', use_bias=False, name=block_name+".conv.conv")(x)
    x = BatchNormalization(epsilon=1e-5, name=block_name+".conv.bn")(x)
    x = Activation("relu")(x)
    
    attn = GlobalAveragePooling2D()(x)
    attn = Reshape((1,1,f))(attn)
    attn = Conv2D(f, 1, use_bias=False, name=block_name+".conv_atten")(attn)
    attn = BatchNormalization(epsilon=1e-5, name=block_name+".bn_atten")(attn)
    attn = Activation("sigmoid")(attn)
    x = Multiply()([x, attn])
    return x

def feature_fusion_block(x1, x2):
    x = Concatenate()([x1, x2])
    x = conv_block(x, 256, 1, block_name="ffm.convblk", layer_id="")    
    attn = GlobalAveragePooling2D()(x)
    attn = Reshape((1,1,256))(attn)
    attn = Conv2D(64, 1, use_bias=False, name="ffm.conv1")(attn)
    attn = Activation("relu")(attn)
    attn = Conv2D(256, 1, use_bias=False, name="ffm.conv2")(attn)
    feat_attn = Activation("sigmoid")(attn)
    attn = Multiply()([x, feat_attn])    
    x = Add()([x, attn])
    return x

def upsampling(x, shape, interpolation="nearest"):    
    if interpolation == "nearest":
        return Lambda(lambda t: tf.image.resize_nearest_neighbor(t, shape, align_corners=True))(x)
    elif interpolation == "bilinear":
        return Lambda(lambda t: tf.image.resize_bilinear(t, shape, align_corners=True))(x)

def maxpool(x, k=3, s=2, pad=1):
    x = ZeroPadding2D(pad)(x)
    x = MaxPooling2D((k,k), strides=s)(x)
    return x
    
def BiSeNet_keras(input_resolution=512):
    inp = Input((input_resolution, input_resolution, 3))
    x = ZeroPadding2D(3)(inp)
    x = Conv2D(64, 7, strides=2, use_bias=False, name="cp.resnet.conv1")(x)
    x = BatchNormalization(epsilon=1e-5, name="cp.resnet.bn1")(x)
    x = Activation('relu')(x)
    x = maxpool(x)
    
    # layer1
    skip = x
    x = conv_block(x, 64, 3, block_name="cp.resnet.layer1.0", layer_id="1")
    x = conv_block(x, 64, 3, block_name="cp.resnet.layer1.0", layer_id="2", use_activ=False)
    x = Add()([x, skip])
    x = Activation("relu")(x)
    skip = x
    x = conv_block(x, 64, 3, block_name="cp.resnet.layer1.1", layer_id="1")
    x = conv_block(x, 64, 3, block_name="cp.resnet.layer1.1", layer_id="2", use_activ=False)
    x = Add()([x, skip])
    x = Activation("relu")(x)
    
    # layer2
    x = res_block_with_downsampling(x, 128, block_name="cp.resnet.layer2")
    feat8 = x
    
    # layer3
    x = res_block_with_downsampling(x, 256, block_name="cp.resnet.layer3")
    feat16 = x
    
    # ARM1
    feat16_arm = attention_refinment_block(feat16, 128, block_name="cp.arm16")
    
    # layer4
    x = res_block_with_downsampling(x, 512, block_name="cp.resnet.layer4")   
    feat32 = x 
    
    # ARM2 and conv_avg
    conv_avg = GlobalAveragePooling2D()(x)
    conv_avg = Reshape((1,1,512))(conv_avg)
    conv_avg = conv_block(conv_avg, 128, 1, block_name="cp.conv_avg", layer_id="")
    avg_up = upsampling(conv_avg, [input_resolution//32, input_resolution//32])
    feat32_arm = attention_refinment_block(x, 128, block_name="cp.arm32")
    feat32_sum = Add(name="feat32_sum")([feat32_arm, avg_up])
    feat32_up = upsampling(feat32_sum, [input_resolution//16, input_resolution//16])
    feat32_up = conv_block(feat32_up, 128, 3, block_name="cp.conv_head32", layer_id="")
    
    feat16_sum = Add(name="feat16_sum")([feat16_arm, feat32_up])
    feat16_up = upsampling(feat16_sum, [input_resolution//8, input_resolution//8])
    feat16_up = conv_block(feat16_up, 128, 3, block_name="cp.conv_head16", layer_id="")
    
    # FFM
    feat_sp, feat_cp8 = feat8, feat16_up
    feat_fuse = feature_fusion_block(feat_sp, feat_cp8)
    
    feat_out = conv_block(feat_fuse, 256, 3, block_name="conv_out.conv", layer_id="")
    feat_out = Conv2D(19, 1, strides=1, use_bias=False, name="conv_out.conv_out")(feat_out)
    feat_out = upsampling(feat_out, [input_resolution, input_resolution], interpolation="bilinear")
    # Ignore feat_out32 and feat_out16 since they are not used in inference phase
    
    return Model(inp, feat_out)