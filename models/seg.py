import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import math
from config import Config


MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

#tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu


#logits = inference_small(images,is_training=True)


def inference_small(x,
                    is_training,
                    use_bias=True, 
                    num_classes=1):
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['num_classes'] = num_classes
    return inference_small_config(x, c)
    

def inference_small_config(x, c):
    c['bottleneck'] = True
    c['ksize'] = 3
    c['stride'] = 1
    c['num_blocks']=3
    c['stack_stride'] = 1
    c['reuse'] = None
    with tf.variable_scope('down_scale1',reuse=c['reuse']):
        c['conv_filters_out'] = 4
        c['block_filters_internal'] = 16
        
        c['num_blocks']=2
        c['ksize'] = 5
        print 'x:',x.get_shape()
        x = conv(x, c)
        x = bn(x, c)
        x0 = activation(x)         
        print 'x0:',x0.get_shape()
        x1=_max_pool(x0, ksize=2, stride=2)
        c['ksize'] = 3
        x1 = stack(x1, c)
        print 'x1:',x1.get_shape()
    with tf.variable_scope('down_scale2',reuse=c['reuse']):
        c['block_filters_internal'] = 32
        c['num_blocks']=2
        x2=_max_pool(x1, ksize=2, stride=2)
        x2 = stack(x2, c)        
        print 'x2:',x2.get_shape()
    
    with tf.variable_scope('down_scale3',reuse=c['reuse']):
        c['block_filters_internal'] = 64
        c['num_blocks']=1
        x3=_max_pool(x2, ksize=2, stride=2)
        x3 = stack(x3, c)
        print 'x3:',x3.get_shape()
        
    with tf.variable_scope('up_scale1',reuse=c['reuse']):
        c['conv_filters_out'] = 32
        c['block_filters_internal'] = 32
        c['num_blocks']=2
        c['stride'] = 2
        x3 = conv_transpose(x3, c)
        x3 = bn(x3, c)
        x3 = activation(x3)
        print 'up_scale1_tansposed:',x3.get_shape()
        x3 = crop_concate(x2,x3) 
        print 'up_scale1_croped:',x3.get_shape()
        c['stride'] = 1           
        x3 = stack(x3, c)
        print 'up_scale1_stack:',x3.get_shape()
    with tf.variable_scope('up_scale2',reuse=c['reuse']):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['num_blocks']=2
        c['stride'] = 2
        x3 = conv_transpose(x3, c)
        x3 = bn(x3, c)
        x3 = activation(x3)
        print 'up_scale2_tansposed:',x3.get_shape()
        x3 = crop_concate(x1,x3)             
        c['stride'] = 1
        x3 = stack(x3, c)
        print 'up_scale2_stack:',x3.get_shape()
    with tf.variable_scope('up_scale3',reuse=c['reuse']):
        c['conv_filters_out'] = 4
        c['block_filters_internal'] = 4
        c['num_blocks']=1
        c['stride'] = 2
        x3 = conv_transpose(x3, c)
        x3 = bn(x3, c)
        x3 = activation(x3)
        print 'up_scale3_tansposed:',x3.get_shape()
        x3 = crop_concate(x0,x3)    
        c['stride'] = 1         
        x3 = stack(x3, c)    
        print 'up_scale3_stack:',x3.get_shape()
    with tf.variable_scope('output',reuse=c['reuse']):
        c['conv_filters_out'] = c['num_classes']
        c['stride'] = 1
        x3 = conv(x3, c)
        print 'output:',x3.get_shape()

    return x3


    
def loss123123(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.scalar_summary('loss', loss_)

    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1),reuse=c['reuse']):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    # m = 4 if c['bottleneck'] else 1
    m = 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a',reuse=c['reuse']):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b',reuse=c['reuse']):
            c['ksize'] = 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c',reuse=c['reuse']):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A',reuse=c['reuse']):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B',reuse=c['reuse']):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut',reuse=c['reuse']):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 3+2*(c['block_stride']-1)
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)
            shortcut = crop(shortcut,x)
            #print shortcut.get_shape()
        else:
            shortcut = crop(shortcut,x) ## cut
            #print shortcut.get_shape()
    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.random_normal_initializer(stddev=0))
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.random_normal_initializer(stddev=0,mean=1))

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.random_normal_initializer(stddev=0),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.random_normal_initializer(stddev=0,mean=1),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x




def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    with tf.name_scope(name) as scope_name:
        with tf.variable_scope(scope_name) as scope:
        #name=scope + name
        #return tf.get_variable(name,
        #                       shape=shape,
        #                       initializer=initializer,
        #                       regularizer=regularizer,
        #                       trainable=trainable)
        #collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
            try:
                return tf.get_variable(name,
                                       shape=shape,
                                       initializer=initializer,
                                       #dtype=dtype,
                                       regularizer=regularizer,
                                       #collections=[RESNET_VARIABLES],
                                       trainable=trainable)
            except:
                scope.reuse_variables()
                return tf.get_variable(name,
                                       shape=shape,
                                       initializer=initializer,
                                       #dtype=dtype,
                                       regularizer=regularizer,
                                       #collections=[RESNET_VARIABLES],
                                       trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, ksize, filters_in, filters_out]
    std=1/math.sqrt(ksize*ksize*ksize*int(filters_in))
    initializer = tf.random_normal_initializer(stddev=std)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    
    x=tf.nn.conv3d(x, weights, [1, stride, stride, stride, 1], padding='VALID')
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.random_normal_initializer(stddev=0))
        return x + bias
    else:
        return x


def conv_transpose(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, ksize, filters_out, filters_in]
    std=1/math.sqrt(ksize*ksize*ksize*int(filters_in))
    initializer = tf.truncated_normal_initializer(stddev=std)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    x_shape = x.get_shape()
    
    output_shape = tf.stack([x_shape[0],x_shape[1]*stride,x_shape[2]*stride,x_shape[3]*stride,int(filters_out)])
    x=tf.nn.conv3d_transpose(x, weights, output_shape, [1, stride, stride, stride, 1]) ##, padding='VALID' ??
    
    if c['use_bias']:
        bias = _get_variable('bias', x.get_shape()[-1:],
                             initializer=tf.random_normal_initializer(stddev=0))
        return x + bias
    else:
        return x

def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool3d(x,
                          ksize=[1, ksize, ksize, ksize, 1],
                          strides=[1, stride, stride,stride, 1],
                          padding='SAME')

def crop(x1,x2):
    # crop x1 into the shape of x2
    x1_shape = x1.get_shape()
    x2_shape = x2.get_shape()
    border = [int(x1_shape[1] - x2_shape[1]), int(x1_shape[2] - x2_shape[2]), int(x1_shape[3] - x2_shape[3])]
    if border[0]>0 and border[1]>0 and border[2]>0:
        return x1[:,border[0]/2:-border[0]/2,border[1]/2:-border[1]/2,border[2]/2:-border[2]/2,:]
    else:
        return x1
    
def crop_concate(x1,x2):
    # crop x1 into the shape of x2, and concated it with x2
    x1_shape = x1.get_shape()
    x2_shape = x2.get_shape()
    #border=(x1_shape[1:-1]-x2_shape[1:-1])/2
    border = [int(x1_shape[1] - x2_shape[1]), int(x1_shape[2] - x2_shape[2]), int(x1_shape[3] - x2_shape[3])]
    return tf.concat([x2,x1[:,border[0]/2:-border[0]/2,border[1]/2:-border[1]/2,border[2]/2:-border[2]/2,:]],4)