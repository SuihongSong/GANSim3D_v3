import numpy as np
import tensorflow.compat.v1 as tf

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Augment only labels by mutiple times.

def labels_augment_func (a, label_size, labels_augment_times):
    b = tf.reshape(a, [-1, label_size, 1])
    c = tf.tile(b, [1, 1, labels_augment_times])
    d = tf.reshape(c, [-1, label_size*labels_augment_times])
    return d

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer. 
  # x with shape  (None, 128, 4, 4, 4)  # x with shape of [N, channels, x_dim, y_dim, z_dim]  
  # input shape should be [batch, in_channels, in_depth, in_height, in_width], if with data_format='NCDHW'
  # the output shape is the same as input.
  # filter shape [filter_depth, filter_height, filter_width, in_channels,out_channels]
def conv3d(x, fmaps, kernel, gain=np.sqrt(2), padding='SAME', use_wscale=False):
    #assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding=padding, data_format='NCDHW')

# convolutional layer with different kernel sizes along axes.
def conv3d_mulKS(x, fmaps, kernel_x, kernel_y, kernel_z, gain=np.sqrt(2), padding='SAME', use_wscale=False):
    #assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel_x, kernel_y, kernel_z, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding=padding, data_format='NCDHW')
#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1].value], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])
    
#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale3d(x, factors):
    [factor_x, factor_y, factor_z] = factors
    with tf.variable_scope('Upscale3d'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1].value, s[2].value, 1, s[3].value, 1, s[4].value, 1])
        x = tf.tile(x, [1, 1, 1, factor_x, 1, factor_y, 1, factor_z])
        x = tf.reshape(x, [-1, s[1].value, s[2].value * factor_x, s[3].value * factor_y, s[4].value * factor_z])
        return x

#----------------------------------------------------------------------------
# Fused upscale3d + conv3d.
# Faster and uses less memory than performing the operations separately.
  # x with shape  (None, 128, 4, 4, 4)
  # input shape should be [batch, in_channels, in_depth, in_height, in_width], if with data_format='NCDHW'
  # the output shape is the same as input.
  # filter shape [filter_depth, filter_height, filter_width, in_channels,out_channels]
  
def upscale3d_conv3d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:, 1:], w[1:, 1:, :-1], w[1:, :-1, 1:], w[1:, :-1, :-1],
                  w[:-1, 1:, 1:], w[:-1, 1:, :-1], w[:-1, :-1, 1:], w[:-1, :-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2].value * 2, x.shape[3].value * 2, x.shape[4].value * 2]
    return tf.nn.conv3d_transpose(x, w, os, strides=[1,1,2,2,2], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.
def downscale3d(x, factors):
    [factor_x, factor_y, factor_z] = factors
    with tf.variable_scope('Downscale3d'):
        ksize = [1, 1, factor_x, factor_y, factor_z]
        return tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Box filter downscaling layer.
def downscale3d_max(x, factors):
    [factor_x, factor_y, factor_z] = factors
    with tf.variable_scope('Downscale3d'):
        ksize = [1, 1, factor_x, factor_y, factor_z]
        return tf.nn.max_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Box filter wellfc_downscale3d_process layer.
# x: [minibatch, 2, resolution_x, resolution_y, resolution_z]; 2 channels: channel 0 is well locations (where 1 for well locations, 0 for no well locations); 
# channel 1 is facies code channel (where code 0 is mud facies, code 1 is channel facies, code 2 is lobe facies; non-well pixels are also code 0 which can be replaced by any value).
def wellfc_downscale3d_process(x, factor_array, prior_codes): 
    # factor_array includes 3 factors to downscale variable x along x, y, and z dimension.
    # prior_codes: facies codes with decreasing priority, e.g., [1, 2, 0]
    if np.all(factor_array == 1): return x
    prior_codes = [float(k) for k in prior_codes]
    ### downsample facies channel with facies priority
    # (1) arrange facies codes into decreasing code values based on priority: non-well cells are assigned code of -9, with-well cells are set e.g., 
    # code 1 (highest priority) -> 99, code 2 (moderate priority) -> 98, code 3 (lowest priority) -> 97.
    facies_channel = tf.where(tf.math.equal(x[:, 0:1], 0.), tf.fill(tf.shape(x[:, 0:1]), -9.), x[:, 1:2]) # shape of [N, 1, x_cells, y_cells, z_cells] e.g., [N, 1, 128, 128, 32]; non-well cells are -9
    dec_codes = [99. - i for i in range(len(prior_codes))] # e.g., [99, 98, 97]
    for i in range(len(prior_codes)):
        code = prior_codes[i]
        dec_code = dec_codes[i]
        facies_channel = tf.where(tf.math.equal(facies_channel, code), tf.fill(tf.shape(facies_channel), dec_code), facies_channel)
    # (2) use max_pool to downsample and get the maximum codes based on the priority code list
    facies_channel = tf.nn.max_pool3d(facies_channel, ksize=factor_array, strides=factor_array, padding='VALID', data_format='NCDHW') # shape of [N, 1, x_cells, y_cells, z_cells] e.g., [N, 1, 128, 128, 32]; non-well cells are -9
    facies_loc = tf.where(tf.math.greater(facies_channel, 0.), tf.fill(tf.shape(facies_channel), 1.), tf.fill(tf.shape(facies_channel), 0.))       
    #(3) get decreased codes back into original codes: e.g., -9 -> 0, 99->1, 98->2, 97->0
    prior_codes_ = prior_codes + [0.]
    dec_codes_ = dec_codes + [-9.]
    for i in range(len(prior_codes_)):
        code = prior_codes_[i]
        dec_code = dec_codes_[i]
        facies_channel = tf.where(tf.math.equal(facies_channel, dec_code), tf.fill(tf.shape(facies_channel), code), facies_channel)
    # (4) combine facies_loc and facies_channel  
    well_comb_ds = tf.concat([facies_loc, facies_channel], axis = 1)
    return well_comb_ds

#----------------------------------------------------------------------------
# Fused conv3d + downscale3d.
# Faster and uses less memory than performing the operations separately.

def conv3d_downscale3d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:, 1:], w[1:, 1:, :-1], w[1:, :-1, 1:], w[1:, :-1, :-1],
                  w[:-1, 1:, 1:], w[:-1, 1:, :-1], w[:-1, :-1, 1:], w[:-1, :-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,2,2,2], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCDHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3], s[4]])   # [GMCDHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCDHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCDHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCDHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCDHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3,4], keepdims=True)      # [M1111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M1111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])             # [N1DHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCDHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    wellfacies_in,                      # Third input: wellfacies [minibatch, 2, resolution, resolution]: well locations and facies code. 
    probcubes_in,                       # Forth input: probcubes [minibatch, 1, resolution, resolution].
    cond_well           = False,    # Whether condition to well facies data.
    cond_prob           = False,    # Whether condition to probability maps.
    cond_label          = False,    # Whether condition to given global features (labels).
    latent_cube_num     = 8,            # Number of input latent cubes.
    facies_codes        = [0, 1, 2],    # list of facies codes
    prior_codes         = [1, 2, 0],    # list of facies codes with decreasing priority when dowansampling
    resolution_z        = 64,           # Output resolution. Overridden based on dataset.
    resolution_x        = 64,
    resolution_y        = 64,
    beta                = 8e3,          # Used in soft-argmax function, to be tuned for specific cases.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 2048,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 96,          # Maximum number of feature maps in any layer.
    latent_size_z       = 12,            # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    latent_size_x       = 12,
    latent_size_y       = 12,
    initCv_k            = 9,            # kernel size of initial Conv layer to process input latent cubes.
    normalize_latents   = False,        # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,         # True = use fused upscale3d + conv3d, False = separate upscale3d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    wellfc_conv_channels = 6,
    prob_conv_channels  = 6,
    facies_indic  = True,        # Decide whether the output is a facies model or several indicator cubes for each facies.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    out_sizes_log2 = np.array(np.log2([resolution_x, resolution_y, resolution_z]).astype(int))
    out_sizes_log2_dif = out_sizes_log2 - min(out_sizes_log2)
    initCv_fmap_size = np.array([latent_size_x - initCv_k + 1, latent_size_y - initCv_k + 1, latent_size_z - initCv_k + 1])  # feature map size of initial Conv layer used in enlarging input size of trained G.
    initCv_fmap_size_log2 = np.array(np.log2(initCv_fmap_size).astype(int)) # log2 of initCv_fmap_size
    full_usc_thr = min(initCv_fmap_size_log2) + out_sizes_log2_dif
    
    out_sizes_log2_lg = max(out_sizes_log2)
    initCv_fmap_size_log2_lg = max(initCv_fmap_size_log2)
    
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    def upscale_factor(fm, full_upscal_threshold):  # fm with shape of [N, channels, x_dim, y_dim, z_dim]  
        fm_sizes_log2 = np.log2([fm.shape[2].value, fm.shape[3].value, fm.shape[4].value]).astype(int) 
        fm_sizes_log2_dif = fm_sizes_log2 - full_upscal_threshold
        if not np.any(fm_sizes_log2_dif):
            ups_fac = [2, 2, 2]
        else:
            ups_fac = np.where(fm_sizes_log2_dif == 0, 1, 2)         
        return ups_fac
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    latents_in.set_shape([None, latent_cube_num, latent_size_x, latent_size_y, latent_size_z]) # (None, 128)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)  # initialized as 0, assigned by main function, change as training.

    if cond_label:
        labels_in.set_shape([None, label_size, latent_size_x, latent_size_y, latent_size_z])  # (None, N, 4, 4, 4)
        combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)   # (None, 128)
    else:
        labels_in.set_shape([None, 0, latent_size_x, latent_size_y, latent_size_z])  # to give a dimension for labels_in
        combo_in = latents_in    
    if cond_well: 
        wellfacies_in.set_shape([None, 2, resolution_x, resolution_y, resolution_z])
        wellfacies_in = tf.cast(wellfacies_in, tf.float32)
    else:     
        wellfacies_in.set_shape([None, 0, resolution_x, resolution_y, resolution_z])
    if cond_prob:    
        probcubes_in.set_shape([None, len(facies_codes)-1, resolution_x, resolution_y, resolution_z])
        probcubes_in = tf.cast(probcubes_in, tf.float32)
    else:
        probcubes_in.set_shape([None, 0, resolution_x, resolution_y, resolution_z])

    # Building blocks.
    def block(x, prob, wellfc, res):   # x with shape of [N, channels, x_dim, y_dim, z_dim]     
        with tf.variable_scope('stage%d_%d' % (res - 1, 2**res)):           
            if res == 2: # 4x4x4
                #x_sizes_log2 = np.log2([4., 4., 4.]).astype(int)
                
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Conv0'):
                    # x = PN(act(apply_bias(conv3d(x, fmaps=64, kernel=latent_size_x-3, padding='VALID', use_wscale=use_wscale))))  
                    x = PN(act(apply_bias(conv3d(x, fmaps=64, kernel=initCv_k, padding='VALID', use_wscale=use_wscale))))   
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, padding='SAME', use_wscale=use_wscale))))     
                    x_sizes_log2 = np.log2([x.shape[2].value, x.shape[3].value, x.shape[4].value]).astype(int)   #np.log2([8., 8., 4.]).astype(int)                                  
                if cond_prob:    
                    with tf.variable_scope('Add_Prob'):
                        prob_downscaled = downscale3d(prob, (2**(out_sizes_log2 - x_sizes_log2)).astype(int))
                        prob_downscaled_conv = apply_bias(conv3d(prob_downscaled, fmaps=prob_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                        x = tf.concat([x, prob_downscaled_conv], axis=1)
                if cond_well:  
                    with tf.variable_scope('Add_Wellfc'):
                        wellfc_downscaled = wellfc_downscale3d_process(wellfc, (2**(out_sizes_log2 - x_sizes_log2).astype(int)), prior_codes)
                        with tf.variable_scope('Conv0'):
                            wellfc_downscaled_conv_1 = apply_bias(conv3d(wellfc_downscaled, fmaps=1, kernel=1, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv1'):
                            wellfc_downscaled_conv_3 = apply_bias(conv3d(wellfc_downscaled, fmaps=1, kernel=3, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv2'):
                            wellfc_downscaled_conv_5 = apply_bias(conv3d(wellfc_downscaled, fmaps=2, kernel=5, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv3'):
                            wellfc_downscaled_conv_7 = apply_bias(conv3d(wellfc_downscaled, fmaps=2, kernel=7, gain=1, use_wscale=use_wscale))
                        x = tf.concat([x, wellfc_downscaled_conv_1, wellfc_downscaled_conv_3, wellfc_downscaled_conv_5, wellfc_downscaled_conv_7], axis=1)
                with tf.variable_scope('Conv2'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))   
                    
            else: # above 4x4x4  
                ups_fac = upscale_factor(x, full_usc_thr)       
                x = upscale3d(x, ups_fac)
                x_sizes_log2_new = np.log2([x.shape[2].value, x.shape[3].value, x.shape[4].value]).astype(int) 
                if cond_prob: 
                    with tf.variable_scope('Add_Prob'):
                        prob_downscaled = downscale3d(prob, (2**(out_sizes_log2 - x_sizes_log2_new)).astype(int))
                        prob_downscaled_conv = apply_bias(conv3d(prob_downscaled, fmaps=prob_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                        x = tf.concat([x, prob_downscaled_conv], axis=1)
                if cond_well:  
                    with tf.variable_scope('Add_Wellfc'):
                        wellfc_downscaled = wellfc_downscale3d_process(wellfc, (2**(out_sizes_log2 - x_sizes_log2_new).astype(int)), prior_codes)
                        with tf.variable_scope('Conv0'):
                            wellfc_downscaled_conv_1 = apply_bias(conv3d(wellfc_downscaled, fmaps=1, kernel=1, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv1'):
                            wellfc_downscaled_conv_3 = apply_bias(conv3d(wellfc_downscaled, fmaps=1, kernel=3, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv2'):
                            wellfc_downscaled_conv_5 = apply_bias(conv3d(wellfc_downscaled, fmaps=2, kernel=5, gain=1, use_wscale=use_wscale))
                        with tf.variable_scope('Conv3'):
                            wellfc_downscaled_conv_7 = apply_bias(conv3d(wellfc_downscaled, fmaps=2, kernel=7, gain=1, use_wscale=use_wscale))
                        x = tf.concat([x, wellfc_downscaled_conv_1, wellfc_downscaled_conv_3, wellfc_downscaled_conv_5, wellfc_downscaled_conv_7], axis=1)
                k = 3 #if res <= 6 else 5      
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=k, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=k, use_wscale=use_wscale))))                    
                with tf.variable_scope('Conv2'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=k, use_wscale=use_wscale))))      
                #with tf.variable_scope('Conv3'):
                #    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=k, use_wscale=use_wscale))))                          
            return x  # tile
        
    def tofm_prob(x, res):    # obtain softmax of facies types (facies proportion channels) fm with shape of [N, channels, x_dim, y_dim, z_dim]
        lod = out_sizes_log2_lg - initCv_fmap_size_log2_lg + 2 - res
        with tf.variable_scope('tofm_prob_lod%d' % lod):           
            #return tf.math.softmax((apply_bias(conv3d(x, fmaps=len(facies_codes), kernel=1, gain=1, use_wscale=use_wscale))), axis = 1)  
            return 1.*(apply_bias(conv3d(x, fmaps=len(facies_codes), kernel=1, gain=1, use_wscale=use_wscale)))

    def softargmax(x, facies_codes, beta): # facies_codes as array or list    
        facies_codes_4d = tf.tile(tf.convert_to_tensor(facies_codes, dtype=x.dtype)[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], \
                                  [tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]])     
        return tf.cast(tf.reduce_sum(tf.math.softmax(x*beta, axis = 1) * facies_codes_4d, axis=1, keepdims=True), dtype)
    def tofm(x, res):    # obtain facies model from softmax channels (facies proportion channels), # fm with shape of [N, channels, x_dim, y_dim, z_dim]
        # x of shape [N, 3, 64, 32, 16]
        lod = out_sizes_log2_lg - initCv_fmap_size_log2_lg + 2 - res
        with tf.variable_scope('Tofm_lod%d' % lod):   
             return softargmax(x, facies_codes, beta)     

    # Linear structure: simple but inefficient. 
    if structure == 'linear':
        x = block(combo_in, probcubes_in, wellfacies_in, 2)
        cubes_out = tofm_prob(x, 2)
        ups_fac = upscale_factor(x, full_usc_thr)
        for res in range(3, out_sizes_log2_lg - initCv_fmap_size_log2_lg + 2 + 1):  # (out_sizes_log2_lg - latent_sizes_log2_lg + 2) is equavalent to 7 for 128x128x32, since the trained G may be used for enlarged fields (see 1280x1280x320)
            lod = out_sizes_log2_lg - initCv_fmap_size_log2_lg + 2 - res  
            x = block(x, probcubes_in, wellfacies_in, res)
            cube = tofm_prob(x, res)  
            cubes_out = upscale3d(cubes_out, ups_fac)
            with tf.variable_scope('Grow_lod%d' % lod):
                cubes_out = lerp_clip(cube, cubes_out, lod_in - lod)
                ups_fac = upscale_factor(x, full_usc_thr)
        cubes_out = tf.math.softmax(cubes_out, axis = 1) if facies_indic else tofm(tf.math.softmax(cubes_out, axis = 1), out_sizes_log2_lg)     # final output facies model  

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, prob, wellfc, res, lod):
            y = block(x, prob, wellfc, res)   # res can be viewed as x.shape[2], i.e., x_dim of x variable  
            ups_fac = out_sizes_log2 - np.log2([y.shape[2].value, y.shape[3].value, y.shape[4].value]).astype(int)
            cube = lambda: upscale3d(tofm_prob(y, res), 2**ups_fac)
            if res > 2: 
                ups_fac_xy = (np.log2([y.shape[2].value, y.shape[3].value, y.shape[4].value]).astype(int)-np.log2([x.shape[2].value, x.shape[3].value, x.shape[4].value]).astype(int))
                cube = cset(cube, (lod_in > lod), lambda: upscale3d(lerp(tofm_prob(y, res), upscale3d(tofm_prob(x, res - 1), 2**ups_fac_xy), lod_in - lod), 2**ups_fac))
            if lod > 0: cube = cset(cube, (lod_in < lod), lambda: grow(y, prob, wellfc, res + 1, lod - 1))
            return cube()
        cubes_out_pre = grow(combo_in, probcubes_in, wellfacies_in, 2, out_sizes_log2_lg - initCv_fmap_size_log2_lg + 2 - 2)  
        cubes_out = tf.math.softmax(cubes_out_pre, axis = 1) if facies_indic else tofm(tf.math.softmax(cubes_out_pre, axis = 1), np.log2(cubes_out_pre.shape[-1].value).astype(int))
        
    assert cubes_out.dtype == tf.as_dtype(dtype)
    cubes_out = tf.identity(cubes_out, name='cubes_out') 
    return cubes_out

#----------------------------------------------------------------------------
# Discriminator network used in the paper.
def gaussian_kernel_3d(size: int, mean: float, std: float):
    """Makes 3D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel_3d = tf.einsum('i,j,k->ijk', vals, vals, vals)
    gauss_kernel = gauss_kernel_3d / tf.reduce_sum(gauss_kernel_3d)
    return gauss_kernel
kernel_size = 5  # complete size is kernel_size * 2 + 1
std_dev = 5.
gaussian_kernel = tf.cast(gaussian_kernel_3d(kernel_size, 0.0, std_dev)[:,:,:,tf.newaxis, tf.newaxis], tf.float32)


def D_paper(
    cubes_in,                          # Input: cubes [minibatch, channel, height, width].
    wellindicator_in,                      # Input: wellfacies [minibatch, 2, resolution, resolution, resolution]
    resolution_z        = 64,           # Output resolution. Overridden based on dataset.
    resolution_x        = 64,
    resolution_y        = 64,
    facies_codes        = [0, 1, 2],    # list of facies codes
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    cond_well           = True,
    fmap_base           = 1024,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    facies_indic  = True,        # Decide whether the input is a facies model or several indicator cubes for each facies.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    inp_sizes_log2 = np.log2([resolution_x, resolution_y, resolution_z]).astype(int)  # e.g., resolution_x, y, z = [128, 64, 32]; [7, 6, 5]
    inp_sizes_log2_lg = max(inp_sizes_log2)  # 7
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def downscale_factor(inp):
        inp_sizes_log2 = np.log2([inp.shape[2].value, inp.shape[3].value, inp.shape[4].value]).astype(int)
        inp_sizes_log2_dist = inp_sizes_log2 - np.array([2, 2, 2])
        dwsc_factor = np.where(inp_sizes_log2_dist > 0, 2, 1)    
        return dwsc_factor
    def downscale_factor_1(res):
        cur_sizes_log2 = inp_sizes_log2 - max(inp_sizes_log2) + res
        cur_sizes_log2 = np.where(cur_sizes_log2 < 2, 2, cur_sizes_log2)
        factor = inp_sizes_log2 - cur_sizes_log2
        return factor    
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    if facies_indic: 
        cubes_in.set_shape([None, len(facies_codes), resolution_x, resolution_y, resolution_z])
    else:
        cubes_in.set_shape([None, 1, resolution_x, resolution_y, resolution_z])
    cubes_in = tf.cast(cubes_in, dtype)
    if cond_well: 
        wellindicator_in.set_shape([None, 1, resolution_x, resolution_y, resolution_z])
    else:
        wellindicator_in.set_shape([None, 0, resolution_x, resolution_y, resolution_z])
    wellindicator_in = tf.cast(wellindicator_in, dtype)
   # wellindicator_in = tf.nn.conv3d(wellindicator_in, gaussian_kernel, strides=[1,1,1,1,1], padding="SAME", data_format='NCDHW') # [N, 1, 128, 128, 32]
    
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromfm(x, res): # res = 2..inp_sizes_log2_lg
        with tf.variable_scope('FromFM_lod%d' % (inp_sizes_log2_lg - res)):
            return act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))

    def to_pat_feat_av(x, fac, lod): # convert to one feature cube represeting realism of patches
        with tf.variable_scope('ToPF_lod%d' % lod):
            with tf.variable_scope('x_Conv_0'):
                x_pat_feat_0 = act(apply_bias(conv3d_mulKS(x, fmaps=32, kernel_x=3, kernel_y=3, kernel_z=1, use_wscale=use_wscale)))
            with tf.variable_scope('x_Conv_1'):
                x_pat_feat_1 = act(apply_bias(conv3d_mulKS(x_pat_feat_0, fmaps=8, kernel_x=1, kernel_y=1, kernel_z=1, use_wscale=use_wscale)))
            with tf.variable_scope('x_Conv_2'):
                x_pat_feat_2 = act(apply_bias(conv3d(x_pat_feat_1, fmaps=1, kernel=1, use_wscale=use_wscale)))    
            if cond_well:
                pat_sum = tf.math.reduce_sum(tf.math.multiply(upscale3d(x_pat_feat_2, fac), wellindicator_in), axis = [1, 2, 3, 4], keepdims = False)  # [N, ]     
                well_data_num = tf.math.reduce_sum(wellindicator_in, axis = [1, 2, 3, 4], keepdims = False)  # [N, ] 
                out = pat_sum/well_data_num
            else:
                out = tf.math.reduce_sum(x, axis = [1, 2, 3, 4], keepdims = False)  # [N, ] 
            return out

    def init_pat_feat(same_1stD_vec): # initialize patch feature score as 0. same_1stD_vec: vector with the same 1st dimension as the output.
        with tf.variable_scope('init_pat_feat'):
            return tf.math.reduce_mean(same_1stD_vec * 1e-7, axis = [1, 2, 3, 4], keepdims = False) # [N, ] zeros

    def block(x, res): # res = 2..inp_sizes_log2_lg
        dwsc_factor = downscale_factor(x)    
        with tf.variable_scope('stage%d_%d' % (res - 1, 2**res)):
            if res >= 3: # above 4x4x4
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Conv1'):
                    x = act(apply_bias(conv3d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                x = downscale3d(x, dwsc_factor)
            else: # 4x4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=18, use_wscale=use_wscale)))  # fmaps=nf(res-2)
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        cube = cubes_in
        x = fromfm(cube, inp_sizes_log2_lg) 
        pat_feat_16 = init_pat_feat(cubes_in) # [N, ] #
        pat_feat_32 = init_pat_feat(cubes_in) # [N, ] #
        pat_feat_64 = init_pat_feat(cubes_in) # [N, ] #
        for res in range(inp_sizes_log2_lg, 2, -1):  # 7, 6, 5, 4, 3
            lod = inp_sizes_log2_lg - res  # 0, 1, 2, 3, 4, 
            dwsc_factor_1 = downscale_factor_1(res)
            pat_feat_lod = to_pat_feat_av(x, 2**dwsc_factor_1, lod)
            with tf.variable_scope('pat_feat_Grow_lod%d' % lod):
                pat_feat_16 = pat_feat_16 + lerp_clip(0., pat_feat_lod, lod - lod_in)
                pat_feat_32 = pat_feat_32 + lerp_clip(0., pat_feat_lod, 2 + lod_in - lod)
                pat_feat_64 = pat_feat_64 + lerp_clip(0., pat_feat_lod, 2 + lod_in - lod)
            x = block(x, res) 
            dwsc_factor = downscale_factor(cube)             
            cube = downscale3d(cube, dwsc_factor)
            y = fromfm(cube, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        lod = 5
        dwsc_factor_1 = downscale_factor_1(2)
        pat_feat_lod = to_pat_feat_av(x, 2**dwsc_factor_1, lod)
        with tf.variable_scope('pat_feat_Grow_lod%d' % lod): # lod = 4
            pat_feat_out_16 = pat_feat_16 + lerp_clip(0., pat_feat_lod, lod - lod_in)
            pat_feat_out_32 = pat_feat_32 + lerp_clip(0., pat_feat_lod, 2 + lod_in - lod)
            pat_feat_out_64 = pat_feat_64 + lerp_clip(0., pat_feat_lod, 2 + lod_in - lod)
        combo_out = block(x, 2)  


    # Recursive structure: complex but efficient.
    # Pay very careful attention: when pat_feat (patch feature discriminator) is used, pat_feat should always be differentiable 
    # w.r.t. input of the discriminator (cubes_in here). The gradients of pat_feat w.r.t. input cubes of discriminator will ba calculated 
    # in loss.py. That's why 'pat_feat = init_pat_feat(cubes_in)' takes 'cubes_in' as input, and in that function '* 1e-7' is used
    # instead of directly '* 0.' (multiplying 0. would make pat_feat no relation with the input cubes_in). Otherwise, very strange suddenly 
    # worse realism is produced in the results starting from when lod_in < 4, possibly due to vanishing gradients of pat_feat. 
    
    if structure == 'recursive':
        def grow(res, lod): # res: log2 of largest dim of input cube  
            dwsc_factor_1 = downscale_factor_1(res)
            x = fromfm(downscale3d(cubes_in, 2**dwsc_factor_1), res)
            pat_feat_16 = init_pat_feat(cubes_in) # [N, ] #
            pat_feat_32 = init_pat_feat(cubes_in) # [N, ] #
            pat_feat_64 = init_pat_feat(cubes_in) # [N, ] #
            if lod > 0: 
                x_and_pat_feat = tf.cond((lod_in < lod), lambda: grow(res + 1, lod - 1), lambda: (x, pat_feat_16, pat_feat_32, pat_feat_64))
                x = x_and_pat_feat[0]
                pat_feat_16 = x_and_pat_feat[1]
                pat_feat_32 = x_and_pat_feat[2]
                pat_feat_64 = x_and_pat_feat[3]
                
            if lod == 3:
                pat_feat_lod = to_pat_feat_av(x, 2**dwsc_factor_1, lod)
                #pat_feat_16 = tf.cond(tf.math.greater(lod, lod_in), lambda: lerp_clip(0., pat_feat_lod, lod - lod_in), lambda: pat_feat_16)  
                pat_feat_16 = tf.cond((lod > lod_in), lambda: lerp_clip(0., pat_feat_lod, lod - lod_in), lambda: pat_feat_16)
            if lod == 2:
                pat_feat_lod = to_pat_feat_av(x, 2**dwsc_factor_1, lod)
                pat_feat_32 = tf.cond((lod > lod_in), lambda: lerp_clip(0., pat_feat_lod, lod - lod_in), lambda: pat_feat_32)        
            if lod == 1:
                pat_feat_lod = to_pat_feat_av(x, 2**dwsc_factor_1, lod)
                pat_feat_64 = tf.cond((lod > lod_in), lambda: lerp_clip(0., pat_feat_lod, lod - lod_in), lambda: pat_feat_64)        
            
            x = block(x, res); y = x
            if res > 2: 
                dwsc_factor_2 = downscale_factor_1(res - 1)
                y = tf.cond((lod_in > lod), lambda: lerp(x, fromfm(downscale3d(cubes_in, 2**dwsc_factor_2), res - 1), lod_in - lod), lambda: y)
            return y, pat_feat_16, pat_feat_32, pat_feat_64
        combo_out, pat_feat_out_16, pat_feat_out_32, pat_feat_out_64 = grow(2, inp_sizes_log2_lg - 2.)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out_global = tf.identity(combo_out[:, :1], name='scores_out_global')
    scores_out_local_16 = tf.identity(pat_feat_out_16, name='scores_out_local_16')
    scores_out_local_32 = tf.identity(pat_feat_out_32, name='scores_out_local_32')
    scores_out_local_64 = tf.identity(pat_feat_out_64, name='scores_out_local_64')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out_global, scores_out_local_16, scores_out_local_32, scores_out_local_64, labels_out

#----------------------------------------------------------------------------
