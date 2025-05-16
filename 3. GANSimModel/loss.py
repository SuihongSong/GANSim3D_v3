import numpy as np
import tensorflow.compat.v1 as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def gaussian_kernel(size: int, mean: float, std: float,):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def wellfc_downscale3d_1step(x, factor_array, prior_codes):
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
    # ksize can not be set as tensors in this environment.
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
    
# Nearest-neighbor upscaling.
def upscale3d(x, factors):
    if np.all(factors == 1): return x
    factor_x = factors[0]
    factor_y = factors[1]
    factor_z = factors[2]    
    s = tf.shape(x)
    #s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1, s[4], 1])
    x = tf.tile(x, [1, 1, 1, factor_x, 1, factor_y, 1, factor_z])
    x = tf.reshape(x, [-1, s[1], s[2] * factor_x, s[3] * factor_y, s[4] * factor_z])
    return x

def indicatorize(facies_cube, codes):
    ind_cubes = tf.zeros((tf.shape(facies_cube)[0], 0, tf.shape(facies_cube)[2], tf.shape(facies_cube)[3], tf.shape(facies_cube)[4]), tf.float32)
    for code in codes:
        ind_cubes = tf.concat([ind_cubes, tf.where(tf.math.equal(facies_cube, code), tf.fill(tf.shape(facies_cube), 1.), tf.fill(tf.shape(facies_cube), 0.))], axis = 1)
    return ind_cubes        
#----------------------------------------------------------------------------
# Generator loss function.

#** Only the labels inputted into G is of the form of cube (same size as latent vectors); labels from D is still of form [None, label size]


def G_wgan_acgan(G, D, lod, 
                 labels, well_facies, prob_cubes, 
                 minibatch_size, 
                 resolution_z        = 64,           # Output resolution. Overridden based on dataset.
                 resolution_x        = 64,
                 resolution_y        = 64,
                 facies_codes        = [0, 1, 2],
                 prior_codes         = [1, 2, 0],# facies codes with decreasing priority when downsampling
                 code_prob_order     = [1, 2],  # e.g., [1, 2] means the probmaps correspond to the second and the third code in facies_codes.
                 facies_indic = True, # Decide wether facies model or facies indicators is produced from the generator          
                 cond_well           = False,    # Whether condition to well facies data.
                 cond_prob           = False,    # Whether condition to probability maps.
                 cond_label          = False,    # Whether condition to given global features (labels).
                 Wellfaciesloss_weight = 0.7, 
                 MudProp_weight = 0.2, 
                 Width_weight = 0.2, 
                 Sinuosity_weight = 0.2, 
                 orig_weight = 2, 
                 labeltypes = None, 
                 Probcubeloss_weight = 0.0000001, 
                 batch_multiplier = 4, 
                 lossnorm = True,
                 GANloss_mean = 0.,    # used to normalize GAN loss into a Gaussian-like range;
                 GANloss_std = 1.,     # global features' mean and std are not included here, can included them if needed.
                 wellloss_mean = 0.,
                 wellloss_std = 1.,
                 probloss_mean = 0.,
                 probloss_std = 1.,
                 global_weight   = 1.,      # weight of global realism term.
                 local_weight    = 0.):     # weight of local realism term.               
    #labeltypes, e.g., labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity', set in config file
    # loss for channel orientation is not designed below, so do not include "0" in labeltypes.
    # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.   
    
    if cond_prob:
        prob_cubes = tf.cast(prob_cubes, tf.float32)
        prob_cubes_lg = tf.reshape(tf.tile(tf.expand_dims(prob_cubes, 1), [1, batch_multiplier, 1, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
    else:
        prob_cubes_lg = tf.zeros([0] + G.input_shapes[3][1:])
        batch_multiplier = 1
    
    if cond_label:
        label_size = len(labeltypes)
        labels_list = []
        for k in range(label_size):
            labels_list.append(tf.random.uniform(([minibatch_size]), minval=-1, maxval=1))
        if 1 in labeltypes:   # mud proportion
            ind = labeltypes.index(1)
            labels_list[ind] = tf.clip_by_value(labels[:, ind] + tf.random.uniform([minibatch_size], minval=-0.2, maxval=0.2), -1, 1)    
        labels_in = tf.stack(labels_list, axis = 1)   
        labels_lg = tf.reshape(tf.tile(tf.expand_dims(labels_in, 1), [1, batch_multiplier, 1]), ([-1] + [G.input_shapes[1][1].value]))
        labels_lg_cube = tf.expand_dims(tf.expand_dims(tf.expand_dims(labels_lg, -1), -1), -1)
        labels_lg_cube = tf.tile(labels_lg_cube, [1,1,G.input_shapes[1][-3], G.input_shapes[1][-2], G.input_shapes[1][-1]])
    else: 
        labels_lg_cube = tf.zeros([0] + G.input_shapes[1][1:])
        
    if cond_well:
        well_facies = tf.cast(well_facies, tf.float32)
        well_facies_lg = tf.reshape(tf.tile(tf.expand_dims(well_facies, 1), [1, batch_multiplier, 1, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))
    else:
        well_facies_lg = tf.zeros([0] + G.input_shapes[2][1:])
        
    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
 
    fake_cubes_out = G.get_output_for(latents, labels_lg_cube, well_facies_lg, prob_cubes_lg, is_training=True) # shape of [N, 3, 128, 128, 32]  
        
    fake_scores_out_global, fake_scores_out_local_16, fake_scores_out_local_32, fake_scores_out_local_64, fake_labels_out = fp32(D.get_output_for(fake_cubes_out, well_facies_lg[:, :1], is_training=True))
    loss_local_16 = -fake_scores_out_local_16 
    loss_local_32 = -fake_scores_out_local_32 
    loss_local_64 = -fake_scores_out_local_64 
    loss_global = - fake_scores_out_global
    loss_local_16 = tfutil.autosummary('Loss_G/GANloss_local_16', loss_local_16)
    loss_local_32 = tfutil.autosummary('Loss_G/GANloss_local_32', loss_local_32)
    loss_local_64 = tfutil.autosummary('Loss_G/GANloss_local_64', loss_local_64)
    loss_global = tfutil.autosummary('Loss_G/GANloss_global', loss_global)
    
    loss = (loss_local_16 + 10 * loss_local_32 + 100 * loss_local_64) * local_weight + loss_global * global_weight    
    if lossnorm: loss = (loss - GANloss_mean) / GANloss_std   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     

    if cond_label:       
        with tf.name_scope('LabelPenalty'):
            def addMudPropPenalty(index):
                MudPropPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: MudPropPenalty = (MudPropPenalty -0.36079434843794) / 0.11613414177144  # To normalize this loss 
                MudPropPenalty = tfutil.autosummary('Loss_G/MudPropPenalty', MudPropPenalty)        
                MudPropPenalty = MudPropPenalty * MudProp_weight  
                return loss+MudPropPenalty
            if 1 in labeltypes:
                ind = labeltypes.index(1)
                loss = addMudPropPenalty(ind)
            
            def addWidthPenalty(index):
                WidthPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: WidthPenalty = (WidthPenalty -0.600282781464712) / 0.270670509379704  # To normalize this loss 
                WidthPenalty = tfutil.autosummary('Loss_G/WidthPenalty', WidthPenalty)             
                WidthPenalty = WidthPenalty * Width_weight            
                return loss+WidthPenalty
            if 2 in labeltypes:
                ind = labeltypes.index(2)
                loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addWidthPenalty(ind), lambda: loss)
            
            def addSinuosityPenalty(index):
                SinuosityPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
                if lossnorm: SinuosityPenalty = (SinuosityPenalty -0.451279248935835) / 0.145642580091667  # To normalize this loss 
                SinuosityPenalty = tfutil.autosummary('Loss_G/SinuosityPenalty', SinuosityPenalty)            
                SinuosityPenalty = SinuosityPenalty * Sinuosity_weight              
                return loss+SinuosityPenalty
            if 3 in labeltypes:
                ind = labeltypes.index(3)
                loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addSinuosityPenalty(ind), lambda: loss)  
    if cond_well:   
        def Wellpoints_L2loss(well_facies, fake_cubes):
            # Theoretically, the easiest way for downsampling well facies data is, I) get lod value from lod tensor,
            # II) calculate the downsampling factor_array as did in networks.py, III) downsampling, IV) upsampling into original resolution.
            # However, the question lies in, how to get the lod value from lod tensor. Here, we use tf1, tf.disable_v2_behavior(), and tf.disable_eager_execution(),
            # in such environment, tf.cond() does not work as expected, I tried many ways to work around tf.cond() but sill not succeed. 
            # Thus, have to iteratively downsample step by step. Also, tf.nn.max_pool3d can not use tensors as the kernels...
            # 1. well facies --downsample-with-priority--> upsample back into original resolution               
            out_sizes_log2 = np.array(np.log2([resolution_x, resolution_y, resolution_z]).astype(int))  # [8, 8, 5]
            out_sizes_log2_lg = max(out_sizes_log2)  # 8
            well_facies_dsmp = well_facies
            for i in range(out_sizes_log2_lg-1):
                i_tf = tf.cast(tf.fill([], i), tf.float32)  
                dw_fct = out_sizes_log2 - i - 1
                dw_fct = np.where(dw_fct >= np.array([2, 2, 2]), 2, 1)   # assume the coarest res is 4x4x4, corresponding to 2, 2, 2               
                well_facies_dsmp = tf.cond(tf.math.less(i_tf, tf.floor(lod)), lambda: wellfc_downscale3d_1step(well_facies_dsmp, dw_fct, prior_codes), lambda: well_facies_dsmp)              
            well_facies = well_facies_dsmp
            
            well_facies_upsmp = well_facies  
            for j in range(out_sizes_log2_lg-1):
                j_tf = tf.cast(tf.fill([], j), tf.float32) 
                up_fct = out_sizes_log2 - j - 1
                up_fct = np.where(up_fct >= np.array([2, 2, 2]), 2, 1)   # assume the coarest res is 4x4x4, corresponding to 2, 2, 2               
                well_facies_upsmp = tf.cond(tf.math.less(j_tf, tf.floor(lod)), lambda: upscale3d(well_facies_upsmp, up_fct), lambda: well_facies_upsmp) 
            well_facies = well_facies_upsmp
            if facies_indic:
                well_facies = tf.concat([well_facies[:,0:1], indicatorize(well_facies[:,1:2], facies_codes) * well_facies[:,0:1]], axis = 1)
            # 2. calculate loss based on difference of input well facies and output fake_cubes    
            loss = tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:] - fake_cubes))
            loss = loss / tf.reduce_sum(well_facies[:, 0:1])
            return loss
        def addwellfaciespenalty(well_facies, fake_cubes_out, loss, Wellfaciesloss_weight):
            with tf.name_scope('WellfaciesPenalty'):
                WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_cubes_out)       
                if lossnorm: WellfaciesPenalty = (WellfaciesPenalty - wellloss_mean) / wellloss_std   # 0.002742
                WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
                loss += WellfaciesPenalty * Wellfaciesloss_weight   
            return loss   
        loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addwellfaciespenalty(well_facies_lg, fake_cubes_out, loss, Wellfaciesloss_weight), lambda: loss)
  
    if cond_prob:
        def addprobloss(probs, fake_probs_, weight, batchsize, relzs, loss):  # fakes as indicators: [N*relzs, 3, 128, 128, 32]        
            with tf.name_scope('ProbcubePenalty'):   
                probs_fake = tf.reduce_mean(tf.reshape(fake_probs_, ([batchsize, relzs] + G.input_shapes[3][1:])), 1)   # probs for different indicators         
                ProbPenalty = tf.nn.l2_loss(probs - probs_fake)  # L2 loss
                if lossnorm: ProbPenalty = (ProbPenalty- probloss_mean) / probloss_std   # normalize
                ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
            loss += ProbPenalty * weight
            return loss
        fake_probs = tf.gather(fake_cubes_out, indices=code_prob_order, axis=1) if facies_indic else tf.gather(indicatorize(fake_cubes_out, facies_codes), indices=code_prob_order, axis=1)
        loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addprobloss(prob_cubes, fake_probs, Probcubeloss_weight, minibatch_size, batch_multiplier, loss), lambda: loss)        
     
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)    
    return loss


#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, opt, minibatch_size, reals, labels, well_facies, prob_cubes, facies_codes,
    cond_well       = False,    # Whether condition to well facies data.
    cond_prob       = False,    # Whether condition to probability maps.
    cond_label      = False,    # Whether condition to given global features (labels).  
    facies_indic    = True, # Decide wether facies model or facies indicators is produced from the generator                 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    label_weight    = 10,       # Weight of the conditioning terms. 
    global_weight   = 0.3,      # weight of global realism term.
    local_weight    = 0.7):     # weight of local realism term.               

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    
    if cond_label:
        labels_cube = tf.expand_dims(tf.expand_dims(tf.expand_dims(labels, -1), -1), -1)
        labels_cube = tf.tile(labels_cube, [1,1,G.input_shapes[0][-3], G.input_shapes[0][-2], G.input_shapes[0][-1]])   
    else:
        labels_cube = tf.zeros([0] + G.input_shapes[1][1:])
    fake_cubes_out = G.get_output_for(latents, labels_cube, well_facies, prob_cubes, is_training=True)  # shape of [N, 3, 128, 128, 32]  
    reals_input = indicatorize(reals, facies_codes) if facies_indic else reals     # indicators of real facies cubes; shape of [N, 3, 128, 128, 32] 
        
    real_scores_out_global, real_scores_out_local_16, real_scores_out_local_32, real_scores_out_local_64, real_labels_out = fp32(D.get_output_for(reals_input, well_facies[:, 0:1], is_training=True))
    fake_scores_out_global, fake_scores_out_local_16, fake_scores_out_local_32, fake_scores_out_local_64, fake_labels_out = fp32(D.get_output_for(fake_cubes_out, well_facies[:, 0:1], is_training=True))
    real_scores_out_global = tfutil.autosummary('Loss_D/real_scores_global', real_scores_out_global)
    real_scores_out_local_16 = tfutil.autosummary('Loss_D/real_scores_out_local_16', real_scores_out_local_16)  
    real_scores_out_local_32 = tfutil.autosummary('Loss_D/real_scores_out_local_32', real_scores_out_local_32)  
    real_scores_out_local_64 = tfutil.autosummary('Loss_D/real_scores_out_local_64', real_scores_out_local_64)  
    fake_scores_out_global = tfutil.autosummary('Loss_D/fake_scores_out_global', fake_scores_out_global)
    fake_scores_out_local_16 = tfutil.autosummary('Loss_D/fake_scores_out_local_16', fake_scores_out_local_16)
    fake_scores_out_local_32 = tfutil.autosummary('Loss_D/fake_scores_out_local_32', fake_scores_out_local_32)
    fake_scores_out_local_64 = tfutil.autosummary('Loss_D/fake_scores_out_local_64', fake_scores_out_local_64)
    
    loss_local_16 = fake_scores_out_local_16 - real_scores_out_local_16 
    loss_local_32 = fake_scores_out_local_32 - real_scores_out_local_32 
    loss_local_64 = fake_scores_out_local_64 - real_scores_out_local_64 
    loss_global = fake_scores_out_global - real_scores_out_global     
    
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1, 1], 0.0, 1.0, dtype=fake_cubes_out.dtype)
        mixed_cubes_out = tfutil.lerp(tf.cast(reals_input, fake_cubes_out.dtype), fake_cubes_out, mixing_factors)
        mixed_scores_out_global, mixed_scores_out_local_16, mixed_scores_out_local_32, mixed_scores_out_local_64, mixed_labels_out = fp32(D.get_output_for(mixed_cubes_out, well_facies[:, 0:1], is_training=True))
        
        mixed_loss_local_16 = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_16))
        mixed_grads_local_16 = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_16, [mixed_cubes_out])[0]))
        mixed_norms_local_16 = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_16), axis=[1,2,3, 4]))
        gradient_penalty_local_16 = tf.square(mixed_norms_local_16 - wgan_target)
        loss_local_16 += gradient_penalty_local_16 * (wgan_lambda / (wgan_target**2))
        loss_local_16 = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_16', loss_local_16)
 
        mixed_loss_local_32 = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_32))
        mixed_grads_local_32 = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_32, [mixed_cubes_out])[0]))
        mixed_norms_local_32 = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_32), axis=[1,2,3, 4]))
        gradient_penalty_local_32 = tf.square(mixed_norms_local_32 - wgan_target)
        loss_local_32 += gradient_penalty_local_32 * (wgan_lambda / (wgan_target**2))
        loss_local_32 = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_32', loss_local_32)

        mixed_loss_local_64 = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_64))
        mixed_grads_local_64 = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_64, [mixed_cubes_out])[0]))
        mixed_norms_local_64 = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_64), axis=[1,2,3, 4]))
        gradient_penalty_local_64 = tf.square(mixed_norms_local_64 - wgan_target)
        loss_local_64 += gradient_penalty_local_64 * (wgan_lambda / (wgan_target**2))
        loss_local_64 = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_64', loss_local_64)

        mixed_loss_global = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_global))
        mixed_grads_global = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_global, [mixed_cubes_out])[0]))
        mixed_norms_global = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_global), axis=[1,2,3, 4]))
        gradient_penalty_global = tf.square(mixed_norms_global - wgan_target)
        loss_global += gradient_penalty_global * (wgan_lambda / (wgan_target**2))
        loss_global = tfutil.autosummary('Loss_D/WGAN_GP_loss_global', loss_global)
              
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty_local_16 = tfutil.autosummary('Loss_D/epsilon_penalty_local_16', tf.square(real_scores_out_local_16))
        loss_local_16 += epsilon_penalty_local_16 * wgan_epsilon
        
        epsilon_penalty_local_32 = tfutil.autosummary('Loss_D/epsilon_penalty_local_32', tf.square(real_scores_out_local_32))
        loss_local_32 += epsilon_penalty_local_32 * wgan_epsilon

        epsilon_penalty_local_64 = tfutil.autosummary('Loss_D/epsilon_penalty_local_64', tf.square(real_scores_out_local_64))
        loss_local_64 += epsilon_penalty_local_64 * wgan_epsilon

        epsilon_penalty_global = tfutil.autosummary('Loss_D/epsilon_penalty_global', tf.square(real_scores_out_global))
        loss_global += epsilon_penalty_global * wgan_epsilon
    
    loss = (loss_local_16 + 10 * loss_local_32 + 100 * loss_local_64) * local_weight + loss_global * global_weight 

    if cond_label:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.l2_loss(labels - real_labels_out)                            
            label_penalty_fakes = tf.nn.l2_loss(labels - fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss_D/label_penalty_fakes', label_penalty_fakes)
            loss += (label_penalty_reals + label_penalty_fakes) * label_weight
  
    loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss
