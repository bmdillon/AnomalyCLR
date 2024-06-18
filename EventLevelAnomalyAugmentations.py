import numpy as np

#################################################
#################################################
###  PHYSICAL AUGMENTATIONS
#################################################
#################################################

def rotate_events(event, scaler=None):
    event_rot = event.copy() # changed
    phi = event[:, 2, :]
    particle = event[:, 3] + event[:, 4] + event[:, 5] + event[:, 6]
    rot_angle = np.random.rand(event.shape[0]) *2*np.pi
    rot_angle = rot_angle - np.pi
    ones  = np.ones_like(phi)
    angle_matrix = np.einsum('ij,i->ij', ones, rot_angle)
    angle_matrix[particle == 0] = 0 
    if scaler != None:
        angle_matrix = angle_matrix / scaler
    phi = phi[:, :]+angle_matrix
    if scaler != None:
        phi = np.where(phi>1, phi - 2, phi) 
        phi = np.where(phi<-1, phi +2, phi) 
    else: 
        phi = np.where(phi>np.pi, phi - 2*np.pi, phi)
        phi = np.where(phi<-np.pi, phi+2*np.pi, phi)
    event_rot[:, 2, :] = phi 
    return event_rot

def energy_smear_jets(batch, scaler_pt, strength=1.0):
    batch_esmeared = batch.copy()
    pT = batch[:, 0, 9:]
    mask = pT > 0
    if scaler_pt != None:
        pTfunc = np.sqrt( 0.052*scaler_pt*pT**2 + 1.502*scaler_pt*pT )
        shift_pt = np.nan_to_num( strength * pTfunc * np.random.randn( pT.shape[0], pT.shape[1] ), posinf = 0., neginf = 0.) * mask
        batch_esmeared[:, 0, 9:] += shift_pt/scaler_pt
    else: 
        pTfunc = np.sqrt( 0.052*pT**2 + 1.502*pT )
        shift_pt = np.nan_to_num( strength * pTfunc * np.random.randn( pT.shape[0], pT.shape[1] ), posinf = 0., neginf = 0.) * mask
        batch_esmeared[:, 0, 9:] += shift_pt
    lz = batch_esmeared[:, 0, 9:] < 0.
    batch_esmeared[:, 0, 9:][lz] = 0.0
    batch_esmeared[:, 6, 9:][lz] = 0.0
    return batch_esmeared

def get_std_rivet(pTs, scaler_pt, A=0.028, B=25, C=0.1):
    #  standard deviation for the Rivet detector simulation
    mask = (pTs > 0)
    np_sett_dict = np.seterr(over = 'ignore')
    if scaler_pt != None:
        std_rivet  = A/(1+np.exp( ( (pTs *scaler_pt) -B)/C) )
    else: 
        std_rivet  = A/(1+np.exp( ( pTs -B)/C) )
    std_rivet[~mask] = 0
    np.seterr(over = np_sett_dict['over'])
    return std_rivet

def etaphi_smear_events(batch, scaler_pt, scale_angle, strength=1.0, ):
    batch_distorted = batch.copy()
    std = get_std_rivet( batch_distorted[:,0, 1:], scaler_pt )
    noise_eta = np.random.normal( loc=0.0, scale=strength*std )
    noise_phi = np.random.normal( loc=0.0, scale=strength*std )
    noise     = np.stack( [noise_eta, noise_phi], axis=1 )
    batch_distorted[:,1:3,1:] += noise
    if scale_angle:
        batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]>1, batch_distorted[:, 2, 1:]-2, batch_distorted[:, 2, 1:]) 
        batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]< -1, batch_distorted[:, 2, 1:]+2, batch_distorted[:, 2, 1:]) 
        crosses_upper_bound_e = batch_distorted[:, 1, 1:5] > (3./4)
        crosses_lower_bound_e = batch_distorted[:, 1, 1:5] < (-3./4.)
        crosses_e = crosses_lower_bound_e | crosses_upper_bound_e
        crosses_upper_bound_mu = batch_distorted[:, 1, 5:9] > (2.1/4.)
        crosses_lower_bound_mu = batch_distorted[:, 1, 5:9] < (-2.1/4.)
        crosses_mu = crosses_lower_bound_mu | crosses_upper_bound_mu
        crosses_upper_bound_jet = batch_distorted[:, 1, 9:] > 1.
        crosses_lower_bound_jet = batch_distorted[:, 1, 9:] < -1.
        crosses_jet = crosses_lower_bound_jet | crosses_upper_bound_jet
        for i in range( np.shape(batch_distorted)[1] ):
            batch_distorted[:, i, 1:5][crosses_e] = 0.
            batch_distorted[:, i, 5:9][crosses_mu] = 0.
            batch_distorted[:, i, 9:][crosses_jet] = 0.
    else: 
        batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]>np.pi, batch_distorted[:, 2, 1:]-np.pi, batch_distorted[:, 2, 1:]) 
        batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]< -np.pi, batch_distorted[:, 2, 1:]+np.pi, batch_distorted[:, 2, 1:]) 
        crosses_upper_bound_e = batch_distorted[:, 1, 1:5] > 3.
        crosses_lower_bound_e = batch_distorted[:, 1, 1:5] < -3.
        crosses_e = crosses_lower_bound_e | crosses_upper_bound_e
        crosses_upper_bound_mu = batch_distorted[:, 1, 5:9] > 2.1
        crosses_lower_bound_mu = batch_distorted[:, 1, 5:9] < -2.1
        crosses_mu = crosses_lower_bound_mu | crosses_upper_bound_mu
        crosses_upper_bound_jet = batch_distorted[:, 1, 9:] > 4.
        crosses_lower_bound_jet = batch_distorted[:, 1, 9:] < -4.
        crosses_jet = crosses_lower_bound_jet | crosses_upper_bound_jet
        for i in range( np.shape(batch_distorted)[1] ):
            batch_distorted[:, i, 1:5][crosses_e] = 0.
            batch_distorted[:, i, 5:9][crosses_mu] = 0.
            batch_distorted[:, i, 9:][crosses_jet] = 0.
    return batch_distorted

def apply_sin(batch_inp, scale_angle):
    # returns an array with shape (batch_size, 8, 19), where phi is split into sin(phi) [2] and cos(phi) [3], the rest is left unchanged 
    batch = batch_inp.copy()
    batch_size = len(batch)
    new_batch = np.ones( (batch_size, 8, 19) )
    splitt = np.split(batch, [2, 3], axis=1)
    new_batch[:, :2, : ] = splitt[0]
    phi = splitt[1]
    phi = phi.reshape((batch_size, 19))
    if scale_angle:
        phi = phi * np.pi
    new_batch[:, 4:, :] = splitt[2]
    no_phi = phi == 0.
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)
    phi_sin[no_phi] = 0.
    phi_cos[no_phi] = 0.
    new_batch[:, 2, :] = phi_sin
    new_batch[:, 3, :] = phi_cos
    return new_batch 

#################################################
#################################################
###  ANOMALY AUGMENTATIONS
#################################################
#################################################

def collinear_fill_e_mu(batch):
    batch_filled = batch.copy()
    # ELECTRONS
    n_constit = 4
    n_nonzero = np.count_nonzero(batch_filled[:, 4, 1:5], axis=1)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = batch_filled[:, 4, 1:5] != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*batch_filled[:, 0, 1:5]
    b = (1.0-r_split) * mask_split*batch_filled[:, 0, 1:5]
    c =                ~mask_split*batch_filled[:, 0, 1:5]
    batch_filled[:, 0, 1:5] = a + c + np.flip(b, axis=1)
    batch_filled[:, 1, 1:5] += np.flip(mask_split*batch_filled[:, 1, 1:5], axis=1)
    batch_filled[:, 2, 1:5] += np.flip(mask_split*batch_filled[:, 2, 1:5], axis=1)
    batch_filled[:, 4, 1:5] += np.flip(mask_split*batch_filled[:, 4, 1:5], axis=1)
    # MUONS
    n_constit = 4
    n_nonzero = np.count_nonzero(batch_filled[:, 5, 5:9], axis=1)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = batch_filled[:, 5, 5:9] != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*batch_filled[:, 0, 5:9]
    b = (1.0-r_split) * mask_split*batch_filled[:, 0, 5:9]
    c =                ~mask_split*batch_filled[:, 0, 5:9]
    batch_filled[:, 0, 5:9] = a + c + np.flip(b, axis=1)
    batch_filled[:, 1, 5:9] += np.flip(mask_split*batch_filled[:, 1, 5:9], axis=1)
    batch_filled[:, 2, 5:9] += np.flip(mask_split*batch_filled[:, 2, 5:9], axis=1)
    batch_filled[:, 5, 5:9] += np.flip(mask_split*batch_filled[:, 5, 5:9], axis=1)
    return batch_filled

def collinear_fill_e_mu_v2(batch, scaler_pt):
    test_batch_c = batch.copy()
    # ELECTRONS
    pt_led = 23/scaler_pt
    pt_min = 3/scaler_pt
    n_constit = 4
    
    mask_split = test_batch_c[:, 4, 1:5] != 0 # loc og n_non_zero
    n_nonzero = mask_split.sum(-1)
    n_allowed = np.maximum(0, ((test_batch_c[:,0, 1:5].sum(-1)-pt_led)/pt_min) - n_nonzero)
    n_adds = np.random.randint(0, high=np.trunc(n_allowed)+1)
    n_adds = np.minimum(n_adds, mask_split.sum(-1), n_constit-mask_split.sum(-1))
    pt_avail = np.maximum(0, test_batch_c[:,0, 1:5].sum(-1) - (pt_led-pt_min) - pt_min*(n_adds+n_nonzero))

    test_batch_c[:, 4, 1:5] = mask_split +  np.flip(mask_split, axis=1)

    row_indices = np.arange(len(test_batch_c)).reshape(-1,1)
    col_indices = np.arange(test_batch_c[:,0,1:5].shape[-1])
    bool_mask = np.flip((col_indices < n_adds[:, None]), axis=-1)
    mask_split |= bool_mask

    sample = np.random.gamma(test_batch_c[:, 4, 1:5].astype(int)*mask_split,1)
    sample_sum = np.sum(sample, axis=1, keepdims=True)
    sample = np.divide(sample, sample_sum, out=np.zeros_like(sample), where=sample_sum!=0)

    test_batch_c[:,0,1:5] = ((test_batch_c[:, 4, 1:5]*pt_avail.reshape(-1,1)*sample + pt_min)*mask_split)
    test_batch_c[:,0,1] += np.maximum(0, (pt_led - pt_min))*mask_split[:,0]
    test_batch_c[:, 1, 1:5] += np.flip(mask_split*test_batch_c[:, 1, 1:5], axis=1)
    test_batch_c[:, 2, 1:5] += np.flip(mask_split*test_batch_c[:, 2, 1:5], axis=1)
    test_batch_c[:, 4, 1:5] *= mask_split

    #MUONS
    mask_split = test_batch_c[:, 5, 5:9] != 0 # loc og n_non_zero
    n_nonzero = mask_split.sum(-1)
    n_allowed = np.maximum(0, ((test_batch_c[:,0, 5:9].sum(-1)-pt_led)/pt_min) - n_nonzero)
    n_adds = np.random.randint(0, high=np.trunc(n_allowed)+1)
    n_adds = np.minimum(n_adds, mask_split.sum(-1), n_constit-mask_split.sum(-1))
    pt_avail = np.maximum(0, test_batch_c[:,0, 5:9].sum(-1) - (pt_led-pt_min) - pt_min*(n_adds+n_nonzero))

    test_batch_c[:, 5, 5:9] = mask_split +  np.flip(mask_split, axis=1)

    row_indices = np.arange(len(test_batch_c)).reshape(-1,1)
    col_indices = np.arange(test_batch_c[:,0,5:9].shape[-1])
    bool_mask = np.flip((col_indices < n_adds[:, None]), axis=-1)
    mask_split |= bool_mask

    sample = np.random.gamma(test_batch_c[:, 5, 5:9].astype(int)*mask_split,1)
    sample_sum = np.sum(sample, axis=1, keepdims=True)
    sample = np.divide(sample, sample_sum, out=np.zeros_like(sample), where=sample_sum!=0)

    test_batch_c[:,0,5:9] = ((test_batch_c[:, 5, 5:9]*pt_avail.reshape(-1,1)*sample + pt_min)*mask_split)
    test_batch_c[:,0,5] += np.maximum(0, (pt_led - pt_min))*mask_split[:,0]
    test_batch_c[:, 1, 5:9] += np.flip(mask_split*test_batch_c[:, 1, 5:9], axis=1)
    test_batch_c[:, 2, 5:9] += np.flip(mask_split*test_batch_c[:, 2, 5:9], axis=1)
    test_batch_c[:, 5, 5:9] *= mask_split
    return test_batch_c

def collinear_fill_jets (batch):
    batch_filled = batch.copy()
    n_constit = 10
    n_nonzero = np.count_nonzero(batch_filled[:, 6, 9:], axis=1)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = batch_filled[:, 6, 9:] != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*batch_filled[:, 0, 9:]
    b = (1.0-r_split) * mask_split*batch_filled[:, 0, 9:]
    c =                ~mask_split*batch_filled[:, 0, 9:]
    batch_filled[:, 0, 9:] = a + c + np.flip(b, axis=1)
    batch_filled[:, 1, 9:] += np.flip(mask_split*batch_filled[:, 1, 9:], axis=1)
    batch_filled[:, 2, 9:] += np.flip(mask_split*batch_filled[:, 2, 9:], axis=1)
    batch_filled[:, 6, 9:] += np.flip(mask_split*batch_filled[:, 6, 9:], axis=1)
    return batch_filled

def collinear_fill_jets_v2 (batch, scaler_pt):
    test_batch_c = batch.copy()
    pt_led = 0
    pt_min = 15/scaler_pt
    n_constit = 10
    mask_split = test_batch_c[:, 6, 9:] != 0 # loc og n_non_zero
    n_nonzero = mask_split.sum(-1)
    n_allowed = np.maximum(0, ((test_batch_c[:,0, 9:].sum(-1)-pt_led)/pt_min) - n_nonzero)
    n_adds = np.random.randint(0, high=np.trunc(n_allowed)+1)
    n_adds = np.minimum(n_adds, mask_split.sum(-1), n_constit-mask_split.sum(-1))
    pt_avail = np.maximum(0, test_batch_c[:,0, 9:].sum(-1) - (pt_led - pt_min) - pt_min*(n_adds+n_nonzero))

    test_batch_c[:, 6, 9:] = mask_split +  np.flip(mask_split, axis=1)

    row_indices = np.arange(len(test_batch_c)).reshape(-1,1)
    col_indices = np.arange(test_batch_c[:,0,9:].shape[-1])
    bool_mask = np.flip((col_indices < n_adds[:, None]), axis=-1)
    mask_split |= bool_mask

    sample = np.random.gamma(test_batch_c[:, 6, 9:].astype(int)*mask_split,1)
    sample_sum = np.sum(sample, axis=1, keepdims=True)
    sample = np.divide(sample, sample_sum, out=np.zeros_like(sample), where=sample_sum!=0)

    test_batch_c[:,0,9:] = ((test_batch_c[:, 6, 9:]*pt_avail.reshape(-1,1)*sample + pt_min)*mask_split)
    test_batch_c[:,0,9] += np.maximum(0, (pt_led - pt_min))
    test_batch_c[:, 1, 9:] += np.flip(mask_split*test_batch_c[:, 1, 9:], axis=1)
    test_batch_c[:, 2, 9:] += np.flip(mask_split*test_batch_c[:, 2, 9:], axis=1)
    test_batch_c[:, 6, 9:] *= mask_split
    return test_batch_c

def add_objects (batch, scaler_pt=1., scale_angle=False):
    batch_filled = batch.copy()
    n_els = 4
    n_mus = 4
    n_jets = 10
    n_nonzero_els = np.count_nonzero(batch_filled[:, 4, 1:5], axis=1)
    n_nonzero_mus = np.count_nonzero(batch_filled[:, 5, 5:9], axis=1)
    n_nonzero_jets = np.count_nonzero(batch_filled[:, 6, 9:], axis=1)
    n_new_els = np.random.randint( 0, high=4-n_nonzero_els+1 )
    n_new_mus = np.random.randint( 0, high=4-n_nonzero_mus+1 )
    n_new_jets = np.random.randint( 0, high=10-n_nonzero_jets+1 )
    maxpts = np.max( batch[:,0,:], axis=-1 )
    for n in range( batch_filled.shape[0] ):
        # electrons
        el_pts = np.expand_dims( 3.0 + (maxpts[n]-3.0) * np.random.rand( n_new_els[n] ), axis=1 )
        el_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
        el_etas = np.expand_dims( 2*3 * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
        el_one_hot = np.concatenate( [np.zeros(shape=(n_new_els[n],1)), np.ones(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1))], axis=1 ) 
        els = np.concatenate( [el_pts, el_etas, el_phis, el_one_hot], axis=1 )
        el_start = 1 + n_nonzero_els[n]
        el_end = 1 + n_nonzero_els[n] + n_new_els[n]
        batch_filled[n,:,el_start:el_end] = np.transpose( els )
        # muons
        mu_pts = np.expand_dims( 3.0 + (maxpts[n]-3.0) * np.random.rand( n_new_mus[n] ), axis=1 )
        mu_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
        mu_etas = np.expand_dims( 2*2.1 * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
        mu_one_hot = np.concatenate( [np.zeros(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1)), np.ones(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1))], axis=1 ) 
        mus = np.concatenate( [mu_pts, mu_etas, mu_phis, mu_one_hot], axis=1 )
        mu_start = 5 + n_nonzero_mus[n]
        mu_end = 5 + n_nonzero_mus[n] + n_new_mus[n]
        batch_filled[n,:,mu_start:mu_end] = np.transpose( mus )
        # jets
        jet_pts = np.expand_dims( 15.0 + (maxpts[n]-15.0) * np.random.rand( n_new_jets[n] ), axis=1 )
        jet_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
        jet_etas = np.expand_dims( 2*4 * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
        jet_one_hot = np.concatenate( [np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.ones(shape=(n_new_jets[n],1))], axis=1 ) 
        jets = np.concatenate( [jet_pts, jet_etas, jet_phis, jet_one_hot], axis=1 )
        jet_start = 9 + n_nonzero_jets[n]
        jet_end = 9 + n_nonzero_jets[n] + n_new_jets[n]
        batch_filled[n,:,jet_start:jet_end] = np.transpose( jets )
        # MET
        old_met_pt = batch_filled[n,0,0]
        old_met_phi = batch_filled[n,2,0]
        old_met = np.array( [ old_met_pt * np.sin(old_met_phi), old_met_pt * np.cos(old_met_phi) ] )
        new_obj = np.concatenate( [ els[:,0:3], mus[:,0:3], jets[:,0:3] ], axis=0 )
        new_met = old_met - np.array( [ new_obj[:,0] * np.sin(new_obj[:,2]), new_obj[:,0] * np.cos(new_obj[:,2]) ] ).sum(axis=-1)
        new_met_pt = np.sqrt( new_met[0]**2 + new_met[1]**2 )
        if new_met[1]<0. and new_met[0]>0.:
            new_met_phi =  np.pi - np.arcsin( new_met[0]/new_met_pt )
        elif new_met[1]<0. and new_met[0]<0.:
            new_met_phi = -np.pi - np.arcsin( new_met[0]/new_met_pt )
        else:
            new_met_phi = np.arcsin( new_met[0]/new_met_pt )
        batch_filled[n,0,0] = new_met_pt / scaler_pt
        if scale_angle:
            batch_filled[n,2,0] = new_met_phi / np.pi
    return batch_filled

def add_objects_wsmear(batch, scaler_pt, scale_angle, etaphi_smear_strength):
    batch_filled = batch.copy()
    batch_filled = add_objects( batch_filled, scaler_pt, scale_angle )
    batch_filled = etaphi_smear_events( batch_filled, scaler_pt, scale_angle, strength=etaphi_smear_strength )
    return batch_filled

def add_objects_constptmet( batch, scaler_pt, scale_angle, etaphi_smear_strength ):
    batch_filled = batch.copy()
    batch_filled = collinear_fill_jets_v2( batch_filled, scaler_pt)
    batch_filled = collinear_fill_e_mu_v2( batch_filled, scaler_pt )
    batch_filled = etaphi_smear_events( batch_filled, scaler_pt, scale_angle, strength=etaphi_smear_strength )
    return batch_filled

def shift_met_or_pt( batch ):
    batch_shifted = batch.copy()
    rands = np.random.randint( low=0, high=3, size=batch_shifted.shape[0] )
    shifts =  1.0 + np.random.rand( batch_shifted.shape[0] ) * 4.0
    shifts_met =  0.5 + np.random.rand( batch_shifted.shape[0] ) * 4.5
    batch_shifted[np.where(rands==0),0,0 ] *= shifts_met[np.where(rands==0)]
    batch_shifted[np.where(rands==1),0,1:] *= np.expand_dims( shifts[np.where(rands==1)], axis=-1 )
    batch_shifted[np.where(rands==2),0,: ] *= np.expand_dims( shifts[np.where(rands==2)], axis=-1 )
    return batch_shifted


def neg_augs( batch, scaler_pt, scale_angle, etaphi_smear_strength, addobj=True, addobj_wcpm=True, shpt=True, shmet=True, shporm=False ):
    batch_aug = batch.copy()
    n_augs = 0
    aug_list = []
    if addobj: n_augs+=1; aug_list.append("ao")
    if addobj_wcpm: n_augs+=1; aug_list.append("aowcpm")
    if shporm: n_augs+=1; aug_list.append("spm")
    rands = np.random.randint( low=0, high=n_augs, size=batch_aug.shape[0] )
    rand_opts = range( n_augs )
    for j in range( n_augs ):
        aug = aug_list[j]
        n = rand_opts[j]
        if aug=="ao":       batch_aug[ np.where(rands==n) ] = add_objects_wsmear( batch_aug[ np.where(rands==n) ], scaler_pt, scale_angle, etaphi_smear_strength)
        if aug=="aowcpm":   batch_aug[ np.where(rands==n) ] = add_objects_constptmet( batch_aug[ np.where(rands==n) ], scaler_pt, scale_angle, etaphi_smear_strength )
        if aug=="spm":      batch_aug[ np.where(rands==n) ] = shift_met_or_pt( batch_aug[ np.where(rands==n) ] )
    return batch_aug
