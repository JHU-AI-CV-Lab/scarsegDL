# Libraries to Import 
import numpy as np 
import cv2 

def draw_contours(imgs,masks): 
    slices = imgs.shape[-1]
    
    contours = np.ndarray(shape = (imgs.shape[0],imgs.shape[1], 3, imgs.shape[2]), dtype= np.uint8) # RGB Images 
    
    contours[:,:,0,:] = imgs
    contours[:,:,1,:] = imgs
    contours[:,:,2,:] = imgs
    
    for kk in range(slices): 
        
        m1 = masks[:,:,kk]
        col_im1 = contours[:,:,:,kk]
        col_im1 = np.ascontiguousarray(col_im1, dtype=np.uint8)
        

        # Myocardium -> All scar is in the myocardium 
        
        myo = np.zeros(shape = (contours.shape[0],contours.shape[1]), dtype = np.uint8)
        myo[m1 >= 1] = 1
        contours1, hierarchy = cv2.findContours(myo,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img_myo = cv2.drawContours(col_im1, contours1, -1, (255,0,0), 1) 
        
        # Scar 
        scar = np.zeros(shape = (contours.shape[0],contours.shape[1]), dtype = np.uint8)
        scar[m1 == 2] = 1
        contours2, hierarchy = cv2.findContours(scar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img_both = cv2.drawContours(img_myo, contours2, -1, (0,0,255), 1) 
        
        contours[:,:,:,kk] = img_both 
    
    return contours 



def mydice(im1,im2):
    im1 = im1.flatten()
    im2 = im2.flatten()
    
    intersection = np.logical_and(im1,im2)
    out = intersection.sum()*2/(np.sum(im1) + np.sum(im2))
    
    return out 

def testing_main(savepath, model, test_imgs, test_masks,test_ids): 
    
    
    # Predict images one by one based for memory and gpu considerations
    # Check this, and if it doesn't work, then figure out something else 
    
    m = test_imgs.shape[0] 
    
    preds0 = np.ndarray(shape = test_masks.shape)
    for ii in range(m):
        preds0 = model.predict(test_imgs[ii:(ii+1),:,:,:,:], verbose = 0)
    
    preds = np.argmax(preds0, axis = 4) 
    
    # Calculate Dice Coefficients  
    my_dice = np.ndarray(shape = (preds.shape[0],),dtype = np.float32)
    
    for jj in range(preds.shape[0]):
        my_dice[jj] = mydice(test_masks[jj,:,:,:,0],preds[jj,:,:,:])
    
    dice_path = savepath+'dice_coeffs.npy'
    np.save(dice_path,my_dice) 
    
    return preds 


def testing_parallel(savepath, model, test_imgs, test_masks, gpus, classes, shift):
    
    # Predict in batches based on the number of gpus 
    test_imgs = np.expand_dims(test_imgs,axis=4)
    test_masks = np.expand_dims(test_masks,axis=4)
    
    m = test_imgs.shape[0]
    
    # Determine how many "extra images" to train on.. 
    rem = (m%gpus)
    
    m_new = m+rem 
    
    test_extra = np.ndarray(shape = (m_new, test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3], test_imgs.shape[4]))
    
    test_extra[0:m,:,:,:,:] = test_imgs
    test_extra[m:,:,:,:,:] = test_imgs[0:rem,:,:,:,:]
    
    M_end = int(m_new/gpus) 
    preds_extra = np.ndarray(shape = (m_new, test_masks.shape[1], test_masks.shape[2],test_masks.shape[3], classes))
    
    for ii in range(M_end): 
        pred_batch = model.predict(test_extra[2*ii:2*(ii+1),:,:,:,:], verbose = 0)
        preds_extra[2*ii:2*(ii+1),:,:,:,:] = pred_batch 
        
   
    preds0 = preds_extra[0:m,:,:,:,:] 
    preds = np.argmax(preds0, axis = 4) 
    
    # Calculate dice coefficients and generating contours 
    my_dice = np.ndarray(shape = (m,), dtype = np.float32)
    test_imgs = test_imgs*255 + shift 
    test_imgs = np.uint8(test_imgs)
    
    contours = np.ndarray(shape = (m,test_imgs.shape[1], test_imgs.shape[2], 3, test_imgs.shape[3]), dtype = np.uint8)
    for jj in range(m):
        my_dice[jj] = mydice(test_masks[jj,:,:,:,0], preds[jj,:,:,:])
        contours[jj,:,:,:,:] = draw_contours(test_imgs[jj,:,:,:,0],preds[jj,:,:,:])

    dice_path = savepath+'dice_coeffs.npy'
    np.save(dice_path, my_dice)
    np.save(savepath+'preds.npy',preds) 
    np.save(savepath+'contours.npy',contours) 
    
    return preds, contours  