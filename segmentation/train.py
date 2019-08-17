from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import pickle
import os
import numpy as np
import time

"""
Train model with generator.
Save weights in Data/Model during iteration. The final model is saved in Data/Model.
"""
def training_generator(model,generator,epochs=1,steps=50,save_name='default',generator_validation=[],step_decay=[]):
    t_start = time.time()
    if not os.path.exists(os.path.join(os.getcwd(),'Data','Information')):
        os.makedirs(os.path.join(os.getcwd(),'Data','Information'))
    if not os.path.exists(os.path.join(os.getcwd(),'Data','Model')):
        os.makedirs(os.path.join(os.getcwd(),'Data','Model'))

    checkpoint = ModelCheckpoint(os.getcwd()+'/Data/Model/'+save_name+'best.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
    # earlystopper = EarlyStopping(patience=50, verbose=1)
    tsboard = TensorBoard(log_dir='/tmp/tb')
    if step_decay ==[]:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.00001)
    else:
        reduce_lr = LearningRateScheduler(step_decay)
    if generator_validation!=[]:
        history=model.fit_generator(generator=generator,
                        validation_data=generator_validation,
                        validation_steps=20,
                        use_multiprocessing=True,
                        steps_per_epoch=steps,nb_epoch=epochs,verbose=1,
                        callbacks=[checkpoint,tsboard,reduce_lr])
    else:
        history=model.fit_generator(generator=generator,
                    use_multiprocessing=True,
                    steps_per_epoch=steps,nb_epoch=epochs,verbose=1,
                    callbacks=[checkpoint,tsboard,reduce_lr])

    elapsed_time = time.time() - t_start
    print("Training time: ",elapsed_time)
    with open(os.path.join(os.getcwd(),'Data','Information',save_name+'plots.p'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(os.path.join(os.getcwd(),'Data','Model',save_name+'.h5'))
    np.save(os.path.join(os.getcwd(),'Data','Model',save_name+'time.npy'),elapsed_time)
    return model

"""
Train model with array as dataset.
Save weights in Data/Model during iteration. The final model is saved in Data/Model.
"""
def train_model_array(model,x_train,y_train,batch_size=32,epochs=1,save_name='default',x_test=[],y_test=[],step_decay=[]):
    t_start = time.time()
    if not os.path.exists(os.path.join(os.getcwd(),'Data','Information')):
        os.makedirs(os.path.join(os.getcwd(),'Data','Information'))
    if not os.path.exists(os.path.join(os.getcwd(),'Data','Model')):
        os.makedirs(os.path.join(os.getcwd(),'Data','Model'))
    if step_decay ==[]:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.00001)
    else:
        reduce_lr = LearningRateScheduler(step_decay)

    checkpoint = ModelCheckpoint(os.getcwd()+'/Data/Information/'+save_name+'best.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
    if x_test!=[]:
        history=model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,
          callbacks=[checkpoint,TensorBoard(log_dir='/tmp/tb'),reduce_lr],
          validation_data=(x_test,y_test))
    else:
        history=model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,
          callbacks=[checkpoint,TensorBoard(log_dir='/tmp/tb'),reduce_lr])
    # ,LearningRateScheduler(step_decay)])
    
    elapsed_time = time.time() - t_start
    print("Training time: ",elapsed_time)
    with open(os.path.join(os.getcwd(),'Data','Information',save_name+'plots.p'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(os.path.join(os.getcwd(),'Data','Model',save_name+'.h5'))
    np.save(os.path.join(os.getcwd(),'Data','Model',save_name+'time.npy'),elapsed_time)
    return model
