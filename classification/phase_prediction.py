import numpy as np
import os
import torch
import torchvision.models as models
# from utils import viterbi, buildTransitionMatrix, display_classification
from .utils import viterbi, buildTransitionMatrix, display_classification


def phase_prediction(model,im,im_test,feature_test,save_out,n_max,nclass,p_tr):

    ## Make prediction
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    im_ = torch.Tensor(im).view(im.shape[0],1,im.shape[1],im.shape[2]).cuda().to(device)
    proba = np.zeros((im.shape[0],nclass))
    proba_pred = np.zeros(im.shape[0])

    K = im.shape[0]//n_max
    c=0
    for k in range(K):
        proba_ = model(im_[c:c+n_max])
        proba_pred[c:c+n_max] = np.argmax(proba_.detach().cpu().numpy(),axis=1)
        proba[c:c+n_max,:] = proba_.detach().cpu().numpy()
        c += n_max

    if np.ceil(im.shape[0]/n_max)-K>0:
        proba_ = model(im_[c:])
        proba_pred[c:] = np.argmax(proba_.detach().cpu().numpy(),axis=1)
        proba[c:c+n_max,:] = proba_.detach().cpu().numpy()

    proba = proba/np.repeat(np.expand_dims(np.sum(proba,1),1),nclass,1)


    ## Viterbi
    mat = buildTransitionMatrix(p_tr)
    phase_pred = viterbi(proba, nclass, mat)
    # np.save(os.path.join('Data','phase_pred.npy'),phase_pred)

    ## Test
    if save_out:
        im_test_ = torch.Tensor(im_test).view(im_test.shape[0],1,im_test.shape[1],im_test.shape[2]).cuda().to(device)
        proba_test = np.zeros((im_test.shape[0],nclass))
        proba_true = np.zeros((im_test.shape[0],nclass))
        proba_pred_test = np.zeros(im_test.shape[0])
        proba_true_test = np.zeros(im_test.shape[0])

        K = im_test.shape[0]//n_max
        c=0
        for k in range(K):
            proba_ = model(im_test_[c:c+n_max])
            proba_pred_test[c:c+n_max] = np.argmax(proba_.detach().cpu().numpy(),axis=1)
            proba_true_test[c:c+n_max] = np.argmax(feature_test[c:c+n_max],axis=1)
            proba_test[c:c+n_max,:] = proba_.detach().cpu().numpy()
            for cc in range(c,c+n_max+1):
                proba_true[cc,int(proba_true_test[cc])]=1
            c += n_max

        if np.ceil(im_test.shape[0]/n_max)-K>0:
            proba_ = model(im_test_[c:])
            proba_pred_test[c:] = np.argmax(proba_.detach().cpu().numpy(),axis=1)
            proba_true_test[c:] = np.argmax(feature_test[c:],axis=1)
            proba_test[c:c+n_max,:] = proba_.detach().cpu().numpy()
            for cc in range(c,im_test.shape[0]):
                proba_true[cc,int(proba_true_test[cc])]=1
        proba = proba/np.repeat(np.expand_dims(np.sum(proba,1),1),nclass,1)

        ## Viterbi
        mat = buildTransitionMatrix(p_tr)
        phase_pred_test = viterbi(proba_test, nclass, mat)

        import ipdb; ipdb.set_trace()
        display_classification(im_test,proba_test,proba_true,leg=['G','early S','mid S','lateS'],save=True,name='visu',L=200)

        import pylab
        pylab.ion()
        if not os.path.exists(os.path.join('Data','Outputs_viterbi')):
            os.makedirs(os.path.join('Data','Outputs_viterbi'))	
        for i in range(im_test.shape[0]):
            pred = phase_pred_test[i]
            tr = proba_true_test[i]
            pylab.imshow(np.squeeze(im_test[i]))
            pylab.title('Prediction: '+ str(pred)+' -- True: '+str(tr))
            pylab.savefig(os.path.join('Data','Outputs_viterbi','test_'+str(i).zfill(5)+'.png'))

    return phase_pred, proba




def main():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    ## Parameters
    save_out = True
    n_max = 40
    nclass = 4
    p_tr = 0.05*np.ones(nclass)

    ## Load data
    im_train = np.load(os.path.join('..','Data','Classification','im_train.npy'))
    im_test = np.load(os.path.join('..','Data','Classification','im_test.npy'))
    feature_train = np.load(os.path.join('..','Data','Classification','feature_train.npy'))
    feature_test = np.load(os.path.join('..','Data','Classification','feature_test.npy'))
    
    ## Load neural network
    model = models.resnet50()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False) # For grayscale images
    model.avgpool=torch.nn.AvgPool2d(kernel_size=6, stride=1, padding=1)
    model.fc = torch.nn.Linear(2048,feature_train.shape[1],bias=True) # to adapt the output

    model = torch.nn.Sequential(
        model,
        torch.nn.Softmax(1)
    )
    model.load_state_dict(torch.load(os.path.join('..','Data','Classification','mytosis_cnn.pt')))
    model.to(device)
    # model = torch.load("mytosis_cnn.pt")
    # model.eval()
    # model.to(device)
    
    phase_pred, proba = phase_prediction(model,im_train,im_test,feature_test,save_out,n_max,nclass,p_tr)

if __name__ == '__main__':
    main()