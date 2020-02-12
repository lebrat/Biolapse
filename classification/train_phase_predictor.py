import numpy as np
import imageio
import os
import torch
import torch.optim as optim
from utils import Net, train
import torchvision.models as models
import utils

"""
train neural network to classify image into features.
"""
def train_phase_predictor(im_train,feature_train,im_test,feature_test,save_out=True,epochs=20,nbatch=30):

    # im_train = torch.Tensor(im_train).view(im_train.shape[0],1,im_train.shape[1],im_train.shape[2]).cuda()
    # im_test = torch.Tensor(im_test).view(im_test.shape[0],1,im_test.shape[1],im_test.shape[2]).cuda()
    # feature_train = torch.Tensor(feature_train).cuda()
    # feature_test = torch.Tensor(feature_test).cuda()


    generator_train=utils.CropGenertor(im_train,feature_train)
    generator_test=utils.CropGenertor(im_test,feature_test)
    train_loader = torch.utils.data.DataLoader(dataset=generator_train, batch_size=nbatch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=generator_test, batch_size=nbatch, shuffle=True)

    ## train neural network
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    # alldata_train = im_train, feature_train
    # alldata_test = im_test, feature_test

    # model = Net().to(device)
    model = models.resnet50()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False) # For grayscale images
    model.avgpool=torch.nn.AvgPool2d(kernel_size=6, stride=1, padding=1)
    model.fc = torch.nn.Linear(2048,feature_train.shape[1],bias=True) # to adapt the output

    model = torch.nn.Sequential(
        model,
        torch.nn.Softmax(1)
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        loss_tr, loss_te = train(model, device, train_loader, test_loader, optimizer, epoch, nbatch=nbatch)
        loss_train[epoch-1] = loss_tr
        loss_test[epoch-1] = loss_te
    # torch.save(model, os.path.join('..','Data','Classification','mytosis_cnn.pt'))
    torch.save(model.state_dict(), os.path.join('..','Data','Classification','mytosis_cnn.pt'))

    ## test prediction
    from torch.autograd import Variable
    for image_test, label_test in test_loader:
        if use_cuda:
            image_test = Variable(image_test).type(torch.float32).cuda()
        else:
            image_test = Variable(image_test).type(torch.float32)
        output_test = model(image_test)
        # if use_cuda:
        #     loss = nn.BCELoss()(output, Variable(label_test.float()).cuda())
        # else:
        #     loss = nn.BCELoss()(output, Variable(label_test.float()))
        break

    # randi = np.random.randint(0,im_test.shape[0],size=im_test.shape[0])
    # dataloc_test, targetloc_test = im_test[randi,:], feature_test[randi,:]
    # dataloc_test.to(device)
    # targetloc_test.to(device)
    # output_test = model(dataloc_test)

    dataloc_test = image_test.detach().cpu().numpy()
    output_test = output_test.detach().cpu().numpy()
    targetloc_test = label_test.detach().cpu().numpy()
    if save_out:
        import pylab
        pylab.ion()
        if not os.path.exists(os.path.join('Data','Outputs_nn')):
            os.makedirs(os.path.join('Data','Outputs_nn'))	
        for i in range(targetloc_test.shape[0]):
            pred = np.argmax(output_test[i])
            tr = np.argmax(targetloc_test[i])
            pylab.imshow(np.squeeze(dataloc_test[i]))
            pylab.title('Prediction: '+ str(pred)+' -- True: '+str(tr))
            pylab.savefig(os.path.join('Data','Outputs_nn','test_'+str(i).zfill(5)+'.png'))

    return loss_train, loss_test


def main():
    ## Load data
    im_train = np.load(os.path.join('Data','im_train.npy'))
    im_test = np.load(os.path.join('Data','im_test.npy'))
    feature_train = np.load(os.path.join('Data','feature_train.npy'))
    feature_test = np.load(os.path.join('Data','feature_test.npy'))
    save_out = True
    
    train_phase_predictor(im_train,feature_train,im_test,feature_test,save_out)


if __name__ == '__main__':
    main()