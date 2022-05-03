project_name = 'wgan_face_generation'

import os
import pandas as pd
from scipy import spatial
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#%matplotlib inline

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break




def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



import torch.nn as nn

discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    # Modification 1: remove sigmoid
    # nn.Sigmoid()

)



latent_size = 64

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)




def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    #print(real_images.size())
    #print(real_images[0])
    #real_preds = discriminator(torch.from_numpy(real_images.cpu().detach().numpy()))
    real_images=real_images.float()
    real_preds = discriminator(real_images)
    # modification: remove binary cross entropy
    # real_targets = torch.ones(real_images.size(0), 1, device=device)
    # real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_loss = -torch.mean(real_preds)

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    # modification: remove binary cross entropy
    # fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    # fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_loss = torch.mean(fake_preds)

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_loss.item(), fake_loss.item()


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    # modificationL remove binary cross entropy
    # targets = torch.ones(batch_size, 1, device=device)
    loss = -torch.mean(preds)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()

from torchvision.utils import save_image
eud=[]
cos_similarity=[]
ssim_scores=[]
mses=[]
mfids=[]
import cv2
def save_samples(index, latent_tensors, show=True):
    cos_sim_per_epoch=0
    dist_per_epoch=0
    ssim_score_per_epoch=0
    mse_per_epoch=0
    fid_per_epoch=0
    #if index==200:
    #    fake_no=600
    #else:
    fake_no=1
    for j in range(fake_no):
        #fixed_latent[0][j] = 1
        latent_tensors = torch.randn(1, latent_size, 1, 1, device=device)
        fake_images = generator(latent_tensors)
        #fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        for real_images in tqdm(train_dl):
            cos_sim,dist,ssim_score,mse,mfid=get_eud_and_cossim(real_images,fake_images[0][0].cpu().detach().numpy())
            ssim_score_per_epoch=ssim_score_per_epoch+ssim_score
            mse_per_epoch=mse_per_epoch+mse
            #cos_sim_per_epoch = cos_sim_per_epoch+cos_sim
            #dist_per_epoch = dist_per_epoch+dist
            #fid_per_epoch=fid_per_epoch+mfid
        #if index==200:
        #fake_fname = f'2-generated-csv-{str(index).zfill(5)}-{str(j).zfill(5)}.csv'
            #save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        #gasf_csv = pd.DataFrame(data=fake_images[0][0].cpu().detach().numpy())
            #mypath = os.path.abspath("C:/Users/Nirho/Desktop/gasf_csv_64/Youtube/")
        #gasf_csv.to_csv(os.path.join(sample_dir, fake_fname), header=False, index=False)
        #print('Saving', fake_fname)
        #fixed_latent[0][j] = 0
        #if index==199:
        #    cv2.imwrite(f'0-generated-csv-{str(index).zfill(5)}-{str(j).zfill(5)}.png', fake_images[0][0].cpu().detach().numpy())
        #if show:
        #    fig, ax = plt.subplots(figsize=(8, 8))
        #    ax.set_xticks([]); ax.set_yticks([])
         #   ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
    if index%5==0:
        sspe = ssim_score_per_epoch / fake_no
        ssim_scores.append(ssim_score_per_epoch / fake_no)
        # cos_similarity.append(cos_sim_per_epoch/fake_no)
        # eud.append(dist_per_epoch/fake_no)
        mspe = mse / fake_no
        mses.append(mse / fake_no)
        #fidpe = fid_per_epoch / fake_no
        #mfids.append(fid_per_epoch / fake_no)
        if sspe>0.5 and mspe<0.07:
            fake_no = 400
            print("founnd")
            for j in range(fake_no):
                latent_tensors = torch.randn(1, latent_size, 1, 1, device=device)
                fake_images = generator(latent_tensors)

                fake_fname = f'2-generated-csv-{str(index).zfill(5)}-{str(j).zfill(5)}.csv'
                # save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
                gasf_csv = pd.DataFrame(data=fake_images[0][0].cpu().detach().numpy())
                # mypath = os.path.abspath("C:/Users/Nirho/Desktop/gasf_csv_64/Youtube/")
                gasf_csv.to_csv(os.path.join(sample_dir, fake_fname), header=False, index=False)
                print('Saving', fake_fname)
            print("I'm done")


def get_eud_and_cossim(real,fake):
    cos_sim=0
    dist=0
    ssim_score=0
    mse=0
    mfid=0
    for j in range(len(real)):
        (score, diff) = compare_ssim(real[j].cpu().detach().numpy()[0], fake, full=True)
        diff = (diff * 255).astype("uint8")
        ssim_score=ssim_score+score
        err=np.mean((real[j].cpu().detach().numpy()[0]-fake)**2)
        mse=mse+err
        #print("SSIM: {}".format(score))
        pts = []
        # print(len(data))
        #for i in range(len(real[j].cpu().detach().numpy()[0])):
        #    pts.append(real[j].cpu().detach().numpy()[0][i][i])
        #dataset = []
        #for k in pts:
        #    dataset.append((np.sqrt((k + 1) / 2)))

        #pts2 = []
        # print(len(data))
        #for i in range(len(fake)):
        #    pts2.append(fake[i][i])
        #dataset2 = []
        #for k in pts2:
        #    dataset2.append((np.sqrt((k + 1) / 2)))
        #print(real[j].cpu().detach().numpy()[0])
        #cos_sim1 = 1 - spatial.distance.cosine(dataset, dataset2)
        #dist1 = np.sqrt(np.sum(np.square(np.array(dataset) - np.array(dataset2))))

        #cos_sim=cos_sim+cos_sim1
        #dist=dist+dist1
        # calculate mean and covariance statistics
        #mu1, sigma1 = real[j].cpu().detach().numpy()[0].mean(axis=0), cov(real[j].cpu().detach().numpy()[0], rowvar=False)
        #mu2, sigma2 = fake.mean(axis=0), cov(fake, rowvar=False)
        # calculate sum squared difference between means
        #ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        #covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        #if iscomplexobj(covmean):
        #    covmean = covmean.real
        # calculate score
        #fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        #mfid=mfid+fid

    ssim_score=ssim_score/len(real)
    #cos_sim=cos_sim/len(real)
    #dist=dist/len(real)
    mse=mse/len(real)
    #mfid=mfid/len(real)
    return cos_sim,dist,ssim_score,mse,mfid



import torch.nn.functional as F


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=lr)

    for epoch in range(epochs):
        #for real_images,_ in tqdm(train_dl):
        #print(train_dl)
        for real_images in tqdm(train_dl):
            # Train discriminator
            # modification: clip param for discriminator
            for parm in discriminator.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        # Save generated images
        save_samples(epoch + start_idx, fixed_latent, show=False)
        #if epoch % 50 == 0:
        #    torch.save(generator.state_dict(), 'Gvid1net' + str(epoch) + '.pth')
        #    torch.save(discriminator.state_dict(), 'Dvid1net' + str(epoch) + '.pth')

    return losses_g, losses_d, real_scores, fake_scores


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(
       # description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    #parser.add_argument('--path', type=str, default="gasf_pm_vid1",
                   #     help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    #DATA_DIR = 'pretrain_gasf'
    DATA_DIR = 'gasf_20_train/Youtube/vid20'

    print(os.listdir(DATA_DIR))

    print(os.listdir(DATA_DIR + '/')[:10])
    image_size = 125
    batch_size = 3
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    #train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    #    T.Resize(image_size),
    #    T.CenterCrop(image_size),
    #    T.ToTensor(),
    #    T.Normalize(*stats)]))
    train_ds=[]
    #for subdir, dirs, files in os.walk(DATA_DIR):
    for files in os.scandir(DATA_DIR):
        for file in os.scandir(files):
            #print( file)
            df = pd.read_csv(file,header=None).values
            df = 0.5 * df + 0.5
            gamma = 0.25
            A = 1
            df = A * np.power(df, gamma)
            #df = df.replace(np.nan, 0)
            #data = df.to_numpy(dtype='float')
            #print(data)
            data=torch.from_numpy(np.array([df]))
            #T.Normalize(data)
            train_ds.append(data)
    print(len(train_ds))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

    #show_batch(train_dl)

    device = get_default_device()
    # device

    train_dl = DeviceDataLoader(train_dl, device)


    discriminator = to_device(discriminator, device)

    discriminator.apply(weight_init)
    generator.apply(weight_init)

    generator.load_state_dict(torch.load('weights20/G20yt600-gm.pth'))
    discriminator.load_state_dict(torch.load('weights20/D20yt600-gm.pth'))

    xb = torch.randn(batch_size, latent_size, 1, 1)  # random latent tensors
    fake_images = generator(xb)
    print(fake_images.shape)
    #show_images(fake_images)

    generator = to_device(generator, device)

    sample_dir = 'generated2'
    os.makedirs(sample_dir, exist_ok=True)

    #fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    fixed_latent = torch.randn(1, latent_size, 1, 1, device=device)
    #fixed_latent[0][0]=1
    #print(fixed_latent)

    save_samples(0, fixed_latent)


    lr = 0.00005
    epochs = 400
    clamp_num=0.01# WGAN clip gradient

    history = fit(epochs, lr)

    losses_g, losses_d, real_scores, fake_scores = history

    # Save the model checkpoints
    #torch.save(generator.state_dict(), 'Gvid1net.pth')
    #torch.save(discriminator.state_dict(), 'Dvid1net.pth')

    from IPython.display import Image

    Image('./generated2/2generated-images-0001.png')

    import cv2
    import os

    vid_fname = 'gans_training.avi'

    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
    files.sort()

    #out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 1, (530,530))
    #[out.write(cv2.imread(fname)) for fname in files]
    #out.release()
   # print(cos_similarity)
    #print(eud)
    #print(ssim_scores)
    #print(mses)

    #plt.plot(losses_d, '-')
    #plt.plot(losses_g, '-')
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.legend(['Discriminator', 'Generator'])
    #plt.title('Losses');

    #plt.plot(real_scores, '-')
    #plt.plot(fake_scores, '-')
    #plt.xlabel('epoch')
    #plt.ylabel('score')
    #plt.legend(['Real', 'Fake'])
    #plt.title('Scores');
    #plt.clf()
    #plt.plot(cos_similarity)
    #plt.savefig("cosine_similarity_2.png")
    #plt.clf()
    #plt.plot(eud)
    #plt.savefig("euclidean_distance_2.png")
    plt.clf()
    plt.plot(ssim_scores)
    plt.savefig("ssim_score_2.png")
    plt.clf()
    plt.plot(mses)
    plt.savefig("mses_2.png")
    plt.clf()
    plt.plot(mfids)
    #plt.savefig("fid_2.png")
    #plt.show()
