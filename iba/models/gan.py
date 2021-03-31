import torch
import torch.nn as nn
import torch.utils.data as Data
from .utils import _to_saliency_map
import random
from mmcv import get_logger
from .model_zoo import get_module
from .pytorch import _IBAForwardHook


class WordEmbeddingMasker(nn.Module):
    def __init__(self, mean, eps, word_embedding_mask_param):
        super().__init__()
        self.eps = eps
        self.mean = mean
        self.word_embedding_mask_param = word_embedding_mask_param
        self.sigmoid = nn.Sigmoid()
        self.gaussian = None

    def set_gaussian_noise(self, gaussian):
        """set gaussian noise"""
        self.gaussian = gaussian

    def forward(self, x):
        if self.gaussian is None:
          return x
        noise = self.eps * self.gaussian + self.mean
        word_mask = self.sigmoid(self.word_embedding_mask_param)
        z = word_mask * x + (1 - word_mask) * noise
        return z


class Generator(torch.nn.Module):
    # generator takes random noise as input, learnable parameter is the img mask.
    # masked img (with noise added) go through the original network and generate masked feature map
    def __init__(self, img, context, device='cuda:0', capacity=None):
        super().__init__()
        self.img = img
        self.context = context

        # use img size
        # TODO make img_mask_param a Parameter
        if capacity is not None:
            #TODO review
            word_embedding_mask_param = torch.tensor(capacity.sum(1).cpu().detach().numpy()).to(device)
            #TODO pass embedding dim from attributer
            self.word_embedding_mask_param = word_embedding_mask_param.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, 100).clone()
        else:
            self.word_embedding_mask_param = torch.zeros(img.shape,
                                              dtype=torch.float).to(device)
        self.word_embedding_mask_param.requires_grad = True
        # TODO make mean and eps Parameters.
        self.mean = torch.zeros((self.word_embedding_mask_param.shape[0], 1, 1)).to(device)
        self.mean.requires_grad = True
        self.eps = torch.ones((self.word_embedding_mask_param.shape[0], 1, 1)).to(device)
        self.eps.requires_grad = True
        self.feature_map = None

        # register hook in trained classification network to get hidden representation of masked input
        def store_feature_map(model, input, output):
            self.feature_map = output

        self._hook_handle = get_module(
            self.context.classifier,
            self.context.layer).register_forward_hook(store_feature_map)

        # construct word embedding masker
        self.masker = WordEmbeddingMasker(self.mean, self.eps, self.word_embedding_mask_param)

        # register hook to mask word embedding
        self._mask_hook_handle = get_module(
            self.context.classifier,
            "embedding").register_forward_hook(_IBAForwardHook(self.masker, 'output'))

    def forward(self, gaussian):
        self.masker.set_gaussian_noise(gaussian)
        _ = self.context.classifier(self.img.unsqueeze(1).expand(-1, gaussian.shape[1]), torch.tensor([self.img.shape[0]]).expand(gaussian.shape[1]).to('cpu'))
        feature_map_padded, feature_map_lengths = nn.utils.rnn.pad_packed_sequence(self.feature_map[0])
        return feature_map_padded

    @torch.no_grad()
    def get_feature_map(self):
        _ = self.context.classifier(self.img.unsqueeze(1), torch.tensor([self.img.shape[0]]).expand(1).to('cpu'))
        feature_map_padded, feature_map_lengths = nn.utils.rnn.pad_packed_sequence(self.feature_map[0])
        return feature_map_padded

    def img_mask(self):
        return self.masker.sigmoid(self.word_embedding_mask_param)

    def clear(self):
        del self.feature_map
        self.feature_map = None
        self.detach()

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._mask_hook_handle.remove()
            self._mask_hook_handle = None
        else:
            raise ValueError(
                "Cannot detach hock. Either you never attached or already detached."
            )


class Discriminator(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        # Input_dim = channels (SentenceLengthxHiddenSize)
        # Output_dim = 1
        self.rnn = nn.LSTM(hidden_dim,
                           hidden_dim,
                           num_layers=1)

        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.output = nn.Sequential(
            # The output of discriminator is no longer a probability, we do not apply sigmoid at the output of discriminator.
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        output, (hidden, cell) = self.rnn(x)
        return self.output(hidden[-1,:,:])


class WGAN_CP(object):

    def __init__(self,
                 context=None,
                 img=None,
                 feature_mask=None,
                 feature_noise_mean=None,
                 feature_noise_std=None,
                 device='cuda:0'):
        self.img = img
        self.feature_mask = feature_mask
        self.feature_noise_mean = feature_noise_mean
        self.feature_noise_std = feature_noise_std
        self.device = device

        self.generator = Generator(img=img,
                                   context=context,
                                   device=self.device,
                                   capacity=feature_mask).to(self.device)
        self.feature_map = self.generator.get_feature_map()

        # channel is determined from feature map
        self.discriminator = Discriminator(self.feature_map.shape[-1]).to(
                                               self.device)

    def _build_data(self, dataset_size, sub_dataset_size, batch_size):
        # create dataset from feature mask and feature map
        num_sub_dataset = int(dataset_size / sub_dataset_size)
        dataset = []
        for idx_subset in range(num_sub_dataset):
            sub_dataset = self.feature_map.expand(
                -1, sub_dataset_size, -1)
            noise = torch.zeros_like(sub_dataset).normal_()
            std = random.uniform(0, 5)
            mean = random.uniform(-2, 2)
            noise = std * noise + mean
            sub_dataset = self.feature_mask.unsqueeze(1) * sub_dataset + (
                1 - self.feature_mask.unsqueeze(1)) * noise
            dataset.append(sub_dataset)

        dataset = torch.cat(dataset, dim=1)
        
        # permute dataset because NLP data use dim 1 for batch size
        dataset = dataset.permute(1,0,2)
        dataset = dataset.detach()
        tensor_dataset = Data.TensorDataset(dataset)
        dataloader = Data.DataLoader(
            dataset=tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        return dataloader

    def train(self,
              logger=None,
              dataset_size=200,
              sub_dataset_size=20,
              lr=0.00005,
              batch_size=32,
              weight_clip=0.01,
              epochs=200,
              critic_iter=5):
        # TODO add learning rate scheduler
        # Initialize generator and discriminator
        if logger is None:
            logger = get_logger('iba')
        data_loader = self._build_data(dataset_size, sub_dataset_size,
                                       batch_size)

        # Optimizers
        optimizer_G = torch.optim.RMSprop([{
            "params": self.generator.masker.mean,
            "lr": 0.1
        }, {
            "params": self.generator.masker.eps,
            "lr": 0.05
        }, {
            "params": self.generator.masker.word_embedding_mask_param,
            "lr": 0.03
        }])
        optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(),
                                          lr=lr)

        # training
        batches_done = 0
        for epoch in range(epochs):
            for i, imgs in enumerate(data_loader):

                # train discriminator
                imgs = imgs[0]
                imgs = imgs.permute(1, 0, 2)
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.img).float()
                z = z.unsqueeze(-1).unsqueeze(-1).expand(imgs.shape[0], imgs.shape[1],  100).clone().normal_().to(self.device)
                # z = z.unsqueeze(0).expand(imgs.shape[0], -1, -1,
                #                           -1).clone().normal_().to(self.device)
                # std = random.uniform(0, 5)
                # mean = random.uniform(-2, 2)
                # noise = std * noise + mean

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(imgs)) + torch.mean(
                    self.discriminator(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

                # Train the generator every n_critic iterations
                if i % critic_iter == 0:
                    # train generator
                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    optimizer_G.step()
                    log_str = f'[Epoch{epoch + 1}/{epochs}], '
                    log_str += f'[{batches_done % len(data_loader)}/{len(data_loader)}], '
                    log_str += f'D loss: {loss_D.item():.5f}, G loss: {loss_G.item():.5f}'
                    logger.info(log_str)
                batches_done += 1

        del data_loader
        self.generator.clear()
