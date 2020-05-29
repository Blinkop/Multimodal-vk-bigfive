import torch

from torch.utils import data
from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable

import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DataGenerator(data.Dataset):
    def __init__(self, dataframe):
        self._data = dataframe

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx,:].astype('float32')

class DataGenerator2(data.Dataset):
    def __init__(self, df1, df2):
        if (len(df1) != len(df2)):
            raise Exception('Both array must be the same size')

        self._data = [df1, df2]

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, idx):
        return [self._data[0][idx,:].astype('float32'), self._data[1][idx,:].astype('float32')]

class Encoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(Encoder, self).__init__()

        self._arch = arch
        self._encoded_size = encoded_size
        self._layers = []

        inputs = input_shape
        for arc in arch:
            # self._layers.append(nn.Linear(inputs, arc))
            self._layers.append(nn.utils.weight_norm(nn.Linear(inputs, arc)))
            self._layers.append(nn.LeakyReLU(0.2))
            # self._layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        self.encoder = nn.Sequential(*self._layers)

        self._fc_mu = nn.Linear(arch[-1], encoded_size)
        self._fc_sg = nn.Linear(arch[-1], encoded_size)

    def encode(self, x):
        z = self.encoder(x)

        return self._fc_mu(z), self._fc_sg(z)

    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, input_shape, encoded_size, arch):
        super(Decoder, self).__init__()

        self._arch = arch
        self._encoded_size = encoded_size
        self._layers = []

        inputs = encoded_size
        for arc in reversed(self._arch):
            # self._layers.append(nn.Linear(inputs, arc))
            self._layers.append(nn.utils.weight_norm(nn.Linear(inputs, arc)))
            self._layers.append(nn.LeakyReLU(0.2))
            # self._layers.append(nn.BatchNorm1d(arc, momentum=0.8))
            inputs = arc

        self.decoder = nn.Sequential(*self._layers)

        self._fc_out = nn.Linear(arch[0], input_shape)

    def decode(self, z):
        return self._fc_out(self.decoder(z))

    def forward(self, z):
        return self.decode(z)

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar

def prior_expert(size, use_cuda=True):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()

    return mu, logvar


class MVAE(nn.Module):
    def __init__(self, input_shapes, encoded_size, beta, archs):
        super(MVAE, self).__init__()

        self._input_shapes = input_shapes
        self._encoded_size = encoded_size
        self._beta = beta

        self.encoder = nn.ModuleList([Encoder(sh, encoded_size, archs[i]) for i, sh in enumerate(input_shapes)])
        self.decoder = nn.ModuleList([Decoder(sh, encoded_size, archs[i]) for i, sh in enumerate(input_shapes)])
        self.expert  = ProductOfExperts()

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.infer(x)
        z = self.reparametrize(mu, logvar)

        decoded = [self.decoder[i](z) for i, inp in enumerate(x)]

        return decoded, mu, logvar

    def infer(self, inputs):
        batch_size = 0
        for inp in inputs:
            if inp is not None:
                batch_size = inp.size(0)
                break

        mu, logvar = prior_expert((1, batch_size, self._encoded_size))

        for i, inp in enumerate(inputs):
            if inp is not None:
                modal_mu, modal_logvar = self.encoder[i](inp)
                # print(modal_mu.shape)
                # print(mu.shape)
                mu     = torch.cat((mu, modal_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, modal_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.expert(mu, logvar)

        return mu, logvar

    def fit(self, train_generator, test_generator, epochs, lr=0.001, log_interval=100):
        self.cuda()
        self._optimizer = opt.Adam(self.parameters(), lr=lr)

        history = {'loss' : np.zeros((epochs)), 'val_loss' : np.zeros((epochs))}

        for epoch in range(epochs):
            history['loss'][epoch] = self._train_epoch(epoch, train_generator, log_interval)
            if (test_generator is not None):
                history['val_loss'][epoch] = self._test_epoch(epoch, test_generator)

        return history

    def _train_epoch(self, epoch, generator, log_interval):
        self.train()
        avg_meter = AverageMeter()

        for batch_idx, (x1, x2) in enumerate(generator):
            X = [x1.cuda(), x2.cuda()]

            self._optimizer.zero_grad()

            total_loss  = self._combinations_loss(X)

            not_nan_all = (~torch.isnan(x1).all(axis=1)) & (~torch.isnan(x2).all(axis=1))

            if not_nan_all.sum().item() > 0:
                inp = [x[not_nan_all] for x in X]

                dec, mu, logvar = self(inp)
                total_loss     += self._elbo_loss(dec, inp, mu, logvar, self._beta)

            total_loss.backward()
            avg_meter.update(total_loss.item(), len(x1))

            self._optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(x1), len(generator.dataset),
                    100. * batch_idx / len(generator), avg_meter.avg))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, avg_meter.avg))

        return avg_meter.avg

    def _test_epoch(self, epoch, generator):
        self.eval()
        avg_meter = AverageMeter()

        with torch.no_grad():
            for x1, x2 in generator:
                X = [x1.cuda(), x2.cuda()]

                total_loss  = self._combinations_loss(X)

                not_nan_all = torch.logical_and(~torch.isnan(x1).all(axis=1),
                                                ~torch.isnan(x2).all(axis=1))

                if not_nan_all.sum().item() > 0:
                    inp = [x[not_nan_all] for x in X]

                    dec, mu, logvar = self(inp)
                    total_loss     += self._elbo_loss(dec, inp, mu, logvar, self._beta)

                avg_meter.update(total_loss.item(), len(x1))

        print('====> Test set loss: {:.4f}'.format(avg_meter.avg))

        return avg_meter.avg

    def _combinations_loss(self, X):
        loss = 0

        for i, x in enumerate(X):
            not_nan = ~torch.isnan(x).all(axis=1)
            inp = [None] * len(X)

            if not_nan.sum().item() == 0:
                continue

            inp[i] = x[not_nan]

            dec, mu, logvar = self(inp)
            loss           += self._elbo_loss(dec, inp, mu, logvar, self._beta)

        return loss

    def _elbo_loss(self, decoded, x, mu, logvar, beta):
        loss = torch.zeros(mu.shape[0]).cuda()

        for dec, inp in zip(decoded, x):
            if inp is not None:
                loss += torch.sum(F.mse_loss(dec, inp.view(-1, np.prod(inp.shape[1:])), reduction='none'), dim=1)
                
        KLD = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss += KLD

        return torch.mean(loss)

    def decode(self, z):
        return [self.decoder[i].decode(z) for i in range(len(self.decoder))]

    def encode(self, x):
        mu, logvar = self.infer(x)
        return self.reparametrize(mu, logvar)

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
