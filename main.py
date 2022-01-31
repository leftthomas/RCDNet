import os
import time

import pandas as pd
import torch
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcdnet import RCDNet
from utils import parse_args, RainDataset


def train_loop(net, data_loader):
    net.train()
    step = 0
    for epoch in range(1, args.num_iter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader):
            im_rain, im_gt = [x.cuda() for x in data]
            optimizer.zero_grad()
            B0, ListB, ListR = net(im_rain)
            loss_Bs = 0
            loss_Rs = 0
            for j in range(opt.num_stage):
                loss_Bs = float(loss_Bs) + 0.1 * F.mse_loss(ListB[j], im_gt)
                loss_Rs = float(loss_Rs) + 0.1 * F.mse_loss(ListR[j], im_rain - im_gt)
            lossB = F.mse_loss(ListB[-1], im_gt)
            lossR = 0.9 * F.mse_loss(ListR[-1], im_rain - im_gt)
            lossB0 = 0.1 * F.mse_loss(B0, im_gt)
            loss = lossB0 + loss_Bs + lossB + loss_Rs + lossR
            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            if ii % 300 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, lr={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, lr))
            step += 1
        mse_per_epoch /= (ii + 1)
        print('Epoch:{:>2d}, Derain_Loss={:+.2e}'.format(epoch + 1, mse_per_epoch))
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'model_'
        save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'step': step + 1,
        }, save_path_model)
        model_prefix = 'DerainNet_state_'
        save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1) + '.pt')
        torch.save(net.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
        print('-' * 100)
    print('Reach the maximal epochs! Finish training')
    return loss


def test_loop(net, data_loader, n_iter):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name in test_bar:
            b_0, list_b, list_r = model(rain.cuda())
            out = torch.clamp(list_b[-1], 0, 255).squeeze(dim=0).permute(1, 2, 0).byte().cpu().numpy()
            # computer the metrics with Y channel
            y = color.rgb2ycbcr(out)[:, :, 0]
            gt = color.rgb2ycbcr(norain.squeeze(dim=0).permute(1, 2, 0).byte().numpy())[:, :, 0]
            psnr = peak_signal_noise_ratio(gt, y, data_range=255)
            ssim = structural_similarity(gt, y, data_range=255)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out).save(save_path)
            test_bar.set_description('Test [{}/{}] PSNR: {:.2f} SSIM: {:.4f}'
                                     .format(n_iter, 1 if args.model_file else args.num_iter,
                                             total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, n_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, n_iter)
    results['PSNR'].append('%.2f' % val_psnr)
    results['SSIM'].append('%.4f' % val_ssim)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, n_iter + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='epoch')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


if __name__ == '__main__':
    args = parse_args()
    test_dataset = RainDataset(args.data_path, args.data_name, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    results = {'PSNR': [], 'SSIM': []}
    best_psnr, best_ssim = 0.0, 0.0

    model = RCDNet(args.num_map, args.num_channel, args.num_block, args.num_stage).cuda()
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model, test_loader, 1)
    else:
        train_dataset = RainDataset(args.data_path, args.data_name, 'train', args.patch_size, args.batch_size * 1500)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        optimizer = Adam(model.parameters(), lr=args.lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=args.milestone, gamma=0.2)
        results['Loss'] = []
        for epoch in range(1, args.num_iter + 1):
            train_loss = train_loop(model, train_loader)
            results['Loss'].append(train_loss)
            save_loop(model, test_loader, epoch)
