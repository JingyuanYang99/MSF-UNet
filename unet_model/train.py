import os
import math
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import U_Net
from dataset import UDataset

# =========================
# 开关
# True: 训练 + 测试
# False: 只测试
# =========================
TRAIN_MODE = True

# =========================
# 是否自动从 last checkpoint 恢复训练
# True: 若存在 last checkpoint，则继续训练
# False: 从头开始训练
# =========================
AUTO_RESUME = True

# =========================
# 超参数
# =========================
batch_size = 6
input_dim_AGRI = 15
input_dim_GIIRS = 1690
output_dim = 37

learning_rate = 5e-5
warmup_steps = 100
num_epochs = 30

log_step_interval = 1
save_step_interval = 100

# 先用稳一点的设置
train_num_workers = 6
test_num_workers = 6
use_pin_memory = False

best_save_path = './checkpoint/unet_best.pth'
last_save_path = './checkpoint/unet_last.pth'
load_path = best_save_path

train_loss_csv = './checkpoint/train_loss.csv'
test_loss_csv = './checkpoint/test_loss.csv'
log_dir = './runs'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# 数据路径
# =========================
trainset_path = r'/home/ub/data/npz_file/with_era5_uvw/train'
testset_path = r'/home/ub/data/npz_file/with_era5_uvw/test'

os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def init_csv_if_needed(csv_path, header):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(header)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    step_in_epoch,
    global_step,
    best_loss,
    num_epochs
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step_in_epoch': step_in_epoch,
        'global_step': global_step,
        'best_loss': best_loss,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        start_step_in_epoch = checkpoint.get('step_in_epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        # 兼容旧版只保存 model.state_dict() 的情况
        model.load_state_dict(checkpoint)
        start_epoch = 0
        start_step_in_epoch = 0
        global_step = 0
        best_loss = float('inf')

    return start_epoch, start_step_in_epoch, global_step, best_loss


def wind_loss(pred_u, pred_v, u, v, criterion):
    u_mse_loss = criterion(pred_u, u)
    v_mse_loss = criterion(pred_v, v)
    total_loss = u_mse_loss + v_mse_loss
    return total_loss, u_mse_loss, v_mse_loss


def get_lr(it, all_iters):
    warmup_iters = warmup_steps
    lr_decay_iters = all_iters
    min_lr = learning_rate / 10

    if it < warmup_iters:
        return learning_rate * it / max(warmup_iters, 1)

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / max((lr_decay_iters - warmup_iters), 1)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def test_one_epoch(model, testloader, criterion, device, global_step=None, writer=None):
    model.eval()

    test_loss_total = 0.0
    test_loss_u = 0.0
    test_loss_v = 0.0

    with torch.no_grad():
        for AGRI_curr, AGRI_prev, GIIRS, GIIRS_delta_time, ERA5_u, ERA5_v in testloader:
            AGRI_curr = AGRI_curr.to(device)
            AGRI_prev = AGRI_prev.to(device)
            GIIRS = GIIRS.to(device)
            GIIRS_delta_time = GIIRS_delta_time.to(device)
            ERA5_u = ERA5_u.to(device)
            ERA5_v = ERA5_v.to(device)

            pred_u, pred_v= model(AGRI_curr, AGRI_prev, GIIRS, GIIRS_delta_time)
            loss, loss_u, loss_v= wind_loss(
                pred_u, pred_v, ERA5_u, ERA5_v, criterion
            )

            test_loss_total += loss.item()
            test_loss_u += loss_u.item()
            test_loss_v += loss_v.item()

    avg_test_total = test_loss_total / len(testloader)
    avg_test_u = test_loss_u / len(testloader)
    avg_test_v = test_loss_v / len(testloader)

    if writer is not None and global_step is not None:
        writer.add_scalar('Loss/test_total', avg_test_total, global_step)
        writer.add_scalar('Loss/test_u', avg_test_u, global_step)
        writer.add_scalar('Loss/test_v', avg_test_v, global_step)

    print(
        f"Test Result -> "
        f"Total: {avg_test_total:.4f}, "
        f"u: {avg_test_u:.4f}, "
        f"v: {avg_test_v:.4f}, "
    )

    return avg_test_total, avg_test_u, avg_test_v


def main():
    train_dataset = UDataset(trainset_path)
    test_dataset = UDataset(testset_path)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=use_pin_memory
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=use_pin_memory
    )

    model = U_Net(input_dim_AGRI, input_dim_GIIRS, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if TRAIN_MODE:
        writer = SummaryWriter(log_dir=log_dir)

        init_csv_if_needed(
            train_loss_csv,
            ['epoch', 'step_in_epoch', 'global_step', 'lr', 'train_total', 'train_u', 'train_v']
        )
        init_csv_if_needed(
            test_loss_csv,
            ['epoch', 'global_step', 'test_total', 'test_u', 'test_v']
        )

        if AUTO_RESUME and os.path.exists(last_save_path):
            start_epoch, start_step_in_epoch, global_step, best_loss = load_checkpoint(
                last_save_path, model, optimizer, map_location=device
            )
            print(f'从断点恢复训练: {last_save_path}')
            print(
                f'start_epoch={start_epoch}, '
                f'start_step_in_epoch={start_step_in_epoch}, '
                f'global_step={global_step}, '
                f'best_loss={best_loss:.6f}'
            )
        else:
            print('从头开始训练')

        iter_per_epoch = len(trainloader)
        all_iters = num_epochs * iter_per_epoch
        print(f'len(trainloader) = {iter_per_epoch}')

        for epoch in range(start_epoch, num_epochs):
            model.train()

            loss_accum_total = 0.0
            loss_accum_u = 0.0
            loss_accum_v = 0.0

            # 只有恢复训练的第一个 epoch 才跳过前面已经跑过的 batch
            resume_skip = start_step_in_epoch if epoch == start_epoch else 0

            if resume_skip > 0:
                print(f'Epoch {epoch} 将跳过前 {resume_skip} 个 step')

            for step, (AGRI_curr, AGRI_prev, GIIRS, GIIRS_delta_time, ERA5_u, ERA5_v) in enumerate(trainloader):
                if step < resume_skip:
                    continue

                AGRI_curr = AGRI_curr.to(device)
                AGRI_prev = AGRI_prev.to(device)
                GIIRS = GIIRS.to(device)
                GIIRS_delta_time = GIIRS_delta_time.to(device)
                ERA5_u = ERA5_u.to(device)
                ERA5_v = ERA5_v.to(device)

                current_iter = global_step
                lr = get_lr(current_iter, all_iters)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad()

                pred_u, pred_v= model(AGRI_curr, AGRI_prev, GIIRS, GIIRS_delta_time)
                loss, loss_u, loss_v= wind_loss(
                    pred_u, pred_v, ERA5_u, ERA5_v, criterion
                )

                loss.backward()
                optimizer.step()

                global_step += 1

                # 保存的是“下一次恢复时要跳过到哪里”
                next_step_in_epoch = step + 1

                if global_step % save_step_interval == 0:
                    save_checkpoint(
                        last_save_path,
                        model,
                        optimizer,
                        epoch,
                        next_step_in_epoch,
                        global_step,
                        best_loss,
                        num_epochs
                    )
                    print(f"Step {global_step} 已保存断点到: {last_save_path}")

                loss_accum_total += loss.item()
                loss_accum_u += loss_u.item()
                loss_accum_v += loss_v.item()

                writer.add_scalar('lr', lr, global_step)

                if global_step % log_step_interval == 0:
                    avg_loss_total = loss_accum_total / log_step_interval
                    avg_loss_u = loss_accum_u / log_step_interval
                    avg_loss_v = loss_accum_v / log_step_interval

                    writer.add_scalar('Loss/train_total', avg_loss_total, global_step)
                    writer.add_scalar('Loss/train_u', avg_loss_u, global_step)
                    writer.add_scalar('Loss/train_v', avg_loss_v, global_step)

                    print(
                        f"[Epoch {epoch} Step {global_step}] "
                        f"Train Total: {avg_loss_total:.4f}, "
                        f"u: {avg_loss_u:.4f}, "
                        f"v: {avg_loss_v:.4f}, "
                        f"lr: {lr:.8f}"
                    )

                    with open(train_loss_csv, 'a', newline='') as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow([
                            epoch, step, global_step, lr,
                            avg_loss_total, avg_loss_u, avg_loss_v
                        ])

                    loss_accum_total = 0.0
                    loss_accum_u = 0.0
                    loss_accum_v = 0.0

            print('=============== start test ============')

            avg_test_total, avg_test_u, avg_test_v = test_one_epoch(
                model, testloader, criterion, device, global_step=global_step, writer=writer
            )

            with open(test_loss_csv, 'a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    epoch, global_step,
                    avg_test_total, avg_test_u, avg_test_v
                ])

            if avg_test_total < best_loss:
                best_loss = avg_test_total
                print("Save Best checkpoint:", best_save_path)
                save_checkpoint(
                    best_save_path,
                    model,
                    optimizer,
                    epoch + 1,
                    0,
                    global_step,
                    best_loss,
                    num_epochs
                )

            # epoch 真正结束后，下一次从 epoch+1 的 step 0 开始
            save_checkpoint(
                last_save_path,
                model,
                optimizer,
                epoch + 1,
                0,
                global_step,
                best_loss,
                num_epochs
            )
            print(f"Epoch {epoch} 完成，已保存断点到: {last_save_path}")

        writer.close()

    else:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'未找到测试权重文件: {load_path}')

        checkpoint = torch.load(load_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f'成功加载模型权重: {load_path}')
        print('=============== only test ============')
        test_one_epoch(model, testloader, criterion, device)


if __name__ == '__main__':
    main()