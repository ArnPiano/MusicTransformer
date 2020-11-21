import os        # Temporary until main.py is not ready
from model import MusicTransformer, test_composition
from data import Data
from config import *
import criteria

import utils
import time
from tqdm import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from midi_processor.processor import decode_midi

# Ugly easy fix until main.py is ready
try:
    os.makedirs(log_dir)
except:
    continue
try:
    os.makedirs(midi_out_dir)
except:
    continue
try:
    os.makedirs(model_dir)
except:
    continue


dataset = Data(data_dir)

writer = SummaryWriter(log_dir)

criterion = criteria.SmoothCrossEntropyLoss()
criterion2 = criteria.TransformerLoss()
accuracy = criteria.CategoricalAccuracy()


model = MusicTransformer(D, L, N, H, writer = writer, rate=rate )
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0, betas=(.9, .98), eps = 1e-9)
scheduler = utils.NoamOptimizer(D, optimizer=optimizer )

idx = 0
start_epoch = 0


if load_model:
    ckp = torch.load(model_dir+stop_and_go)
    model.load_state_dict(ckp['state_dict'])
    scheduler.optimizer.load_state_dict(ckp['optimizer'])
    start_epoch = ckp['epoch']+1
    idx = start_epoch * (len(dataset) // B)


for e in range(start_epoch, epochs):
    print(f'\nEpoch {e + 1} start\n', flush=True)

    total_loss_list = []

    for b in tqdm(range(len(dataset.file_dict["train"])//B)):
        optimizer.zero_grad()
        try:
            x, y = dataset.slide_seq2seq_batch(B, L)
            x = torch.from_numpy(x).contiguous().to(device, non_blocking=True, dtype = torch.int)
            y = torch.from_numpy(y).contiguous().to(device, non_blocking=True, dtype = torch.int)
        except IndexError:
            continue
        start_time = time.time()
        model.train()
        x = model(x)

        loss = criterion(x, y)  # (x.view(-1, vocab_size), y.view(-1).to(torch.long))
        loss2 = criterion2(x, y)
        acc = accuracy(x, y)

        total_loss_list.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scheduler.step()
        end_time = time.time()

        writer.add_scalar('Smooth Cross Entropy Loss', loss, global_step=idx)
        writer.add_scalar('Cross Entropy Loss', loss2, global_step=idx)
        writer.add_scalar('Accuracy', acc, global_step=idx)
        writer.add_scalar('Seconds per batch', end_time-start_time, global_step=idx)
        writer.add_scalar('Learning rate', scheduler.rate(), global_step=idx)

        if b % eval_frequency == 0:

            x, y = dataset.slide_seq2seq_batch(B, L, 'eval')
            x = torch.from_numpy(x).contiguous().to(device, dtype=torch.int)
            y = torch.from_numpy(y).contiguous().to(device, dtype=torch.int)
            model.eval()

            out, weights = model(x)


            e_loss = criterion(out, y)
            e_loss2 = criterion2(out, y)
            e_acc = accuracy(out, y)

            if b == 0:
                writer.add_histogram('Source Analysis', x, global_step=idx)
                writer.add_histogram('Target Analysis', y, global_step=idx)
                for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(
                        attn_log_name, weight, step=idx, writer=writer)

                if e == 0:

                    with torch.no_grad():
                        writer.add_graph(model, x)

            writer.add_scalars('Smooth Loss Comparison', {'Eval Loss': e_loss, 'Train Loss': loss}, global_step=idx)
            writer.add_scalars('Cross Loss Comparison', {'Eval Loss': e_loss2, 'Train Loss': loss2}, global_step=idx)
            writer.add_scalars('Accuracy Comparison', {'Eval Accuracy': e_acc, 'Train Accuracy': acc}, global_step=idx)



        torch.cuda.empty_cache()
        idx += 1

    total_loss = (sum(total_loss_list) / len(total_loss_list))

    # save best model for later
    if not load_model:
        best_val_loss = total_loss
        best_ckp = {'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss}
        torch.save(best_ckp, model_dir + best_model)
        print('-> Best Model Saved <-')
    else:
        best_ckp = torch.load(model_dir + best_model)
        best_val_loss = best_ckp['best_val_loss']
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            best_ckp = {'state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss}
            torch.save(best_ckp, model_dir + best_model)
            print('-> Best Model Saved <-')

    test_composition(dataset, model, composition_length,
                     filename='midi_epoch-{:04}'.format(e),
                     n_attempts=number_of_trials_before_giving_up)
    model.infer_mode(False)

    ckp = {
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': scheduler.optimizer.state_dict(),
    }
    if e % ckp_frequency == 0:
        torch.save(ckp, model_dir+train_ckp(e))
        print('-> Global Checkpoint Saved <-')

    torch.save(ckp, model_dir+stop_and_go)
    print('-> Stop&Go Checkpoint Saved <-')

ckp = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer': scheduler.optimizer.state_dict(),
    }
torch.save(ckp, model_dir+final)
print('-> Final Checkpoint Saved <-')

writer.close()
