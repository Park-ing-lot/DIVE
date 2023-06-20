from datetime import datetime
import random

import numpy as np
from torch.cuda.amp import autocast

import torch
import copy


def fine_tune(
        epoch,
        model,
        train_loader,
        optimizer,
        device,
        args,
        logger=None,
        callback=None,
        log_interval=1,
        tb_writer=None,
        tb_interval=1,
        scaler=None,
        actual_batch_size=64
):
    total_step = len(train_loader)
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # Forward pass
        with autocast(enabled=args.amp):
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
                output_hidden_states=True, ###
                # actual_batch_size=actual_batch_size,
                actual_batch_size=len(batch['input_ids']),
            )
            
            o_loss = outputs[0]
            orig_hidden = outputs[2]
            enc_hidden = outputs[-1]
            orig_hidden = torch.nn.functional.normalize(orig_hidden)
            enc_hidden = torch.nn.functional.normalize(enc_hidden)
            
            ### self-retrieval
            arange = torch.arange(len(orig_hidden)).to(device)
            
            retrieve = torch.matmul(orig_hidden, enc_hidden.t()) #[b,b]

            loss_fn = torch.nn.CrossEntropyLoss()
            retrieve_loss = loss_fn(retrieve, arange.long())

        ### self-retrieval
        loss = o_loss + retrieve_loss * 0.5

        total_loss += loss.item()
        # Backward and optimize
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ETA: {}'.format(
                epoch + 1,
                args.epochs,
                i + 1,
                total_step,
                loss.item(),
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))

        if tb_writer is not None and i % tb_interval == 0:
            step = epoch * total_step + i + 1
            tb_writer.add_scalars('loss/step', {'loss': o_loss.item()}, step)
            tb_writer.add_scalars('loss/step_cl_ret', {'contrastive': retrieve_loss.item()}, step)


        if callback is not None:
            callback(
                step=i,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                args=args,
                logger=logger
            )

    if tb_writer is not None:
        tb_writer.add_scalars('loss/epoch', {'train': total_loss / total_step}, epoch + 1)
