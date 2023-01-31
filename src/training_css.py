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
        scaler=None
):
    total_step = len(train_loader)
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batch in enumerate(train_loader):
        # Forward pass
        with autocast(enabled=args.amp):
            batch['image_features'] = list(map(lambda x: x.to(device).requires_grad_(), batch['image_features']))
            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=batch['image_features'],
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
                output_hidden_states=True,
            )

            o_loss = outputs[0]
            orig_hidden = outputs[2]
            loss = o_loss + (orig_hidden.sum() * 0) #/ args.gpu_num

            visual_grad=torch.autograd.grad(o_loss, batch['image_features'], retain_graph=True)
            cf_image_mask = copy.deepcopy(batch['attention_mask'])
            f_image_mask = copy.deepcopy(batch['attention_mask'])
            image_mask = copy.deepcopy(batch['attention_mask'])
            text_mask = copy.deepcopy(batch['attention_mask'])


            total_loss += loss.item()
            # Backward and optimize
            if args.amp:
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(retain_graph=True)
                optimizer.step()
            optimizer.zero_grad()

            for j, v_grad in enumerate(visual_grad):
                v_grad = v_grad.sum(-1)
                v_grad_score, v_grad_ind = v_grad.sort(0,descending=True)
                v_grad_score=torch.nn.functional.softmax(v_grad_score,dim=0)
                v_grad_sum=torch.cumsum(v_grad_score,dim=0)
                v_grad_mask=(v_grad_sum<=0.65).long()
                # v_grad_mask[0] = 1

                num = len(torch.nonzero(v_grad_mask))
                cf_image_mask[j][v_grad_ind[:num]+2] = 0 # +2 becasue of the <begin_img>, tpe token
                f_image_mask[j][v_grad_ind[num:]+2] = 0
                image_mask[j][v_grad_ind+2] = 0

            for j, features in enumerate(batch['image_features']):
                image_mask[j][:len(features)+2] = 0
                text_mask[j][len(features)+2:] = 0
                # cf_image_mask[j][len(features)+2:] = 0
                # f_image_mask[j][len(features)+2:] = 0

            outputs_pos_image = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=batch['image_features'],
                attention_mask=f_image_mask.to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
                output_hidden_states=True,
            )
            pos_loss = outputs_pos_image[0]
            pos_hidden = outputs_pos_image[2]

            outputs_neg_image = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=batch['image_features'],
                attention_mask=cf_image_mask.to(device),
                decoder_input_ids=batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch else None,
                decoder_attention_mask=batch['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in batch else None,
                labels=batch['labels'].to(device),
                answer_ids=batch['answer_ids'].to(device) if 'answer_ids' in batch else None,
                answer_attention_mask=batch['answer_attention_mask'].to(
                    device) if 'answer_attention_mask' in batch else None,
                output_hidden_states=True,
            )
            neg_loss = outputs_neg_image[0]
            neg_hidden = outputs_neg_image[2]   

            
            ### pos as neg
            orig_hidden = torch.nn.functional.normalize(orig_hidden)
            pos_hidden = torch.nn.functional.normalize(pos_hidden)
            neg_hidden = torch.nn.functional.normalize(neg_hidden)
            
            img_pos = torch.matmul(orig_hidden, pos_hidden.t()) #[b,b]
            arange = torch.arange(len(img_pos))

            img_neg = torch.matmul(orig_hidden, neg_hidden.t()) #[b,b]

            img_p_logit = torch.cat((img_pos,img_neg), dim=-1) # [b, 2b]
        
            img_p_softmax_logit = torch.nn.functional.softmax(img_p_logit, 1)

            img_contras_loss = - torch.log(img_p_softmax_logit[arange, arange])
            img_contras_loss = img_contras_loss.mean()
            
            loss = pos_loss * 0.5 + img_contras_loss * 0.5
            # 0.5 or 0.2임
            
            
            # ### only neg as neg
            # orig_hidden = torch.nn.functional.normalize(orig_hidden)
            # pos_hidden = torch.nn.functional.normalize(pos_hidden)
            # neg_hidden = torch.nn.functional.normalize(neg_hidden)
            
            # img_pos = torch.matmul(orig_hidden, pos_hidden.t()) #[b,b]
            # arange = torch.arange(len(img_pos))
            # img_pos = img_pos[arange,arange]
            # img_pos = img_pos.unsqueeze(-1) #[b,1]

            # img_neg = torch.matmul(orig_hidden, neg_hidden.t()) #[b,b]

            # img_p_logit = torch.cat((img_pos,img_neg), dim=-1)
            # img_p_softmax_logit = torch.nn.functional.softmax(img_p_logit, 1) #[b, 2]

            # img_contras_loss = - torch.log(img_p_softmax_logit[:, 0])
            # img_contras_loss = img_contras_loss.mean()
            # loss = pos_loss * 0.2 + img_contras_loss * 0.2 ###

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
            tb_writer.add_scalars('loss/step_cl_img', {'contrastive': img_contras_loss.item()}, step)
            tb_writer.add_scalars('loss/step_cf', {'loss': neg_loss.item()}, step)
            tb_writer.add_scalars('loss/step_f', {'loss': pos_loss.item()}, step)
            # tb_writer.add_scalars('loss/step_txt', {'loss': text_loss.item()}, step)
            # tb_writer.add_scalars('loss/step_img', {'loss': img_loss.item()}, step)
            

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
