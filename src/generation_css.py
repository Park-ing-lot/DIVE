import copy
from datetime import datetime
import torch
import pickle

from torch.cuda.amp import autocast


def generate_text(
        model,
        gen_loader,
        tokenizer,
        args,
        device,
        logger=None,
        log_interval=1,
):
    total_step = len(gen_loader)
    model.eval()
    generated = []
    start_time = datetime.now()

    critical_objects = []

    for i, batch in enumerate(gen_loader):
        with autocast(enabled=args.amp):     
            outputs = model.generate(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=1, #args.num_beams,
                num_return_sequences=1, #args.num_gen,
                do_sample=False,#args.do_sample if hasattr(args, 'do_sample') else False,
                top_p=1.0, #args.top_p if hasattr(args, 'top_p') else 1.0,
                top_k=0, #args.top_k if hasattr(args, 'top_k') else 0,
                early_stopping=True,
                max_length=30,
            )

            generations = []
            for output in outputs:
                generations.append(tokenizer.decode(output, skip_special_tokens=True))

            encoded_labels = tokenizer.encode_label(label=generations, img_num=None)
            

            batch['image_features'] = list(map(lambda x: x.to(device).requires_grad_(), batch['image_features']))

            outputs = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=batch['image_features'],
                attention_mask=batch['attention_mask'].to(device),
                decoder_input_ids=encoded_labels['decoder_input_ids'].to(device) if 'decoder_input_ids' in encoded_labels else None,
                decoder_attention_mask=encoded_labels['decoder_attention_mask'].to(
                    device) if 'decoder_attention_mask' in encoded_labels else None,
                labels=encoded_labels['labels'].to(device),
                answer_ids=None,
                answer_attention_mask=None,
            )
            loss = outputs[0]

            visual_grad=torch.autograd.grad(loss, batch['image_features'], retain_graph=True)
            cf_image_mask = copy.deepcopy(batch['attention_mask'])
            f_image_mask = copy.deepcopy(batch['attention_mask'])

            for j, v_grad in enumerate(visual_grad):
                v_grad = v_grad.sum(-1)
                v_grad_score, v_grad_ind = v_grad.sort(0,descending=True)
                v_grad_score=torch.nn.functional.softmax(v_grad_score,dim=0)
                v_grad_sum=torch.cumsum(v_grad_score,dim=0)
                v_grad_mask=(v_grad_sum<=0.65).long()
                # v_grad_mask[0] = 1

                num = len(torch.nonzero(v_grad_mask))
                cf_image_mask[j][v_grad_ind[:num]+2] = 0 # +2 becasue of the <begin_img> token
                f_image_mask[j][v_grad_ind[num:]+2] = 0

            for c_i, critical in enumerate(f_image_mask):
                # image_objects = critical[1:-len(batch['input_ids'][c_i])]
                critical_objects.append(critical[1:len(batch['image_features'][c_i])+1].tolist())            

            outputs = model.generate(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=f_image_mask.to(device),
                num_beams=1, #args.num_beams,
                num_return_sequences=5, #args.num_gen,
                do_sample=True,#args.do_sample if hasattr(args, 'do_sample') else False,
                top_p=0.9, #args.top_p if hasattr(args, 'top_p') else 1.0,
                top_k=0, #args.top_k if hasattr(args, 'top_k') else 0,
                early_stopping=True
            )

        # decode generated sentences and append to "generated"
        for j in range(len(batch['index'])):
            generations = []
            for output in outputs[j * args.num_gen: (j + 1) * args.num_gen]:
                generations.append(tokenizer.decode(output, skip_special_tokens=True))

            generated.append({
                'index': batch['index'][j],
                'task_type': batch['task_type'][j],
                'generations': generations
            })

        if (i + 1) % log_interval == 0:
            logger.info('Generating, Step [{}/{}], ETA: {}'.format(
                i + 1,
                total_step,
                str((total_step - (i + 1)) / (i + 1) * (datetime.now() - start_time))
            ))

    # with open('/home/user16/HT/KM-BART-ACL/cf_generated/critical_objects.pickle', 'wb') as f:
    #     pickle.dump(critical_objects, f)

    return generated

'''
        num_beams=1, #args.num_beams,
        num_return_sequences=5, #args.num_gen,
        do_sample=True,#args.do_sample if hasattr(args, 'do_sample') else False,
        top_p=0.9, #args.top_p if hasattr(args, 'top_p') else 1.0,
        top_k=0, #args.top_k if hasattr(args, 'top_k') else 0,
'''