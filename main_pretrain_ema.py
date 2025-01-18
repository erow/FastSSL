import argparse
import math
import sys
import main_pretrain
from main_pretrain import main, get_args_parser
import torch
import util.misc as misc
import util.lr_sched as lr_sched
        
def train_one_epoch(model: torch.nn.Module,online_prob, 
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    metric_logger.add_meter('m', misc.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.data_set == "ffcv":
            samples = data[:-1]
            targets = data[-1]            
        else:
            samples, targets = data
        
        if isinstance(samples,list) or isinstance(samples,tuple):
            samples = [i.to(device, non_blocking=True) for i in samples]
            if len(samples)==1:
                samples = samples[0]
        else:
            samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).flatten()
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            epoch_i = data_iter_step / len(data_loader) + epoch
            lr_sched.adjust_learning_rate(optimizer, epoch_i, args)
            m = lr_sched.adjust_moco_momentum(epoch_i, args)        
            model.module.update(m)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            loss, log = model(samples,targets=targets, epoch=epoch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            torch.save(model.module, "nan_model.pt")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()            
                
            if online_prob:
                log.update(online_prob.step(samples,targets))                

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[-1]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(m=m)
        for k,v in log.items():
            metric_logger.update(**{k:v})
        

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('epoch_1000x',epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            for k,v in log.items():
                log_writer.add_scalar(f'{k}', v, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    # replace with the new train function with momentum
    from util.helper import aug_parse
    main_pretrain.train_one_epoch = train_one_epoch
    parser = get_args_parser()
    parser.add_argument("-m",type=float, default=0.996)
    args = aug_parse(parser)
    main(args)
