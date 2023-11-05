import torch


def save_checkpoint(fn, hp, model, optimizer, scheduler, epoch, steps, best_valid_loss):
    ckpt = {
        'hp': hp,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'steps': steps,
        'best_valid_loss': best_valid_loss,
    }
    torch.save(ckpt, fn)


def load_checkpoint(fn, hp, model, optimizer, scheduler):
    ckpt = torch.load(fn, map_location="cpu")
    for k in hp.keys():
        if hp[k] != ckpt['hp'][k]:
            print(f'WARNING : current hyperparameters differ from those in ckpt "{fn}"; ckpt hps will be ignored')
            break
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['steps'], ckpt['best_valid_loss']

