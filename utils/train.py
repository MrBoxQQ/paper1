import json, datetime, torch, swanlab, argparse, numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from torch import nn

from dataset import make_dataloaders
from utils.model import Model
import os

@dataclass(frozen=True)
class TrainConfig:

    cache_root: Path
    n_class: int = 8
    n_domain: int = 5
    batch_size: int = 64
    num_epochs: int = 50
    num_workers: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-3
    amp: bool = False
    seed: int = 42
    eta_min: float = 1e-6
    out_dir: Path = Path(None)
    sl: str = "disabled"
    cf: str = None
    in_channel:int = 1

def lambda_schedule(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    P_Num = 1
    if p <= P_Num:
      return 0.0
    else:
        return float((2.0 / (1.0 + np.exp(-7.0 * (p - P_Num))) - 1.0)/5)


@torch.no_grad()
def evaluate(model: Model, loader, *, device: torch.device) -> float:
    model.eval()
    y_pred, y_correct, d_pred, d_correct = 0, 0, 0, 0
    total = 0
    for x, y, d in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        out = model(x, lambd=0.0)
        y_pred = out.y_logits.argmax(dim=1)
        y_correct += int((y_pred == y).sum().item())
        d_pred = out.d_logits.argmax(dim=1)
        d_correct += int((d_pred == d).sum().item())
        total += int(y.numel())
    if total == 0:
        return {"y_acc": 0.0, "d_acc": 0.0}
    return {
        "y_acc": float(y_correct / total),
        "d_acc": float(d_correct / total)
    }


def train_with_loaders(cfg: TrainConfig, *, train_loader, val_loader, test_loader=None) -> Dict[str, float]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run = swanlab.init(
        project="wifi_CSI_paper_new",
        experiment_name=f"exp_{time_str}_{cfg.cf}",
        config = cfg.__dict__,
        description="MSDG-DANN-Project",
        mode = cfg.sl
    )

    model = Model(n_class=int(cfg.n_class), n_domain=int(cfg.n_domain), in_channels = cfg.in_channel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, 
        T_max=int(cfg.num_epochs),
        eta_min=cfg.eta_min         
    )
    ce = nn.CrossEntropyLoss()

    total_steps = int(cfg.num_epochs) * len(train_loader)
    if total_steps <= 0:
        raise ValueError("total_steps computed as 0; check dataset size and batch_size")

    scaler = torch.amp.GradScaler(device = device, enabled=bool(cfg.amp) and device.type == "cuda")
    
    out_dir_t = cfg.out_dir / time_str
    out_dir_t_1 = cfg.out_dir / time_str / "all"
    out_dir_t.mkdir(parents=True, exist_ok=True)
    out_dir_t_1.mkdir(parents=True, exist_ok=True)
    log_path = out_dir_t / "train_log.json"
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    history = {
    "config": cfg_dict,
    "epochs": []
    }
    best_val_acc = -1.0
    best_test_acc = -1.0
    global_step = 0
    print("\n"*20)
    pbar_epoch = tqdm(range(int(cfg.num_epochs)), desc="Overall Progress", unit="epoch", initial = 0)

    for epoch in pbar_epoch:
        model.train()
        running_task = 0.0
        running_domain = 0.0
        running_total = 0.0
        running_grad = 0.0

        pbar_batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, unit="batch")

        for x, y, d in pbar_batch:
            global_step += 1
            p = float(global_step / total_steps)
            lambd = lambda_schedule(p)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type = device.type, enabled=scaler.is_enabled()):
                out = model(x, lambd=lambd)
                loss_task = ce(out.y_logits, y)
                loss_domain = ce(out.d_logits, d)
                loss = loss_task + float(lambd) * loss_domain

            scaler.scale(loss).backward()
            
            scaler.unscale_(opt)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            scaler.step(opt)
            scaler.update()

            l_t = float(loss_task.detach().item())
            l_d = float(loss_domain.detach().item())
            l_total = float(loss.detach().item())

            running_task += l_t
            running_domain += l_d
            running_total += l_total
            running_grad += total_grad_norm

            pbar_batch.set_postfix({
                "T_Loss": f"{l_t:.3f}",
                "D_Loss": f"{l_d:.3f}",
                "Grad": f"{total_grad_norm:.5f}",
                "Total": f"{l_total:.3f}"
            })
        scheduler.step()
        avg_loss_task = running_task / max(1, len(train_loader))
        avg_loss_domain = running_domain / max(1, len(train_loader))
        avg_loss_total = running_total / max(1, len(train_loader))
        avg_grad = running_grad / max(1, len(train_loader))
        val_metrics  = evaluate(model, val_loader, device = device)
        val_acc = val_metrics["y_acc"]
        val_d_acc = val_metrics["d_acc"]
        if test_loader is not None and hasattr(test_loader, "dataset") and len(test_loader.dataset) > 0:
            test_metrics = evaluate(model, test_loader, device=device)
            test_acc = test_metrics["y_acc"]
        else:
            test_acc = 0.0, 0.0
        epoch_log = {
            "epoch": int(epoch),
            "loss_task": avg_loss_task,
            "loss_domain": avg_loss_domain,
            "loss_total": avg_loss_total,
            "grad_avg": avg_grad,
            "learning_rate": opt.param_groups[0]["lr"],
            "train_Lambda": float(lambd),
            "val_acc": float(val_acc),
            "val_d_acc": float(val_d_acc),
            "test_acc": float(test_acc),
            "global_step": int(global_step),
        }
        history["epochs"].append(epoch_log)
        log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        pbar_epoch.set_postfix({
                "T_Loss": f"{avg_loss_task:.3f}",
                "Total": f"{avg_loss_total:.3f}",
                "Grad": f"{avg_grad:.3f}",
                "V_Acc": f"{val_acc:.3%}",
                "V_D_Acc": f"{val_d_acc:.3%}",
                "T_Acc": f"{test_acc:.3%}"
            })
        swanlab.log({
            "Train/Loss_Task": float(avg_loss_task),
            "Train/Loss_Domain": float(avg_loss_domain),
            "Train/Loss_Total": float(avg_loss_total),
            "Train/Lambda": float(lambd),
            "Train/Learning_Rate": float(opt.param_groups[0]["lr"]),
            "Val/Accuracy": float(val_acc),
            "Val/Domain_Accuracy": float(val_d_acc),
            "Test/Accuracy": float(test_acc),
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg.__dict__
                },
                out_dir_t / "best_val.pth",
            )
        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            test_val_acc = float(val_acc)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg.__dict__
                },
                out_dir_t / "best_test.pth",
            )

        # torch.save(
        #     {
        #         "model_state": model.state_dict(),
        #         "opt_state": opt.state_dict(),
        #     },
        #     out_dir_t/ f"all/e{epoch}.pth")

    swanlab.finish()

    return {"best_val_acc": float(best_val_acc), "| best_test_acc": float(best_test_acc),"at val_acc":test_val_acc}


def train(cfg: TrainConfig) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "Wrong")
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg.cache_root,
        batch_size=int(cfg.batch_size),
        in_channel=int(cfg.in_channel),
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        mmap=True,    
        preload_to_device = True,
        device = device,
    )
    return train_with_loaders(cfg, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("-b","--batch_size", type=int, default=None)
    p.add_argument("-e","--num_epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("-sl", "--swanlab", type=str, default=None)
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("-c","--config", type=str, default=None)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device) 
    cfg = TrainConfig(
        cache_root=Path(args.cache_root),
        batch_size=int(args.batch_size),
        num_epochs=int(args.num_epochs),
        num_workers=int(args.num_workers),
        out_dir=Path(args.out_dir),
        amp=bool(args.amp),
        sl = str(args.swanlab),
        cf = str(args.config)
    )
    summary = train(cfg)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    
    os.system('cls' if os.name == 'nt' else 'clear')
    main()
