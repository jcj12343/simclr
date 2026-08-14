import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import LocalCIFAR10, TwoCropTransform, get_simclr_augment
from model import SimCLR
from loss import nt_xent_loss

os.makedirs("../logs", exist_ok=True)
os.makedirs("../results", exist_ok=True)
os.makedirs("../report", exist_ok=True)

def main():
    # --------------------------超参--------------------------
    DATA_ROOT = r"C:\Users\jcj\Downloads\cifar-10-python\cifar-10-batches-py"
    BATCH_SIZE = 128
    EPOCHS = 20
    TAU = 0.2
    LR = 3e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # 计时起点
    start_pretrain = time.time()
    # 日志文件
    log_path = "../logs/pretrain_loss.txt"
    full_log_path = "../logs/pretrain_full_log.txt"
    loss_log_file = open(log_path, "w", encoding="utf‑8")
    full_log = open(full_log_path, "w", encoding="utf-8")
    loss_log_file.write("epoch,avg_loss\n")

    aug = get_simclr_augment()
    two_crop = TwoCropTransform(aug)
    train_ds = LocalCIFAR10(root=DATA_ROOT, train=True, transform=two_crop)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    model = SimCLR().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 写入头部日志信息
    full_log.write("==== SimCLR 预训练完整日志 ====\n")
    full_log.write(f"训练开始时间: {time.ctime()}\n")
    full_log.write(f"Device: {DEVICE}\n")
    full_log.write(f"Batch Size: {BATCH_SIZE}\n")
    full_log.write(f"Pretrain Epochs: {EPOCHS}\n")
    full_log.write(f"Temperature tau: {TAU}\n")
    full_log.write(f"Learning Rate: {LR}\n")
    full_log.write(f"训练样本总数: {len(train_ds)}\n")
    full_log.write("-" * 50 + "\n")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pre‑Train Epoch {epoch+1}/{EPOCHS}")
        for (view1, view2), _ in pbar:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            x_cat = torch.cat([view1, view2], dim=0)
            _, z_cat = model(x_cat)
            loss = nt_xent_loss(z_cat, temperature=TAU)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        loss_log_file.write(f"{epoch+1},{avg_loss:.6f}\n")
        full_log.write(f"Epoch {epoch+1} | Average Loss: {avg_loss:.6f}\n")
        scheduler.step()

    # 预训练结束统计耗时
    end_pretrain = time.time()
    pretrain_cost = end_pretrain - start_pretrain
    loss_log_file.close()

    # 保存encoder权重
    torch.save(model.encoder.state_dict(), "../simclr_encoder.pth")
    print(f"预训练完成。loss log -> {log_path}, encoder -> simclr_encoder.pth")
    print(f"预训练总耗时：{pretrain_cost:.2f} 秒 | {pretrain_cost/60:.2f} 分钟")

    # 写入收尾日志
    full_log.write("-" * 50 + "\n")
    full_log.write(f"预训练结束时间: {time.ctime()}\n")
    full_log.write(f"预训练总耗时：{pretrain_cost:.2f} s ({pretrain_cost/60:.2f} min)\n")
    full_log.write("预训练权重已保存: ../simclr_encoder.pth\n")
    full_log.close()

    # 保存超参json
    pre_info = {
        "mode": "simclr_pretrain",
        "train_samples": len(train_ds),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "temperature": TAU,
        "lr": LR,
        "pretrain_time_sec": round(pretrain_cost,2)
    }
    with open("../results/pretrain_config.json", "w", encoding="utf‑8") as f:
        json.dump(pre_info, f, indent=2)

if __name__ == "__main__":
    main()