import os
import json
import time
import torch
import matplotlib.pyplot as plt
from math import ceil
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import LocalCIFAR10, get_eval_transform
from model import LinearProbeModel

os.makedirs("../report", exist_ok=True)
os.makedirs("../results", exist_ok=True)
os.makedirs("../logs", exist_ok=True)

def plot_loss_curve(log_file, save_img):
    epochs = []
    losses = []
    with open(log_file, "r", encoding="utf‑8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            e, l = line.strip().split(",")
            epochs.append(int(e))
            losses.append(float(l))
    plt.figure(figsize=(8,4))
    plt.plot(epochs, losses, label="Pre‑train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NT‑Xent Loss")
    plt.title("SimCLR Pretrain Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_img, dpi=150)
    plt.close()

def main():
    DATA_ROOT = r"C:\Users\jcj\Downloads\cifar-10-python\cifar-10-batches-py"
    ENCODER_PATH = "../simclr_encoder.pth"
    BATCH_SIZE = 128
    PROBE_EPOCHS =20
    SHOW_NUM = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 线性探测计时起点
    start_probe = time.time()
    probe_log_path = "../logs/linear_probe_log.txt"
    probe_log = open(probe_log_path, "w", encoding="utf-8")

    # 加载预训练encoder
    encoder = resnet18(pretrained=False)
    encoder.conv1 = torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    encoder.maxpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    model = LinearProbeModel(encoder, num_classes=10).to(DEVICE)

    eval_trans = get_eval_transform()
    ds_train = LocalCIFAR10(DATA_ROOT, train=True, transform=eval_trans)
    ds_test = LocalCIFAR10(DATA_ROOT, train=False, transform=eval_trans)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.linear.parameters(), lr=3e-3)
    probe_scheduler = CosineAnnealingLR(opt, T_max=PROBE_EPOCHS, eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    # 写入线性探测日志头部
    probe_log.write("==== Linear Probe 线性探测日志 ====\n")
    probe_log.write(f"开始时间: {time.ctime()}\n")
    probe_log.write(f"Batch Size: {BATCH_SIZE}\n")
    probe_log.write(f"Probe Epochs: {PROBE_EPOCHS}\n")
    probe_log.write(f"线性训练集数量: {len(ds_train)}\n")
    probe_log.write(f"测试集总数量: {len(ds_test)}\n")
    probe_log.write("epoch,test_acc,best_acc\n")

    for epoch in range(PROBE_EPOCHS):
        model.train()
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # test
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader_test:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"[Linear Probe] Epoch {epoch+1:2d} | Test Acc: {acc:.4f}, Best Acc:{best_acc:.4f}")
        probe_log.write(f"{epoch+1},{acc:.6f},{best_acc:.6f}\n")
        probe_scheduler.step()

    # 线性探测结束耗时
    end_probe = time.time()
    probe_cost = end_probe - start_probe
    print(f"Linear Probe 训练耗时：{probe_cost:.2f} 秒 | {probe_cost/60:.2f} 分钟")

    # 读取预训练耗时，计算总耗时
    with open("../results/pretrain_config.json","r",encoding="utf-8") as f:
        pre_cfg = json.load(f)
    pre_time = pre_cfg["pretrain_time_sec"]
    total_all = pre_time + probe_cost

    # 日志收尾
    probe_log.write("-"*40 + "\n")
    probe_log.write(f"线性探测结束时间: {time.ctime()}\n")
    probe_log.write(f"线性探测耗时: {probe_cost:.2f} s\n")
    probe_log.write(f"预训练耗时: {pre_time:.2f} s\n")
    probe_log.write(f"两段训练总耗时: {total_all:.2f} s ({total_all/60:.2f} min)\n")
    probe_log.write(f"最终最优全局测试准确率: {best_acc:.4f}\n")
    probe_log.close()

    # 保存实验结果
    result_data = {
        "linear_probe_epochs": PROBE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_set_size": len(ds_test),
        "train_set_size": len(ds_train),
        "final_best_test_acc": round(best_acc,4),
        "probe_time_sec": round(probe_cost,2),
        "pretrain_total_sec": round(total_all,2)
    }
    with open("../results/result.json", "w", encoding="utf‑8") as f:
        json.dump(result_data, f, indent=2)

    # 绘制预训练loss曲线
    plot_loss_curve("../logs/pretrain_loss.txt", "../report/loss_curve.png")

    # 可视化预测样本 + 计算该批图片准确率
    test_sample_loader = DataLoader(ds_test, batch_size=SHOW_NUM, shuffle=True)
    imgs_sample, labels_sample = next(iter(test_sample_loader))
    imgs_sample = imgs_sample.to(DEVICE)

    with torch.no_grad():
        logits = model(imgs_sample)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    imgs_cpu = imgs_sample.cpu()
    labels_cpu = labels_sample.cpu().numpy()
    class_names = ds_test.classes

    # 统计展示样本的局部准确率
    sample_correct_count = 0
    for idx in range(SHOW_NUM):
        if preds[idx] == labels_cpu[idx]:
            sample_correct_count += 1
    sample_acc = sample_correct_count / SHOW_NUM

    print("="*60)
    print(f"可视化 {SHOW_NUM} 张测试样本统计结果：")
    print(f"预测正确数量：{sample_correct_count} / {SHOW_NUM}")
    print(f"该批次样本局部准确率：{sample_acc:.4f} ({sample_acc*100:.2f} %)")
    print("="*60)

    # 自动计算行列
    cols = 10
    rows = ceil(SHOW_NUM / cols)
    plt.figure(figsize=(cols*1.6, rows*1.8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for idx in range(SHOW_NUM):
        plt.subplot(rows, cols, idx + 1)
        img = imgs_cpu[idx].permute(1,2,0).numpy()
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        img = img * std + mean
        img = img.clip(0,1)
        plt.imshow(img)
        t = class_names[labels_cpu[idx]]
        p = class_names[preds[idx]]
        plt.title(f"T:{t}\nP:{p}", fontsize=6)
        plt.axis("off")
        # 200张取消打印避免刷屏，需要打开就取消注释
        # print(f"sample{idx+1:2d} | True:{t:12s} | Pred:{p:12s}")

    plt.tight_layout()
    plt.savefig("../report/sample_predict.png", dpi=150)
    plt.show()
    print(f"\n图片已输出至 report/: loss_curve.png, sample_predict.png")
    print(f"完整训练日志保存在 ../logs/ 文件夹")

if __name__ == "__main__":
    main()