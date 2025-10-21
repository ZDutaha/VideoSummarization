
import os, sys, json, time, subprocess, re, yaml
from pathlib import Path

CFG_PATH = Path("experiments/config.yaml")
CFG_BAK  = Path("experiments/config.yaml.bak")
LOGS_DIR = Path("experiments/logs")
SAVED_DIR= Path("experiments/saved_models")

def sh(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def read_yaml(p): return yaml.safe_load(open(p))
def write_yaml(p,obj): p.parent.mkdir(parents=True, exist_ok=True); yaml.safe_dump(obj, open(p,"w"), sort_keys=False)

def parse_metrics(s):
    # 兼容: [TEST] F1=0.6161  Spearman=0.8001  Kendall=0.6553
    m = re.findall(r"\[TEST\]\s*F1\s*=\s*([0-9.]+).*?Spearman\s*=\s*([0-9.]+).*?Kendall\s*=\s*([0-9.]+)", s)
    if m:
        f1, rho, tau = map(float, m[-1]); return {"F1":f1,"Spearman":rho,"Kendall":tau}
    m2 = re.findall(r"F1\s*=\s*([0-9.]+)", s)
    return {"F1":float(m2[-1]) if m2 else None, "Spearman":None, "Kendall":None}

def run_fold(k):
    cfg = read_yaml(CFG_PATH)
    # 强制跨模态对齐（你已经生成了 384 维 SBERT）
    cfg.setdefault("model",{})
    cfg["model"]["use_text"] = True
    cfg["model"]["text_dim"] = cfg["model"].get("text_dim", 384)

    cfg.setdefault("data",{})
    cfg["data"]["dataset"] = "TVSum"
    cfg["data"]["fold"] = int(k)

    save_dir = SAVED_DIR / f"TVSum_fold_{k}"
    cfg.setdefault("train",{})
    cfg["train"]["save_dir"] = str(save_dir)

    write_yaml(CFG_PATH, cfg)

    print(f"\n=== Train TVSum Fold {k} ===")
    rc, out = sh([sys.executable, "-m", "train.train"])
    print(out)
    if rc != 0: return {"ok":False, "stage":"train", "stdout":out}

    print(f"=== Eval  TVSum Fold {k} ===")
    rc, eout = sh([sys.executable, "-m", "evaluation.test"])
    print(eout)
    if rc != 0: return {"ok":False, "stage":"eval", "stdout":eout}

    return {"ok":True, "metrics":parse_metrics(eout)}

def main():
    CFG_BAK.write_text(CFG_PATH.read_text())
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        per=[]
        for k in range(1,6):
            r = run_fold(k)
            if not r["ok"]:
                per.append({"fold":k, "ok":False})
                continue
            per.append({"fold":k, "ok":True, "metrics":r["metrics"]})

        valid = [x["metrics"]["F1"] for x in per if x.get("ok") and x.get("metrics") and x["metrics"]["F1"] is not None]
        summary = {
            "dataset":"TVSum",
            "folds": per,
            "F1_max": max(valid) if valid else None,
            "F1_avg": sum(valid)/len(valid) if valid else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        outp = LOGS_DIR / "fivefold_tvsum_results.json"
        json.dump(summary, open(outp,"w"), indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {outp}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        # 还原配置
        if CFG_BAK.exists():
            CFG_PATH.write_text(CFG_BAK.read_text())
            CFG_BAK.unlink()

if __name__ == "__main__":
    main()
