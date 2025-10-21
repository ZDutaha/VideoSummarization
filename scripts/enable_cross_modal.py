import yaml, sys
def set_cross_modal(cfg_path="experiments/config.yaml", text_dim=384, use_contrastive=True):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["use_text"] = True
    cfg["model"]["text_dim"] = int(text_dim)
    cfg["model"]["use_contrastive"] = bool(use_contrastive)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print("[OK] Updated", cfg_path, "use_text=", cfg["model"]["use_text"],
          "text_dim=", cfg["model"]["text_dim"],
          "use_contrastive=", cfg["model"]["use_contrastive"])
if __name__=="__main__":
    text_dim = int(sys.argv[1]) if len(sys.argv)>1 else 384
    set_cross_modal(text_dim=text_dim, use_contrastive=True)
