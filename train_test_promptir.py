"""PromptIR 适应入口；UIE 对应 adapt_target_uie.py，共同核心位于 uie_tta.py。"""
from uie_tta import main as run


def main(argv=None):
    return run(argv, default_backbone="promptir", default_mode="adapt")


if __name__ == "__main__":
    main()
