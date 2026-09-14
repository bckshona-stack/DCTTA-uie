"""PromptIR 基线入口；UIE 对应 test_uie.py，共同实现位于 uie_tta.py。"""
from uie_tta import main as run


def main(argv=None):
    return run(argv, default_backbone="promptir", default_mode="baseline")


if __name__ == "__main__":
    main()
