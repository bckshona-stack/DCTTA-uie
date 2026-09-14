"""UIE 测试入口，对应原项目的 test_promptir.py。

默认直接评价源模型；--mode x8 使用八视图集成，--mode evaluate 评价已适应权重。
模型、数据和评价实现统一放在 uie_tta.py。
"""
from uie_tta import main as run


def main(argv=None):
    return run(argv, default_backbone="nu2net", default_mode="baseline")


if __name__ == "__main__":
    main()
