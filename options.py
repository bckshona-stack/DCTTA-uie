"""UIE / PromptIR 共用参数；导入模块不解析命令行，入口显式调用 parse_args。"""

import argparse
from pathlib import Path


EXPERIMENT_DIR = str(Path(__file__).resolve().parent / 'experiments' / 'UIEBblue_weaktostrong_NU2Net')


def parse_diagnostic_args(argv=None):
    parser = argparse.ArgumentParser(description='UIE static audit, observer acceptance, full trace and offline evaluation')
    parser.add_argument('--phase', choices=('audit', 'accept', 'run', 'evaluate', 'inspect', 'fisher_probe',
                                           'numeric_control', 'compare_numeric'), required=True)
    parser.add_argument('--protocol', choices=('fixed_fisher_strict','legacy'), default='fixed_fisher_strict')
    parser.add_argument('--experiment_dir', default=EXPERIMENT_DIR)
    parser.add_argument('--output_dir', default=EXPERIMENT_DIR + '_diagnostic_v2')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--acceptance_steps', type=int, default=22)
    parser.add_argument('--severity_scores', help='Optional original severity_scores.csv for independent review')
    parser.add_argument('--inspect_steps', type=int, nargs='+', default=[1,20,133])
    args = parser.parse_args(argv)
    if Path(args.output_dir).resolve() == Path(args.experiment_dir).resolve():
        parser.error('Diagnostic output must differ from the original experiment')
    return args


def parse_source_args(argv=None):
    """源训练参数独立解析，避免把 DCTTA 学习率/损失配置用于源训练。"""
    parser = argparse.ArgumentParser(description='NU2Net L1 source training / paired manifest preparation')
    parser.add_argument('--experiment_dir', default=EXPERIMENT_DIR)
    parser.add_argument('--prepare_only', action='store_true')
    # 下划线属于文件夹名称，不是目录分隔符：raw-890_4classes_severity / blue_tint。
    raw = r'F:\qmh\DCTTA-uie\dataset\UIEB\raw-890_4classes_severity\blue_tint'
    ref = r'F:\qmh\DCTTA-uie\dataset\UIEB\reference-890_4classes_severity\blue_tint'
    parser.add_argument('--source_raw', default=raw + r'\weak')
    parser.add_argument('--source_ref', default=ref + r'\weak')
    parser.add_argument('--target_raw', default=raw + r'\strong')
    parser.add_argument('--target_ref', default=ref + r'\strong')
    parser.add_argument('--expected_source', type=int, default=134)
    parser.add_argument('--expected_target', type=int, default=133)
    parser.add_argument('--val_count', type=int, default=27)
    parser.add_argument('--exclude_source_ids', nargs='*', default=['582_img_.png', '648_img_.png'],
                        help='Approved source manifest exclusions; originals remain untouched')
    parser.add_argument('--scene_groups', help='Optional CSV: domain,sample_id,scene_id; manual scene grouping')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--basic_channel', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', help='Path to last.pth; resume at a completed epoch boundary')
    parser.add_argument('--stop_after_epoch', type=int, help='Controlled interruption for smoke tests; does not change scheduler horizon')
    args = parser.parse_args(argv)
    if (min(args.expected_source, args.expected_target, args.val_count, args.epochs,
            args.batch_size) < 1 or args.num_workers < 0 or args.lr <= 0
            or args.basic_channel < 4 or args.patch_size < 32 or args.patch_size % 8):
        parser.error('Invalid source training or preparation parameters')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error('stop_after_epoch must be within 1..epochs')
    return args


def parse_target_args(argv=None):
    """只解析实验编排参数，剩余参数交给原 DCTTA 参数解析器。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--experiment_dir', default=EXPERIMENT_DIR)
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser.parse_known_args(argv)


def parse_args(argv=None, default_backbone="nu2net", default_mode="adapt"):
    parser = argparse.ArgumentParser(description="DCTTA code-reference migration to UIE")
    parser.add_argument("--backbone", choices=("nu2net", "promptir"), default=default_backbone)
    parser.add_argument("--mode", choices=("baseline", "x8", "adapt", "evaluate"), default=default_mode)
    parser.add_argument("--model_path", help="Source weights, or adapted weights for --mode evaluate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--lq_dir")
    parser.add_argument("--gt_dir")
    parser.add_argument("--manifest", help="CSV: sample_id,raw_path,reference_path (optional)")
    parser.add_argument("--data_root", help="Base for manifest paths; defaults to manifest's directory")
    parser.add_argument("--dataset_name")
    parser.add_argument("--result_dir")
    parser.add_argument("--metrics_filename", default="metrics.txt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, choices=(1,), default=1)
    parser.add_argument("--patch_size", type=int, default=320,
                        help="NU2Net adaptation crop upper bound; small images use their largest supported multiple of 8. PromptIR uses a fixed crop.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--shuffle", type=int, choices=(0, 1), default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--teacher_weight", type=float, default=5.0)
    parser.add_argument("--ema_decay", type=float, default=0.45)
    parser.add_argument("--fisher_ratio", type=float, default=0.6)
    parser.add_argument("--compute_fisher", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fisher_cache_path")
    parser.add_argument("--recompute_fisher", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_rddm_results", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.num_workers < 0 or args.iterations < 1 or args.lr <= 0:
        parser.error("Invalid limit, num_workers, iterations or learning rate")
    if not 0 <= args.fisher_ratio <= 1 or not 0 <= args.ema_decay <= 1 or args.teacher_weight < 0:
        parser.error("Invalid Fisher ratio, EMA decay or teacher weight")
    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("Adam betas must be in [0, 1)")
    if args.manifest and (args.lq_dir or args.gt_dir):
        parser.error("Use either --manifest or --lq_dir/--gt_dir")
    if Path(args.metrics_filename).name != args.metrics_filename or not args.metrics_filename.endswith(".txt"):
        parser.error("metrics_filename must be a .txt filename")
    return args
