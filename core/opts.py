import argparse


def add_global_arguments(parser):

    parser.add_argument("--dataset", type=str, help="Determines dataloader to use (only Pascal VOC supported)")
    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")
    parser.add_argument('--work-dir', type=str, help='workspace for single config file.')

    parser.add_argument('--num-gpus', default=1, type=int, help='number of gpus for distributed training')
    parser.add_argument('--port', default='10002', type=str, help='port used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
        'e.g. [key] [value] [key] [value]',
        default=[],
        nargs='+')

    parser.add_argument("--random-seed", type=int, default=64, help="Random seed")


def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)

    return args
