from corefai.utils.configs import Config



def init(parser):
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='num of threads')
    parser.add_argument('--workers', '-w', default=0, type=int, help='num of processes used for data loading')
    parser.add_argument('--cache', action='store_true', help='cache the data for fast loading')

    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)

    resolve(args)


def resolve(args):
    Resolver = args.pop('Resolver')
    if args.mode == 'train':
        resolver = Resolver.load(**args) if args.checkpoint else Resolver(**args)
        resolver.train(**args)
    elif args.mode == 'evaluate':
        resolver = Resolver.load(**args)
        resolver.evaluate(**args)
    elif args.mode == 'predict':
        resolver = Resolver.load(**args)
        resolver.predict(**args)