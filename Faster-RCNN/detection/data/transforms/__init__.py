from .transforms import *

TRANSFORMS = {
    'random_flip': random_flip,
    'resize': resize,
    'normalize': normalize,
    'collect': collect,
    'pad': pad,
}


def build_transforms(transforms):
    results = []
    for cfg in transforms:
        args = cfg.copy()
        name = args.pop('name')
        transform = TRANSFORMS[name](**args)
        results.append(transform)
    return compose(results)
