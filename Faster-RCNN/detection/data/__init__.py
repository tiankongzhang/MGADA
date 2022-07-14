def collate_fn(batch):
    """
    Args:
        batch: list of tuple, 0 is images, 1 is img_meta, 2 is target
    Returns:
    """
    batch = list(zip(*batch))
    imgs = batch[0]
    img_metas = batch[1]
    targets = batch[2]

    if len(imgs) == 1:
        batched_imgs = imgs[0].unsqueeze_(0)
    else:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))

        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs.contiguous(), img_metas, targets
