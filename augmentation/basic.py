import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]   # ImageNet stats, for the pretrained backbone
STD = [0.229, 0.224, 0.225]

def get_train_transform(size=64):
    # No flips: they change the meaning of directional signs.
    return T.Compose([
        T.Resize((size, size)),
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        T.RandomPerspective(distortion_scale=0.1, p=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

def get_eval_transform(size=64):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])