https://gist.github.com/m-klasen/651297e28199b4bb7907fc413c49f58f#file-detr_finetune-md

Get pretrained weights:
```python
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```
Remove class weights
```python
checkpoint = torch.load("detr-r50-e632da11.pth", map_location='cpu')
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
torch.save(checkpoint,"detr-r50_no-class-head.pth")
```
and make sure to set non-strict weight loading in `main.py`
```python
model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
```

Your dataset should ideally be in the COCO-format.
Make your own data-builder (alternatively rename your train/valid/annotation file to match the COCO Dataset)
In `datasets.coco.py` add:
```python
def build_your_dataset(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "annotations" / f'train.json'),
        "val": (root / "valid", root / "annotations" / f'valid.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
```
In `datasets.__init__.py` add your builder as an option:
```python
def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'your_dataset':
        return build_your_dataset(image_set, args)
    [...]
```
And lastly define how many classes you have in `models.detr.py`
```python
def build(args):
    [...]
    if args.dataset_file == 'your_dataset': num_classes = 4
    [...]
```
Run your model (example): 
`python main.py --dataset_file your_dataset --coco_path data 
    --epochs 50 --lr=1e-4 
    --batch_size=2 --num_workers=4 
    --output_dir="outputs" --resume="detr-r50_no-class-head.pth"`