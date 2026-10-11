# aerial-landuse-resolution-shift

A small study of how transfer learning strategies hold up when the pixel size (GSD) of aerial images changes.

## Question
When pixel size grows from 0.3 m toward satellite scale, which way of reusing an ImageNet ResNet-50 loses the least accuracy, and how many labelled target images close the gap?

## Data (public)
- UC Merced Land Use: 2,100 aerial tiles, 21 classes, 100 per class, 0.3 m pixels. Public domain imagery (USGS National Map). https://huggingface.co/datasets/torchgeo/ucmerced
- EuroSAT RGB: Sentinel-2 tiles, 10 m pixels, MIT licence. https://github.com/phelber/EuroSAT

## Method
- Strategies: S1 linear probe (frozen), S2 last-stage fine-tune, S3 fine-tune all, S4 fine-tune all with random pixel size (0.3/0.6/1.2/2.4 m) during training
- Pixel size simulated by box downsampling then smooth upsampling (factors 1, 2, 4, 8, 16 -> 0.3 to 4.8 m)
- 3 seeds (0, 1, 2), stratified random 70/10/20 split
- Backbone: ResNet-50, ImageNet-pretrained

## Results

### Experiment A: accuracy as pixel size grows (UC Merced, 21 classes, mean over 3 seeds)

| Strategy | 0.3 m | 0.6 m | 1.2 m | 2.4 m | 4.8 m (unseen by S4) | Drop |
|---|---|---|---|---|---|---|
| S1 linear probe | 93.7% | 93.0% | 87.8% | 57.5% | 24.4% | 69.3 pts |
| S2 last-stage fine-tune | 98.1% | 97.1% | 92.9% | 71.2% | 31.6% | 66.5 pts |
| S3 fine-tune all | 97.5% | 97.2% | 84.8% | 56.5% | 22.8% | 74.7 pts |
| S4 fine-tune all + GSD jitter | 97.4% | 97.5% | 97.5% | 95.1% | 72.1% | 25.2 pts |

Across 3 seeds, S4 lost the least accuracy from 0.3 m to 4.8 m (about 25 points), compared with 66 to 75 points for S1, S2 and S3. Among the non-augmented strategies, S2 held up best, better than fully fine-tuning (S3) or freezing the backbone (S1).

### Experiment B: simulated shift versus a real sensor shift (5 shared classes)

| Strategy | Same domain (0.3 m) | Simulated shift (4.8 m) | Real shift (EuroSAT, 10 m) |
|---|---|---|---|
| S1 | 98% | 68% | 32% |
| S3 | 98% | 45% | 27% |
| S4 | 99% | 80% | 32% |

S4's size-jitter augmentation closed most of the simulated-shift gap, but it barely helped on real satellite tiles: all three strategies landed around 27 to 32%, only a little above the 20% chance level for 5 classes. The simulated shift and the real shift behave very differently. Pixel size alone does not explain the real drop; EuroSAT also differs in sensor, geography, label definitions and ground footprint per tile.

### Experiment C: how many labelled target images close the gap?

| k per class | Aerial-pretrained then adapt | ImageNet-only, adapt directly |
|---|---|---|
| 0 | 27% | n/a |
| 5 | 78% | 62% |
| 10 | 83% | 76% |
| 20 | 91% | 92% |
| 50 | 96% | 96% |

With zero labelled satellite tiles, accuracy was only 27%. Just 5 to 10 labelled tiles per class closed most of the gap, and aerial-pretraining had a clear edge over ImageNet-only at that point. By 20 tiles per class the two starting points converged, and by 50 both reached 96%. A small amount of target-domain labelling closed the real gap far more effectively than pixel-size augmentation alone did in Experiment B.

## Limits
- Random split (UC Merced has no coordinates), so absolute accuracies are probably optimistic.
- Experiment A changes pixel size only, not haze, viewing angle, motion blur or light.
- This is tile-level classification, not block-level mapping.
- UC Merced pixels (0.3 m) are coarser than typical drone pixels; this covers an aerial to satellite-like range, not a drone range.
- Experiment B's satellite test is confounded: sensor, place, labels and footprint all differ, not just pixel size.
- One dataset family, one backbone (ResNet-50).
- Experiment B and C test sets are 100-400 tiles per seed; small differences are noise.

## Reproduce
python aerial_resolution_shift.py all
