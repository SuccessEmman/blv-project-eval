# Sampling and blur analysis : Ebere and Samuel

The final sample contains 250 images: 70 clear, 70 blurry, 60 low-light and 50 manually approved text images.

A separate 200-image pilot contains 100 images for choosing the blur threshold and 100 for evaluating it. It does not overlap with the final sample.

## Files

- `notebooks/BLV Project.ipynb`: loads and validates saved results and checks image readability.
- `notebooks/BLV Project_history.ipynb`: earlier workflow, including Colab-specific code. Review and adapt it before running.
- `outputs/final_sample/`: final sample, blur scores, settings and inspection notes.
- `outputs/blur_calibration/`: pilot records, threshold and evaluation results.
- `requirements.txt`: package versions from Ebere’s Python environment.

## Local setup

Use Python 3.12 and install the packages from the repository root in your chosen environment:

```bash
python -m pip install -r sampling/requirements.txt
```

Open the current notebook and select that environment as its kernel.

The image files are stored separately from GitHub. Obtain the train and validation images from the shared project data. In Cell 4, change `IMAGE_ROOT` to the folder containing your extracted `train` and `val` folders. Its current value points to Ebere’s local image location.

Run the four cells in order. Expected results: 250 sample records, 200 pilot records, no overlap and 450 readable images.

## Blur results

The selected threshold is 164.67810814512066; lower scores are predicted as blurry. Balanced accuracy was 77% during calibration and 71% on the reserved evaluation images.

Original annotation-based groups remain unchanged. Blur scores provide supplementary information. The threshold has not been validated for low-light or text images.