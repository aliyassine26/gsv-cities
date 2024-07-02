# Enhanced Visual Place Recognition with GSV-Cities

This repository contains the code and resources for our enhanced visual place recognition project, building on the foundational work of the GSV-Cities report. Our extensions include various miners, loss functions, and aggregation modules to improve the performance of visual geo-localization systems

- The dataset GSV XS is hosted on [[Kaggle](https://www.kaggle.com/datasets/giovannimonco22/gsv-xs)].
- The datasets Tokyo XS and SF XS are available on [[Google Drive](https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf?usp=share_link)]
- Training and testing can be run from `runner.ipynb`, the code is commented and should be clear. Feel free to open an issue if you have any question.

---

## **Summary of the paper**

Our paper, "Leveraging Losses, Miners, and Aggregators for Enhanced Visual Place Recognition with GSV-Cities," explores advancements in Visual Geo-Localization (VG), which involves identifying an image’s location by comparing it against a comprehensive database of geo-tagged images. We build on the foundational work of the GSV-Cities framework by experimenting with various components to enhance the performance of image retrieval frameworks.

Key contributions include:

- **Backbone Network** : Using ResNet-18 as the baseline network.
- **Loss Functions** : Experimenting with MultiSimilarity Loss, Angular Loss, Triplet Margin Loss, FastAP Loss, and NTXent Loss to improve retrieval accuracy.
- **Miners** : Incorporating MultiSimilarityMiner, UniformHistogramMiner, AngularMiner, BatchHardMiner, and BatchEasyHardMiner to select informative samples during training.
- **Aggregation Modules** : Exploring AVG, GeM, MixVPR, Cosplace, and ConvAP to enhance embedding quality.

Our results demonstrate significant improvements in VG tasks by optimizing the combination of these components. The best-performing configuration utilizes Cosplace aggregation with MultiSimilarityMiner and MultiSimilarityLoss, achieving state-of-the-art results on benchmark datasets. The code and datasets used in this study are available in this repository, enabling further research and development in the field of visual place recognition.

## GSV-Cities XS dataset overview

- GSV-Cities contains ~530,000 images representing ~62,000 different places, spread across multiple cities around the globe.
- All places are physically distant (at least 100 meters between any pair of places).

The following figure represents the adapted architecture
![architecture](https://github.com/aliyassine26/gsv-cities/blob/main/images/architecture.jpg)


## Setting

The project can be run both in a local environment (if at least a GPU is present) or on a google colab. refer to the file runner.ipynb in the notebooks folder.

### Google colab

1. Clone the project repository
   `git clone https://github.com/aliyassine26/gsv-cities.git `
2. Install the required libraries

   ```python
   !pip install -q condacolab
   import condacolab
   condacolab.install()
   # Create the Conda environment in the specified folder
   !conda create --prefix "$env_path" python=3.8.4 -y
   !source activate /content/gsv_env_main && pip install -r /content/drive/MyDrive/requirements2.txt
   pip install -r /content/drive/MyDrive/GSV_CITIES_DATA/requirements.txt
   ```

3. Add the needed datasets

   - gsvxs: download it and copy it under /content
   - sfxs: download it and copy it under /content
   - tokyo: download it and copy it under /content

4. change directory to be in gsv-cities repo
   !cd gsv-cities
5. Go to the run section

# Example Run
```python
source activate /content/gsv_env_main && python3 main.py \
   --batch_size 100
   --img_per_place 4
   --min_img_per_place 4
   --shuffle_all False
   --random_sample_from_each_place True
   --image_size "(320, 320)"
   --num_workers 8
   --show_data_stats True
   --val_set_names '["sfxs_val"]'
   --test_set_names '["sfxs_test", "tokyoxs_test"]'
   --backbone_arch "resnet18"
   --pretrained True
   --layers_to_freeze -1
   --layers_to_crop '[4]'
   --agg_arch "Gem"
   --agg_config '{"p":3}'
   --lr 0.0002 --optimizer "adam"
   --weight_decay 0
   --momentum 0.9
   --warmpup_steps 600
   --milestones '[5, 10, 15, 25]'
   --lr_mult 0.3
   --loss_name "MultiSimilarityLoss"
   --miner_name "BatchHardMiner"
   --miner_margin 0.1
   --faiss_gpu True
   --monitor "sfxs_val/R1"
   --filename "{self.backbone_arch}\_epoch({epoch:02d})\_step({step:04d})\_R1[{pitts30k_val/R1:.4f}]\_R5[{sfxs_val/R5:.4f}]"
   --auto_insert_metric_name False
   --save_weights_only True
   --save_top_k 3 --mode "max"
   --accelerator "gpu"
   --devices 1
   --default_root_dir "./LOGS/{self.backbone_arch}"
   --num_sanity_val_steps 0
   --precision 32
   --max_epochs 8
   --check_val_every_n_epoch 1
   --reload_dataloaders_every_n_epochs 1
   --log_every_n_steps 20
   --fast_dev_run False
   --model_path ''
   --experiment_phase "train"
```
The configuration argument are:

- `{batch_size}`: Number of images per batch.
- `{img_per_place}`: Number of images per place.
- `{min_img_per_place}`: Minimum number of images per place.
- `{shuffle_all}`: Boolean to shuffle all data.
- `{random_sample_from_each_place}`: Boolean to randomly sample from each place.
- `{image_size}`: Size of the images, e.g., "(320, 320)".
- `{num_workers}`: Number of workers for data loading.
- `{show_data_stats}`: Boolean to show data statistics.
- `{val_set_names}`: Names of the validation sets, e.g., '["sfxs_val"]'.
- `{test_set_names}`: Names of the test sets, e.g., '["sfxs_test", "tokyoxs_test"]'.
- `{backbone_arch}`: Architecture of the backbone network, e.g., "resnet18".
- `{pretrained}`: Boolean for using pretrained weights.
- `{layers_to_freeze}`: Layers to freeze during training, e.g., -1.
- `{layers_to_crop}`: Layers to crop, e.g., '[4]'.
- `{agg_arch}`: Aggregation architecture, e.g., "Gem".
- `{agg_config}`: Configuration for the aggregation module, e.g., '{"p":3}'.
- `{lr}`: Learning rate.
- `{optimizer}`: Optimizer type, e.g., "adam".
- `{weight_decay}`: Weight decay for the optimizer.
- `{momentum}`: Momentum for the optimizer.
- `{warmpup_steps}`: Number of warmup steps.
- `{milestones}`: Milestones for learning rate scheduler, e.g., '[5, 10, 15, 25]'.
- `{lr_mult}`: Learning rate multiplier.
- `{loss_name}`: Name of the loss function, e.g., "MultiSimilarityLoss".
- `{miner_name}`: Name of the miner, e.g., "BatchHardMiner".
- `{miner_margin}`: Margin for the miner.
- `{faiss_gpu}`: Boolean to use FAISS on GPU.
- `{monitor}`: Metric to monitor, e.g., "sfxs_val/R1".
- `{filename}`: Filename template for saving models.
- `{auto_insert_metric_name}`: Boolean to auto-insert metric name in filenames.
- `{save_weights_only}`: Boolean to save weights only.
- `{save_top_k}`: Number of top models to save.
- `{mode}`: Mode for saving best models, e.g., "max".
- `{accelerator}`: Accelerator type, e.g., "gpu".
- `{devices}`: Number of devices (GPUs) to use.
- `{default_root_dir}`: Default root directory for logs and models.
- `{num_sanity_val_steps}`: Number of sanity validation steps.
- `{precision}`: Precision for training, e.g., 32.
- `{max_epochs}`: Maximum number of epochs.
- `{check_val_every_n_epoch}`: Frequency of validation checks.
- `{reload_dataloaders_every_n_epochs}`: Frequency of reloading data loaders.
- `{log_every_n_steps}`: Frequency of logging.
- `{fast_dev_run}`: Boolean for fast development run.
- {model_path}: trained model path for testing
- `{experiment_phase}`: Experiment phase, e.g., "train", "test", or "all".

# Results

Results from various combinations of miners, losses and aggregation can be found in the file VPR Runs for further insights

# Visualization

Visualization of the prediction files resultiing from testing can be visualized using the file images.ipynb in the notebook folder

## Acknowledgements

This project builds on the foundational work of Ali-bey, Amar, Chaib-draa, Brahim, and Giguère, Philippe in their paper "GSV-Cities: Toward Appropriate Supervised Visual Place Recognition". We extend our gratitude to the original authors for their significant contributions to the field.We would like to extend our sincere gratitude to Professor Barbara Caputo and Teaching Assistant Gabrielle Trivigno for their invaluable guidance, support, and insights throughout the course of this project. Their expertise and encouragement have been instrumental in the successful completion of our work.

## References

Refer to the [Enhanced VPR Report](https://drive.google.com/file/d/16CRoMpZiEWOgb9R-ypXJBS4GIVcbOLu-/view?usp=drive_link) for a comprehensive list of references used in this project.
