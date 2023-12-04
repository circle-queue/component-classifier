# component-classifier
The code repository for _Semi-supervised classification of engine parts_.

Unfortunately, no data can be made available due to confidentiality, meaning code will not execute in its current state.

### Entrypoints
If you're only interested in our implementation of FixMatch, you can see the [train pipeline](src\component_classifier\train_loop_builder.py).

[One-time scripts](src\component_classifier\one_time_scripts) perform preprocessing steps such as downloading images, creating static image embeddings, splitting data etc.

[main.py](src\component_classifier\main.py) is configured to open a browser with each of the plots from the paper. The underlying runs are executed via [dataset_study.py](src\component_classifier\dataset_study.py) and 
[ablation_study.py](src\component_classifier\ablation_study.py).

The one-class classification results are displayed using [occ_study.py](src\component_classifier\occ_study.py).

### Abstract
This paper tries to re-implement the semi-supervised method provided by FixMatch for it to be used to
fine-tune on new domain, engine component classification. A proprietary dataset was provided by MANES
Copenhagen, which includes a bunch of "no-class" samples totaling ≈100,000 images. About 10% of the
dataset has been labeled by a previously project. Our results are summarised in three experiments. Firstly,
One-class classification to have a method to sort out the "no-class" images in the dataset. Secondly, Engine
parts classification to see if the FixMatch method were an improvement compared to a normal supervised
approach including a abblation study of the data amount needed and the parameters of FixMatch. Thirdly,
FixMatch dataset comparison to see if our re-implementation could achieve the same trends as presented in the
FixMatch paper. We find promising results with respect to both rejecting "no-class" samples and classifying
engine components, however, the FixMatch method does not yield consistent improvements compared to
traditional supervised fine-tuning.

