# Multi-task Seals Auscultation Classifier
## Abstract
Pulmonary auscultation is one of the most valuable and fundamental tools available to veterinarians to assess lung conditions in animals quickly. Despite all the advances in the medical field, electronic chest auscultation can sometimes represent an unreliable method due to the non-stationary property of lung sounds. Automating this process can aid the clinicians in their diagnosis process and hopefully improve the survival rate of seals that arrive at the Pieterburen Zeehondencentrum in Groningen. One way to do this is through means of deep learning. However, most of the time, audio classification tasks are usually treated as independent tasks. As lung sounds are known to be related to one another in some form or shape, a single-task approach, by focusing only on one task, misses most of the information necessary to make a difference when classifying closely related tasks. This paper aims to show the potential of multi-task learning (MTL) in the context of seals lung sound classification. We proposed three different types of multi-task convolutional neural network architectures. These models are evaluated on the mel-cepstral coefficients (MFCCs) features and per-channel energy normalized spectrograms (PCEN). Experiments are conducted on a dataset of 142 samples gathered from both the left and right lungs of multiple seals. The two types of abnormal sounds present in this dataset are Wheezing and Rhonchus. Results show that the MFCC features, together with our custom-built CNN obtained an accuracy of 73\% when classifying wheezing and 63\% in the case of rhonchus, outperforming the classification of PCEN images by 15\% and 25\%  respectively. Lastly, the same model manage to obtain a survival prediction accuracy of 80\% and succesfully showing the potential of MTL in auscultation classification.

## Project Structure
    .
    ├── notebooks               # Notebooks used for sketching
    ├── src                     # Main directory
    │   ├── data                # Audio files and metadata
    │   ├── features            # Data processing scripts
    │   ├── model               # Models, training and prediction scripts
    │   ├── plots               # Figures, graphs, etc...
    │   ├── reports             # Classification repoorts, results
    │   └── util                # Utility scripts
    └── README.md

## Getting started
All the scripts in this project are properly documented and easy to use/change.

### Model's folder
The most importan scripts can be found in the `model` folder. In here the `training.py` script is the one responsible for everything. It trains the model and generates the results based on the prediction on the test input.