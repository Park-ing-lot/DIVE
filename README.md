# Deep DIVE into Visual Commonsense Generation: Diverse and Descriptive Reasoning about stories
This is the repository of DIVE.

Most of the code in this repo are copied/modified from [KM-BART](https://github.com/FomalhautB/KM-BART)


## Installation
1. Clone the repository recursively
    ```
    git clone --recursive https://github.com/CallessCaller/DIVE.git
    ```
2. Install requirements
    ```
    pip install -r requirments.txt
    ```
    You can change your cuda or torch version.

## Data preparation

### Visual Commonsense Graphs
1. Download the text data and visual features from [VisualCOMET](https://visualcomet.xyz) and decompose them into 'data'.
    ```
    mkdir data
    cd data
    unzip visualcomet.zip
    unzip features.zip
    ```
2. Download the images from [VCR](https://visualcommonsense.com) and decompose images into 'data'.
    ```
    cd data
    unzip vcr1images.zip
    ```

3. Download the filtered annotations from [here](https://drive.google.com/file/d/1BiqbBRI3X2usf7bBWC-n1HMMqX-BvJCy/view?usp=sharing) and put them into 'data/visualcomet/'.
    ```
    cd data
    unzip filtered_vcg.zip
    mv filtered_vcg/. visualcomet/
    ```

4. Prepare training data.
    ```
    mkdir data/dive
    python -m scripts.prepare_vcg --output_dir data/dive --annot_dir data/visualcomet/ --data_dir data/vcr/vcr1images/
    ```

## Training
1. Train DIVE.
    ```
    mkdir output
    python vcg_train_css.py --data_dir data/dive/ --checkpoint facebook/bart-base --validate_score --validate_loss --dropout 0.3 --batch_size 256 --lr 5e-5
    ```

## Generation
1. Generation with counterfactual-aware inference generation.
    ```
    mkdir generated
    python vcg_generate_css_gt.py --do_sample --data_dir data/dive/ --output_file generated/dive --checkpoint {model path}
    ```

## Evalaution
1. Evauation the generated infererences.
    ```
    python vcg_eval.py --generation generated/dive --reference data/dive/val_filtered_ref.json --annotation data/dive/train_ref.json
    ```