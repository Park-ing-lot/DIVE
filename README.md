# Deep DIVE into Visual Commonsense Generation: Diverse and Descriptive Reasoning about Stories
This is the repository of DIVE.

Most of the codes in this repo are copied/modified from [KM-BART](https://github.com/FomalhautB/KM-BART)


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

## Data Preparation

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

3. unzip the filtered annotations and unique/novel reference file (for the validation) and put them into 'data/visualcomet/'.
    ```
    unzip filtered_vcg.zip
    mv filtered_vcg/. data/visualcomet/
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
    python vcg_generate_css_gt.py --do_sample --top_p 0.9 --num_beams 1 --data_dir data/dive/ --output_file generated/dive --checkpoint {model path}
    ```

## Evaluation
1. Evaluating the generated inferences.
    ```
    python vcg_eval.py --generation generated/dive --reference data/dive/val_ref.json --annotation data/dive/train_ref.json
    ```
2. Evaluating with a unique validation subset.
    ```
    python vcg_eval.py --generation generated/dive --reference data/visualcomet/val_unique_ref.json --annotation data/dive/train_ref.json
    ```
3. Evaluating with a novel validation subset.
    ```
    python vcg_eval.py --generation generated/dive --reference data/visualcomet/val_novel_ref.json --annotation data/dive/train_ref.json
    ```
