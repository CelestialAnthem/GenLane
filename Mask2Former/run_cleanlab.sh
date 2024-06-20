#!/bin/bash

# Define the configurations
CONFIG_SEMANTIC="/root/GenLane/Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
CONFIG_PANOPTIC="/root/GenLane/Mask2Former/configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
CHECKPOINT="/mnt/share_disk/LIV/generation_group/models/diffusers/public/m2f_mapillary_semantic.pkl"

INPUT_FILE_LIST="/mnt/share_disk/songyuhao/seg_cleanlab/100.txt"
SAVE_DIR="/mnt/share_disk/songyuhao/seg_cleanlab/output"

echo $SAVE_DIR
mkdir -p $SAVE_DIR
mkdir -p $SAVE_DIR/seg
mkdir -p $SAVE_DIR/pred
mkdir -p $SAVE_DIR/res
mkdir -p $SAVE_DIR/comb
mkdir -p $SAVE_DIR/clean_res

# Split the input file into chunks of 1000 lines
split -l 10 $INPUT_FILE_LIST ${INPUT_FILE_LIST}_part_

# Process each chunk
for INPUT_PART in ${INPUT_FILE_LIST}_part_*; do

    file_part=$(basename "$INPUT_PART" | awk -F'_' '{print $NF}')
    echo "Processing $INPUT_PART"
    
    # Run the demo script
    python ./demo/demo_cleanlab.py --config-file ${CONFIG_SEMANTIC} --input ${INPUT_PART} --output ${SAVE_DIR} --dict ${file_part} --opts MODEL.WEIGHTS ${CHECKPOINT} 

    # Run the combine script for predictions and results
    python ./tools/postprocess/combine.py $SAVE_DIR/pred $SAVE_DIR/pred
    python ./tools/postprocess/combine.py $SAVE_DIR/res $SAVE_DIR/res

    rm $SAVE_DIR/pred/*.npy
    rm $SAVE_DIR/res/*.npy

    # Pair up pred and res files and run the cleanlab script for each pair
    for pred_file in $SAVE_DIR/pred*.npy; do
        file_name=$(basename -- "$pred_file")
        suffix="${file_name#pred_}"
        res_file="$SAVE_DIR/res_$suffix"
        base="${file_name%.npy}"

        if [[ -f "$res_file" ]]; then
            echo "Processing pair: $pred_file and $res_file"
            python /root/cleanlab/cleanlab/run.py "$pred_file" "$res_file" "$base" "$SAVE_DIR/clean_res/" $file_part True
        else
            echo "No matching res file for $pred_file"
        fi
    done

    rm $SAVE_DIR/*.npy
done

# Clean up temporary files
rm ${INPUT_FILE_LIST}_part_*

echo "Inference done."
