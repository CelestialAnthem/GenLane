conda init
source activate
conda activate diffusers

CONFIG="./configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
# CONFIG="/root/GenLane/Mask2Former/configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
CHECKPOINT="/share/songyuhao/seg_cleanlab/models/m2f_mapillary_semantic.pkl"

# INPUT_DIR="/mnt/ve_share/chenminghua/dataset/0528/*"
# INPUT_FILE_LIST="/mnt/share_disk/songyuhao/seg_cleanlab/test/100.txt"
SAVE_DIR="./output_x/"

echo $SAVE_DIR
mkdir -p $SAVE_DIR
mkdir -p $SAVE_DIR/seg
mkdir -p $SAVE_DIR/pred
mkdir -p $SAVE_DIR/res
mkdir -p $SAVE_DIR/comb
mkdir -p $SAVE_DIR/clean_res

# Function to run the demo.py script
inf() {
    local gpu_id=$1
    local index=$2

    mkdir -p $SAVE_DIR/pred/$index
    mkdir -p $SAVE_DIR/res/$index
    mkdir -p $SAVE_DIR/seg/$index
    mkdir -p $SAVE_DIR/comb/$index
    mkdir -p $SAVE_DIR/clean_res/$index

    INPUT_FILE_LIST="/share/songyuhao/seg_cleanlab/online_data/$index/$gpu_id.txt"
    CUDA_VISIBLE_DEVICES=$gpu_id python ./demo/demo_cleanlab.py --config-file ${CONFIG} \
                                                                --input ${INPUT_FILE_LIST} \
                                                                --output ${SAVE_DIR} \
                                                                --dict $index  \
                                                                --opts MODEL.WEIGHTS ${CHECKPOINT} &
}

cleanlab(){
    local index=$1

    python ./tools/postprocess/combine.py $SAVE_DIR/pred/$index $SAVE_DIR/comb/$index pred
    python ./tools/postprocess/combine.py $SAVE_DIR/res/$index $SAVE_DIR/comb/$index res

    # Pair up pred and res files and run the cleanlab script for each pair
    for pred_file in $SAVE_DIR/comb/$index/pred*.npy; do
        file_name=$(basename -- "$pred_file")
        suffix="${file_name#pred_}"
        res_file="$SAVE_DIR/comb/$index/res_$suffix"
        base="${file_name%.npy}"

        if [[ -f "$res_file" ]]; then
            echo "Processing pair: $pred_file and $res_file"
            python ../../cleanlab/cleanlab/run.py "$pred_file" "$res_file" "$base" "$SAVE_DIR/clean_res/$index/" $index True 0.1
        else
            echo "No matching res file for $pred_file"
        fi
    done
}

# Run tasks on different GPUs
inf_8() {
    local index=$1
    inf 0 $index
    inf 1 $index
    inf 2 $index
    inf 3 $index
    inf 4 $index
    inf 5 $index
    inf 6 $index
    inf 7 $index
}



for i in {0..49}
do
    inf_8 $i
    wait
    cleanlab $i
    cp -r ./output_x/clean_res/$i /share/songyuhao/seg_cleanlab/res
done

# Wait for all background jobs to complete
wait

