CONFIG="/root/GenLane/Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
# CONFIG="/root/GenLane/Mask2Former/configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
CHECKPOINT="/mnt/share_disk/LIV/generation_group/models/diffusers/public/m2f_mapillary_semantic.pkl"

# INPUT_DIR="/mnt/ve_share/chenminghua/dataset/0528/*"
INPUT_DIR="/root/GenLane/origin_images_10w.txt"
SAVE_DIR="/root/GenLane/output_2/"

echo $SAVE_DIR
mkdir -p $SAVE_DIR

# Function to run the demo.py script
run_demo() {
    local gpu_id=$1
    local start=$2
    local end=$3

    for i in $(seq -f "%02g" $start $end)
    do
        INPUT_DIR="/root/GenLane/output_${i}.txt"
        CUDA_VISIBLE_DEVICES=$gpu_id python ./demo/demo.py --config-file ${CONFIG} \
                                                           --input ${INPUT_DIR} \
                                                           --output ${SAVE_DIR} \
                                                           --opts MODEL.WEIGHTS ${CHECKPOINT} &
    done
}

# Run tasks on different GPUs
run_demo 3 0 3
run_demo 4 4 7
run_demo 5 8 11
run_demo 6 12 15
run_demo 7 16 19

# Wait for all background jobs to complete
wait

