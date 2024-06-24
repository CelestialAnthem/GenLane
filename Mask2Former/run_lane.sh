CONFIG="/root/GenLane/Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
# CONFIG="/root/GenLane/Mask2Former/configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
CHECKPOINT="/mnt/share_disk/LIV/generation_group/models/diffusers/public/m2f_mapillary_semantic.pkl"

# INPUT_DIR="/mnt/ve_share/chenminghua/dataset/0528/*"
INPUT_DIR="/mnt/ve_share/songyuhao/dm_test.txt"
SAVE_DIR="/root/GenLane/dm"

echo $SAVE_DIR
mkdir $SAVE_DIR
mkdir $SAVE_DIR/img
mkdir $SAVE_DIR/pred
mkdir $SAVE_DIR/res


python ./demo/demo.py --config-file ${CONFIG} \
                         --input ${INPUT_DIR} \
                         --output ${SAVE_DIR} \
                         --opts MODEL.WEIGHTS ${CHECKPOINT}



echo "Inference done."