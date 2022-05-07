python test_dasiamrpn.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --snapshot ../experiments/DaSiamRPN/SiamRPNBIG.model \
    --trackername DaSiamRPN

python test_lighttrack.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch LightTrackM_Subnet \
    --snapshot ../experiments/LightTrack/model.pth \
    --path_name back_04502514044521042540+cls_211000022+reg_100000111_ops_32 \
    --trackername LightTrack

python test_ocean.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch Ocean \
    --snapshot ../experiments/Ocean/model.pth \
    --trackername Ocean

python test_sesiamfc.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SESiamFC/config.yaml \
    --snapshot ../experiments/SESiamFC/model.pth \
    --trackername SE-SiamFC

python test_siamapn.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamAPN/config.yaml \
    --snapshot ../experiments/SiamAPN/model.pth \
    --trackername SiamAPN

python test_siamapn++.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamAPN++/config.yaml \
    --snapshot ../experiments/SiamAPN++/model.pth \
    --trackername SiamAPN++

python test_siamban.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamBAN/config.yaml \
    --snapshot ../experiments/SiamBAN/model.pth \
    --trackername SiamBAN

python test_siamcar.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamCAR/config.yaml \
    --snapshot ../experiments/SiamCAR/model.pth \
    --trackername SiamCAR

python test_siamdw_fc.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch SiamFCIncep22 \
    --snapshot ../experiments/SiamDW_FCIncep22/model.pth \
    --trackername SiamFC+_CI

python test_siamdw_fc.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch SiamFCNext22 \
    --snapshot ../experiments/SiamDW_FCNext22/model.pth \
    --trackername SiamFC+_CX

python test_siamdw_fc.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch SiamFCRes22 \
    --snapshot ../experiments/SiamDW_FCRes22/model.pth \
    --trackername SiamFC+_CR

python test_siamdw_rpn.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --arch SiamRPNRes22 \
    --snapshot ../experiments/SiamDW_RPNRes22/model.pth \
    --trackername SiamRPN+_CR

python test_siamfc++.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamFC++/siamfcpp_googlenet.yaml \
    --trackername SiamFC++

python test_siamgat.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamGAT/config.yaml \
    --snapshot ../experiments/SiamGAT/model.pth \
    --trackername SiamGAT

python test_siammask.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamMask/config.yaml \
    --snapshot ../experiments/SiamMask/model.pth \
    --trackername SiamMask

python test_siamrpn++.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamRPN++_alex/config.yaml \
    --snapshot ../experiments/SiamRPN++_alex/model.pth \
    --trackername SiamRPN++_A

python test_siamrpn++.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamRPN++_mobilev2/config.yaml \
    --snapshot ../experiments/SiamRPN++_mobilev2/model.pth \
    --trackername SiamRPN++_M

python test_siamrpn++.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --config ../experiments/SiamRPN++_r50/config.yaml \
    --snapshot ../experiments/SiamRPN++_r50/model.pth \
    --trackername SiamRPN++_R

python test_updatenet.py \
    --dataset UAVTrack112 \
    --datasetpath home/ \
    --snapshot ../experiments/UpdateNet/SiamRPNBIG.model \
    --update ../experiments/UpdateNet/vot2018.pth.tar \
    --trackername UpdateNet
