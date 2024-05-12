TARGET_DIR=$(pwd)/weights/face-upscale

[ -f $TARGET_DIR/RealESRGAN_x2plus.pth ] || wget -nv -P $TARGET_DIR https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
[ -f $TARGET_DIR/detection_Resnet50_Final.pth ] || wget -nv -P $TARGET_DIR https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
[ -f $TARGET_DIR/parsing_parsenet.pth ] || wget -nv -P $TARGET_DIR https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

[ -f $TARGET_DIR/GFPGANv1.3.pth ] || wget -nv -P $TARGET_DIR https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
[ -f $TARGET_DIR/GFPGANv1.4.pth ] || wget -nv -P $TARGET_DIR https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
[ -f $TARGET_DIR/RestoreFormer.pth ] || wget -nv -P $TARGET_DIR https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth
