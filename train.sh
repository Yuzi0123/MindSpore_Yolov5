# export DEVICE_NUM=8
# mpirun --allow-run-as-root -n ${DEVICE_NUM} --output-filename log_output --merge-stderr-to-stdout \
# python train.py \
#   --ms_strategy="StaticShape" \
#   --ms_amp_level="O0" \
#   --ms_loss_scaler="none" \
#   --ms_loss_scaler_value=1.0 \
#   --ms_optim_loss_scale=1.0 \
#   --ms_grad_sens=1.0 \
#   --overflow_still_update=True \
#   --clip_grad=False \
#   --device_target=GPU \
#   --is_distributed=True \
#   --epochs=300 \
#   --batch-size=32 > log.txt 2>&1 &

# python train.py \
#   --ms_strategy="StaticShape" \
#   --ms_amp_level="O2" \
#   --ms_loss_scaler="none" \
#   --ms_loss_scaler_value=1.0 \
#   --ms_optim_loss_scale=1.0 \
#   --ms_grad_sens=1.0 \
#   --overflow_still_update=True \
#   --clip_grad=False \
#   --device_target=GPU \
#   --device 0 \
#   --epochs=300 \
#   --batch-size=8 > log_level2.txt 2>&1 &

# python train.py \
#   --ms_strategy="StaticShape" \
#   --ms_amp_level="O0" \
#   --ms_loss_scaler="none" \
#   --ms_loss_scaler_value=1.0 \
#   --ms_optim_loss_scale=1.0 \
#   --ms_grad_sens=1.0 \
#   --overflow_still_update=True \
#   --clip_grad=False \
#   --device_target=GPU \
#   --device=2 \
#   --epochs=300 \
#   --batch-size=8 > log_level0.txt 2>&1 &

python train.py \
  --ms_strategy="StaticShape" \
  --ms_amp_level="O0" \
  --ms_loss_scaler="none" \
  --ms_loss_scaler_value=1.0 \
  --ms_optim_loss_scale=1.0 \
  --ms_grad_sens=1.0 \
  --overflow_still_update=True \
  --clip_grad=False \
  --device_target=GPU \
  --device=7 \
  --ms_mode="pynative" \
  --epochs=300 \
  --batch-size=8 > log_level0_pynative.txt 2>&1 &

