## Quickstart

### Install
pip install -r requirements.txt

### Train (PPO baseline)
python train.py --env Hopper-v5 --seed 0 --total-steps 2000000 --n-envs 8 --logdir runs/ppo_hopper_seed0 --save checkpoints/ppo_hopper_seed0.zip

### TensorBoard
tensorboard --logdir runs

### Eval (CSV)
python eval.py --env Hopper-v5 --model checkpoints/ppo_hopper_seed0.zip --episodes 20 --seed 0 --out results/eval/ppo_seed0.csv

### Record Video
python record_video.py --env Hopper-v5 --model checkpoints/ppo_hopper_seed0.zip --video videos/ppo_seed0/hopper_ppo.mp4 --steps 3000 --seed 0
