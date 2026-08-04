source beam_optimization/.venv/bin/activate

python -m beam_optimization train_policies \
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output beam_optimization/results/train/rl/sb3_sac_h30_smooth050_action030_seed42 \
  --rl-steps 200000 \
  --max-ep-steps 30 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 1 \
  --eval-every 1000 \
  --eval-episodes 5 \
  --distance-penalty-weight 0.02 \
  --action-penalty-weight 0.30 \
  --action-smoothness-penalty-weight 0.50 \
  --score-regression-penalty-weight 5.0 \
  --skip sac_custom td3_custom ppo_custom ddpg_custom a2c_custom reinforce_custom trpo_custom ppo td3 ddpg a2c dyna svg