# LLM-distill: Training Pipeline

This module handles reinforcement learning and knowledge distillation.

## Strategy

1. **Experience Collection**: Run the agent on tasks, collect trajectories
2. **Reward Model**: Score trajectories based on task completion
3. **GRPO Training**: Use Group Relative Policy Optimization to improve the model

## Files

- `experience.py` - Experience buffer for trajectory storage
- `rewards.py` - Reward scoring functions (TODO)
- `grpo.py` - GRPO training loop (TODO)

## Future Work

- [ ] Implement reward model
- [ ] GRPO training with collected experiences
- [ ] Distillation from larger models (teacher-student)
- [ ] Online RL with tool-use feedback
