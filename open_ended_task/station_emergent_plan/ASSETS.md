# External assets

Download or mount the companion large-data bundle, then set:

```bash
export OPEN_RESEARCH_ASSETS_ROOT=/path/to/large-data-bundle
export DRC_TRAIN_DATA_PATH="$OPEN_RESEARCH_ASSETS_ROOT/station_emergent_plan/data/train_data_inputs_rewards_250m_episodes_list.pt"
export DRC_TEST_DATA_PATH="$OPEN_RESEARCH_ASSETS_ROOT/station_emergent_plan/data/test_data_inputs_rewards_250m_episodes_list.pt"
export DRC_CHECKPOINT_PATH="$OPEN_RESEARCH_ASSETS_ROOT/station_emergent_plan/checkpoints/ckp_actor_realstep250m.tar"
```

The train and test files are required for probing. The checkpoint is required
for the causal-intervention demo. That demo also needs compatible `thinker` and
`gym_sokoban` installations. The companion bundle preserves the Sokoban
resource archive at
`station_emergent_plan/runtime/sokoban/resources.zip`; install it according to
the selected `gym_sokoban` package's resource setup.

No path from the machine that assembled the bundle is required.
