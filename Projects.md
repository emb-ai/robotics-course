### World Model Trajectory Selection

**Core features**:
- World Model (Video Generation)
- Probably Video Games too, not only robotic data
- A lot of gpu compute (hopefully)

**Brief description**:
Given a diffusion based video generator, estimate which trajectories are good candidates for World Model Predictive Control. Apply inverse dynamics to estimate quality of the selection

**Goals** (Either):
- Probabilistic trajectory selection (log-likelihood, uncertainty, etc.)
- VLM trained on human preference or sim synthetic data

**Availability**:
- Any location

### Lerubick

**Core features**:
- Finite State Machine Planning
- Deterministic pre-recorded/calibrated actions
- RGB+Depth Perception
- Bi-manual manipulation

**Brief description**:
Two robot arms solve the 2x2 rubicks cube (maximum 14 90-degree moves from any state to solved state). Observations come from Intel Real Sense camera. Arms positions are adjusted using aruco markers, executing predefined actions otherwise. Mostly low-lewel classical control with error correction + FSM planning.

**Goals** (Either):
- pre-record actions and show that they accumulate error in execution fairly slowly
- make digital twin with configuration following real-world setup via perception 

**Availability**:
- Moscow Preferred

### Lechess (not lichess, not leeches)

**Core features**:
- Precision end-effector positioning
- learnable actions (i believe CNN policy will do the best)
- RGB Perception
- chess <3

**Brief description**:
Robot Arm makes specified chess moves. Board is recognised using an overhead rgb camera, challenges involve board state extraction and precise collision free chess piece manipulation.

**Goals** (Either):
- extract board state with high accuracy
- train policy to pick chess pieces on mostly empy board (data collection or even RL-HIL)

**Availability**:
- Moscow Preferred

### UR10

**Core features**:
- BIG AND FREAKING EXPENSIVE ROBOT ARM

**Brief description**:
Turn it on, gather all data, re-create 3d scene, compute grasping position and execute pick-and-place.

**Goals**:
- do just as described, probably enough time

**Availability**:
- Moscow Preferred

### VR Teleop

**Core features**:
- VR Headset
- Teleoperation
- RGB+Depth Perception
- Dexterous Robot Hand
- Joint mapping between robot and human

**Brief description**:
Take Meta Quest Headset with it's automatic hand recognition. Extract all joints and map them to a simpler robot hand. Relay actions first in sim, then in real world.

**Goals**:
- do just as described, probably enough time

**Availability**:
- Moscow Preferred

### VLA Trajectory Collection + Finetune 

**Core features**:
- VLA
- Trajectory Collection
- A lot of gpu compute (hopefully)

**Brief description**:
Very straightforward, collect trajectory data via teleoperation, finetune VLA on it, evaluate on real world.

**Goals** (Either):
- demonstrate generalization.
- demonstrate instruction following.

**Availability**:
- Moscow
- Minsk

### Robot Tool Calling

**Core features**:
- Agents
- Huge simulation

**Brief description**:
Give an agent the embodiment of a household robot. Full control, chat with it like it's your maid and tamagochi simultaneously. But in sim -_-

**Goals**:
- Any simple and reasonable tool set.

**Availability**:
- Anywhere

### Maxwell's Demon Mean Field

**Contact Sonya**

**Goals**:
- probably you will have to write down probabilistic description of pucks distribution wrt action policy

**Availability**:
- Phystech :)

### Homework 4 (clown face)

**Contact me...**

**Availability**:
- Anywhere