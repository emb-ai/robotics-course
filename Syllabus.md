# AI in Robotics

#### What is this course:

Цель курса — дать теоретические знания и практический опыт, которые позволят принимать участие в передовой исследовательской и прикладной разработке искисственного интелекта для роботов. Причем как в масштабе самодельных роботов/маленьких лабораторий, так и в индустриальных масштабах series A стартапов

#### People:

* primary TA: Andrei Spiridonov (tg: @Penchekrak)

#### Prerequisites:

* **Basic math**: linear algebra, calculus, optimization
* **Coding**: Python, C++, graph algorithms, numerical methods
* **AI**: CV+3D, NLP, RL

#### Legend:

* **Lectures** are lectures
* **Seminars** are **interactive** lectures with **live execution of code** examples
* **Homeworks** are **math/coding problems with automatic grading** (mostly)
* **Modern** = extensive use of **large models, vast training, end-to-end** methods
* **Projects** are mandatory in-curriculum **team-work** projects that require most **work from scratch** (compared to homeworks), **strong supervision**, and are a primary way for a **student to interact with physical hardware and robots**

# Programme

## week 1: Intro and Math 1

### Lecture

Overview of robotics: sub-fields, startups/companies, embodiments, hardware and software + ai solutions in robotics

* Why it is important?
* Planning, manipulation, navigation, locomotion, self-driving, UAVs, specialized robots, robot design
* Boston Dynamics, Unitree, DJI, Figure One, HMND, Physical Intelligence, Yandex Robotics, Sber Robotics
* Rovers, robotic arms, quadrupeds, antropomorphic, drones, industrial robots
* Just one slide: Cameras, lidars, infrared, ultrasonic, IMU, motor encoders/proprio, GPS, tactile
* Just one slide: ROS2, arduino.h, micropython, many simulation packages, many control/panning packages, lerobot

### Seminar

Introduction into robot math and mechanics. Kinematics

* Notions of robot configuration, joint, phase and state spaces, degrees of freedom
* Joints, joint types, links, Kinematic chain/tree and related concepts — endeffector space, root frame, endeffector frame
* 6D rigid body transformations, matrix form, conversion between frames of reference, Euclidean semigroup, Lie algebra, exponential map, geodesics, homogeneous coordinates, Euler angles, Gimbal lock, quaternions, screw motion, Jacobian
* Forward kinematics, Inverse kinematics, solvers
* Trajectories of robot kinematic (idealised and without collision) movement
* URDF 1

### Homework

* theory problems with computations for pendulum, double pendulum, robot arm with revolutionary joints in 3D of forward kinematics, phase spaces and trajectories, inverse kinematics solutions and approximations, maybe some analytical path planning in idealised (non-dynamic) setup
* all math problems are accompanied by code problems which require to visualize and describe solutions

## week 2: Math 2, Simulation 1

### Lecture

Robot math and mechanics. Dynamics

* Contact, friction and potential forces, torque
* Rigid body dynamics
* Physical interactions and constraints
* Equations of motion for multi-body systems (Newton-Euler, Lagrangian formulation)
* Numerical approximation and simulation (ODE solvers, time-stepping methods)

### Seminar

Start the work on own simulator for 6 DoF robotic arm (the hard part):

* Handling motors and actuators
* Detecting collisions (BVH, collision between primitives)
* Handling friction (pyramidal approximation to friction cone)

### Homework

Finish the work on own simulator (easier part):

* collisions and resulting forces/movement
* overall `step` function with Euler integrator

## week 3: Math 3, Control 1

### Lecture

Classical control and motion planning

* closed-loop control problem statement and examples
* PID, MPC
* differential inverse kinematics control, inverse dynamics control
* different control modes of a robotic arm (ee, delta, joint, velocity)
* optimization-based controls
* state-based ML/RL methods

### Seminar

Implementing controller within own simulator

* implement joint and ee controllers
* within obstacle-free environment plan a motion for pick-place task

### Homework

Implement trajectory-level control within own simulator with obstacles for pick-place

* graph based trajectory search
* optimization based trajectory search
* execute trajectory using seminar controller and then improve by implementing veocity-level controller for smooth and close-to-source trajectory following
* dynamic scene (?)

## week 4: Mobile robots

### Lecture

(assume most of it is taught in SDC course)

* Dynamical models of cars and rovers: Dubin's, bycicle, tricycle, differential drive, Ackermann steering
* Controlled differential equations, integrators (as in smoothing control variables), solutions
* Line follower, PID tuning
* Localization, odometry-based localization, Particle swarm localization, Kalman filter
* SLAM 1 — bayesian inference from sensor and odometry data
* SDC
* Legged mobile robots (intro)
* Mobile manipulators (intro)
* Multiagent mobile setup

### Seminar

Implement cartography and task-level planning+navigation based on it

* Navigation and planning overview
* graph based path planning
* graph based scene representation
* implement simplest  stay-in-lane for autonomous bus
* plan and execute a route of autonomous bus, that collects citizens on demand

### Homework

* Line follower for warehouse
* plan and execute a route for warehouse inspection robot with dynamic updates, line following and branching

## week 5: Perception

### Lecture

Sensors and perception

* Cameras and image processing
* Depth sensors, lidars and 3D data processing
* Beacons, gps and spatial data processing
* IMU, Odometry
* Tactile sensing
* Structure from motion, visual odometry
* SLAM 2 — camera calibration, ARUCO markers, stereo vision, metric depth, reprojection

### Seminar

Combine all sensing for localization and mapping

* 2D bayesian slam in sim for Dubin's car/differential drive based on moisy 2D lidar and beacons just like your robot vacuum would do

### Homework

Ingeniuity-style localization and movement registration based on odometry and bottom-camera image visuals in simulator

## week 6: Simulation 2, Sim-to-Real, Learning from Interaction

### Lecture

Simulators in depth

* Robot/scene description formats: URDF (recap), MJCF, USD, SDF
* Simulation platforms and their trade-offs: Gazebo (ROS integration), PyBullet (lightweight), MuJoCo / MJX / Playground (fast contact dynamics, JAX-accelerated), Isaac Sim / Isaac Lab (GPU-parallel, photorealistic), SAPIEN (articulated objects), Genesis (multi-physics)
* Physics engines under the hood: PhysX 5, Bullet, MuJoCo engine — contact models, soft body, cloth, fluids
* Rendering for robotics: rasterization vs ray-tracing, synthetic data generation

Sim-to-Real transfer

* The reality gap: visual, dynamics, action discrepancies
* Domain Randomization (visual, dynamics, action noise) — OpenAI Rubik's cube, VIRAL (humanoid loco-manipulation with large-scale visual DR)
* Domain Adaptation: paired/unpaired sim-real alignment, CycleGAN for visual transfer
* System Identification: fitting simulator parameters to real-world data
* Teacher-Student architectures for privileged-to-deployable policy distillation
* Digital Twins: concept, calibration, continuous synchronization

Learning from interaction

* RL problem formulation for robotics: reward design, sparse vs dense rewards, reward shaping pitfalls
* Imitation Learning recap: Behavioral Cloning limitations (covariate shift)
* DAgger: dataset aggregation, interactive expert, convergence guarantees
* HG-DAgger, CR-DAgger (compliant residual corrections for contact-rich tasks)
* Human-in-the-Loop RL: combining demonstrations, corrections, and autonomous exploration
* When to use IL vs RL vs hybrid — practical guidelines

### Seminar

MuJoCo environment setup and sim-to-real pipeline

* Load a robotic arm in MuJoCo from MJCF, explore simulation parameters
* Implement visual and dynamics domain randomization pipeline
* Train a reaching policy with PPO, evaluate under randomization

### Homework

* Train a pick-and-place RL policy in MuJoCo with domain randomization
* Implement and compare BC vs DAgger on the same manipulation task using expert demonstrations
* Ablation: measure performance degradation when removing DR components (visual, dynamics, action noise)

## week 7: Legs (19.03)

### Lecture

Classical locomotion and stability, path planning

* legged robots equations of motion
* static and dynamic equillibrium
* movement, walking
* RL/ML solutions

### Seminar

* prepare MoCap training data for walking
* Make humanoid walk and dance in a simulator via BC/RL

### Homework

Quadruped motion

* move the robot dog from A to B with classical heuristic solutions in shortest time (mini competition, homework is graded by thresholds)
* train a model for random terrain with stairs and obstacles (mini competition 2)

## week 8: Software 1

### Lecture

ROS2 and other robotic software

* ROS Architecture, topics, pub-sub, rosmaster, rosaction, rosbag
* ros_control, ros_perception, rviz
* TF, nav_2, MoveIt, OMPL
* LeRobot

### Seminar

Install ROS via Docker and have fun with turtle-bot in Gazebo (?)

### Homework

Solve week4 seminar with ROS packages (gmapping, slam_toolbox, cartographer) ((here we may need the physical robot))

## ==PROJECTS ARE INTRODUCED==

## week 9: Robotics Data and Benchmarks

### Lecture

* Open X Embodiment, DROID, RT-1, BridgeData V2, and Robo-DM, Lerobot dataset
* Datasets formats (rlds, lerobot, mkv)
* Data Adaptation and enhancemet: RoboMimic, UMI...
* Physics/grounding benchmarks for VLMs
* Embodied Question Answering
* LIBERO/++, VLABench, CALVIN, RLBench, Robosuite, Habitat, RoboCasa, Simpler, IN-ACT, VL-Think, RoboArena

### Seminar

* Trajectory collection via teleoperation with VR and SO101 arms
* exploration of existing datasets \[lerobot, bridge\]

### Homework

* Implement spatially consistent data augmentations and pre-post-processors
* Implement data filtering for open-source datasets

## week 10: Modern Manipulation 1

### Lecture

Grasping, affordances and visuomotor policies

* Affordance models: what, where, and how to grasp — heatmaps, keypoints, contact points
* Grasping pipelines: GraspNet-1Billion, AnyGrasp (93% SR on 300+ unseen objects), Contact-GraspNet
* Dexterous grasping: AnyDexGrasp — cross-hand generalization with minimal data
* Diffusion Policy: denoising action sequences, action chunking, training stability, multi-modality handling
* 3D Diffusion Policy (DP3): compact point-cloud representations, 24% improvement over 2D baselines
* Flow-matching alternatives: FlowPolicy (7x faster single-step inference), 3D Flow Diffusion Policy
* ACT (Action Chunking with Transformers): architecture, CVAE formulation, temporal ensembling
* Practical comparison: when to use diffusion vs ACT vs flow-matching

### Seminar

* Point-cloud observation grasping with AnyGrasp-style pipeline
* Diffusion policy training on a simple tabletop task (push, pick-place)

### Homework

* Image-observation grasping pipeline: detection + grasp pose estimation + execution
* Train and evaluate diffusion policy vs ACT on a contact-rich manipulation task in simulation

## week 11: Modern 3D Vision for Robotics

### Lecture

Neural 3D representations and their applications in robotics

* 3D representations recap: point clouds, meshes, voxels, occupancy grids, signed distance fields
* Neural Radiance Fields (NeRF): volumetric rendering, positional encoding, training from posed images, Instant-NGP
* 3D Gaussian Splatting (3DGS): explicit Gaussian primitives, differentiable rasterization, real-time rendering, advantages over NeRF (speed, editability)
* Foundation models for 3D reconstruction: DUSt3R (dense unconstrained stereo 3D reconstruction), MASt3R (matching + stereo + reconstruction), MASt3R-SfM — end-to-end SfM without classical feature matching
* Metric depth estimation: Depth Anything V2, Metric3D, UniDepth — monocular metric depth for robotics
* Neural SLAM: DROID-Splat (end-to-end tracking + 3DGS rendering), SplaTAM (dense RGB SLAM with 3DGS), DenseSplat (NeRF priors for sparse views)
* 3D scene understanding: open-vocabulary 3D segmentation (LERF, OpenScene), scene graphs for robotics (ConceptGraphs)
* Applications: novel view synthesis for data augmentation, real-time scene reconstruction for manipulation, semantic 3D maps for navigation and planning

### Seminar

3D reconstruction pipelines for robotics

* Photogrammetry with MASt3R: reconstruct a manipulation workspace from a handful of RGB images
* Compare with classical SfM (COLMAP) on reconstruction quality and robustness
* Visualize and query semantic features in reconstructed scenes

### Homework

* Build an end-to-end 3D mapping pipeline: RGB images → MASt3R reconstruction → semantic feature lifting (CLIP/DINOv2) → queryable 3D scene representation
* Evaluate: reconstruction quality, semantic query accuracy, robustness to viewpoint sparsity

## week 12: VLA, Modern manipulation 2

### Lecture

Overview of all current SOTAs, training methods and data

* VLA models: VIMA, RT-1, RT-2, PerAct, Palm-E, RT-1X, Octo, OpenVLA, CogACT, Openvla-OFT, Spatial-VLA, pi0/0.5/0.6, Gemini-Robotics/ER, SmolVLA, Trace-VLA, V-JEPA2, GR00T-N1 (humanoid foundation model), Magma, RoboBrain, MotionGPT, EO-1 (interleaved vision-text-action), HAMSTER (hierarchical off-domain pretraining)
* VLA Architectures: VLM+Action head, VLM+Action queries, system1-system2, WorldModelVLA, MOE
* Robotics pretraining methods (openvla, palmE, pi0.5)
* Action decoders: diffusion policy head, autoregressive, token
* Action Tokenization: pi0-FAST
* Reasoning VLA: EcoT-VLA, OneTwoVLA,  pi0.5, Gemini-Robotics
* VLA + RL: pi\*0.6 (Recap: RL with Experience & Corrections via Advantage-conditioned Policies), PRIME-RL, rl-inf, rl4vla, RL100
* Robotic Startups/Companies Insights: FigureAI, 1X, PhysicalIntelligence, BostonDynamics, Sunday
* Challenges and future directions: generalization in action space, modality fusion, cross-embodiment, memory, datasets

### Seminar

Make your own VLA model (VLA-Adapter like)

* Evaluation: Simpler, LIBERO++
* Base VLM: Qwen2.5-VL-0.5B, SmolVLM,
* Add some Robotic Pretraining
* Try Different action decoding methods: action detokenization, action-chunking, diffusion head, flow-matching expert

### Homework

* Finish your own VLA model and evaluate it on Simpler/LIBERO tasks
* Extra: Add RL fine-tuning

## week 13: Modern Planning

### Lecture

From classical task planning to foundation-model-based reasoning

* Classical task planning: STRIPS, PDDL — symbolic state, operators, goal specification
* Task and Motion Planning (TAMP): the interface between discrete task planning and continuous motion planning, PDDLStream, bi-level planning with samplers
* LLM/VLM as planners — grounding language in robot capabilities:
  * SayCan: affordance-weighted LLM planning via value functions
  * SayPlan: 3D scene graphs for scalable multi-room planning
  * Inner Monologue: closed-loop replanning with environment feedback
* Code as Policies (CaP): LLMs generating executable robot programs, ProgPrompt
* Spatial reasoning for manipulation:
  * VoxPoser: LLM-generated 3D value maps composed with VLM grounding for zero-shot 6-DoF planning
  * ReKep: relational keypoint constraints as Python cost functions, DINOv2 + GPT-4o for automatic constraint generation
* Hierarchical and long-horizon planning: ReCAP (recursive context-aware planning), plan decomposition, subgoal chaining
* Open challenges: hallucination and grounding errors, real-time replanning latency, safety constraints, combining learned and symbolic planning

### Seminar

* Implement an LLM-based task planner: given natural language instruction, decompose into a sequence of primitive skills, execute in simulation
* Compare: PDDL planner vs LLM planner on structured multi-step tabletop rearrangement tasks — success rate, plan quality, failure modes

### Homework

* Build a VoxPoser-inspired system: LLM generates spatial cost maps from language instructions, optimization-based motion planner executes 6-DoF trajectories in simulation
* Evaluate on 5+ diverse manipulation tasks (pick-place, pour, stack, open drawer, wipe surface) — measure success rate and generalization to novel object arrangements

## week 14: World models, generative models for robotics, latent actions (07.05)

### Lecture

* World Models: the concept, components, limitations.
* Training inside Imagination: Dreamer (1,2,3,4)
* Latent Action Pretraining from Videos: Genie, LAPA, what actions do we actually learn
* World understanding and physics inconsistency problem in video generation?
* Video Generation for Robotics: GR-2, MagicDriveDiT, DriveDreamer4D, DrivingDiffusion, Cosmos, COMBO

### Seminar

* Launch the pre-trained Dreamer and show it component by component and in general how it works. Fine-tune in live?

### Homework

* To implement and train Genie (using template) for Flappy Bird.

## week 15: Multi-Agent Systems, Collaborative Robotics

### Lecture

Multi-agent coordination and human-robot collaboration

* Multi-Agent Path Finding (MAPF): problem formulation, CBS (Conflict-Based Search), ECBS, priority-based planning, LaCAM
* Scalable MAPF: PRISM (decentralized, 3.4x more agents than centralized), lifelong MAPF for warehouse settings
* Learning-based MAPF: MAPF-GPT (autoregressive transformer planner), MAPF-World (action world models with zero-shot generalization)
* Multi-Agent RL (MARL): centralized training decentralized execution (CTDE), QMIX, MAPPO, communication learning
* Warehouse and logistics robotics: fleet management, task allocation, deadlock resolution, throughput optimization
* Human-Robot Interaction (HRI):
  * Safety standards: ISO 10218, ISO/TS 15066, speed-and-separation monitoring, power-and-force limiting
  * Shared autonomy: adjustable autonomy, human intent prediction, assistive teleoperation
  * Trust calibration and personalized safety models
* Collaborative manipulation: dual-arm coordination, multi-robot assembly, handover protocols
* NeurIPS MARS challenge: multi-agent embodied AI with VLMs for planning and coordination

### Seminar

* Implement a MAPF solver (CBS or priority-based) for a warehouse grid scenario
* Multi-robot task allocation: assign pick-up and delivery tasks to a fleet, optimize makespan
* Visualize agent coordination, detect and resolve deadlocks

### Homework

* Multi-agent warehouse optimization: implement and evaluate MAPF + task allocation pipeline on a realistic warehouse map with dynamic task arrivals (mini-competition: minimize total completion time)
* Implement cooperative dual-arm manipulation: two arms coordinate to perform a task that a single arm cannot (e.g., folding, lifting a large object) in simulation

## ==PROJECT DEFENSE==


# Projects

# TODO

* multi agent path planning, mapfgpt
* more on motors and robot engineering

# Timeline

| week | date | topic | lecturer | TA |
|----|----|----|----|----|
| 1 | 12.02 | Intro and Math 1 | Vlad Shakhuro (YSDA, AIRI) | Andrei Spiridonov (YSDA, AIRI) |
| 2 | 19.02 | Math 2, Simulation 1 | Andrei Spiridonov (YSDA, AIRI) | Konstantin Soshin (AIRI) |
| 3 | 26.02 | Math 3, Control 1 | Andrei Spiridonov (YSDA, AIRI) |    |
| 4 | 05.03 | Mobile robots | Andrei Spiridonov (YSDA, AIRI) |    |
| 5 | 12.03 | Perception | Vlad Shakhuro (YSDA, AIRI) | Andrei Spiridonov (YSDA, AIRI) |
| 6 | 19.03 | Simulation 2, Sim-to-Real, Learning from Interaction | Andrei Spiridonov (YSDA, AIRI) |    |
| 7 | 26.03 | Legs | Egor Maslennikov (YSDA, Sber.Robotics) |    |
| 8 | 02.04 | Software 1 | Andrei Spiridonov (YSDA, AIRI) |    |
| 9 | 09.04 | Robotics Data and Benchmarks | Sergei Davidenko (Sber Robotics Lab) |    |
| 10 | 16.04 | Modern Manipulation 1 | Andrey Moskalenko (YSDA, AIRI) |    |
| 11 | 23.04 | Modern 3D Vision for Robotics | Andrei Spiridonov (YSDA, AIRI) |    |
| 12 | 30.04 | VLA, Modern manipulation 2 | Nikita Kachaev (AIRI) |    |
| 13 | 07.05 | Modern Planning | Andrei Spiridonov (YSDA, AIRI) |    |
| 14 | 14.05 | World models, generative models for robotics, latent actions | Danil Tokhchukov (YSDA, Skoltech) |    |
| 15 | 21.05 | Multi-Agent Systems, Collaborative Robotics | Yaroslav Khripkov (YSDA, Avito) |    |

# Literature

## Robotics

* Kevin Lynch and Frank Park, [Park Modern robotics: Mechanics, planning, and control](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf). ❗️
* Russ Tedrake, [Robotic Manipulation](https://manipulation.csail.mit.edu/index.html) ❗️

## Rigid Body Dynamics

* Roy Featherstone, [Rigid Body Dynamics Algorithms](https://gaoyichao.com/Xiaotu/papers/2008%20-%20Rigid%20body%20dynamics%20algorithms.pdf) ❗️
* Baraff/Witkin ([SIGGRAPH course notes](https://web.mat.upc.edu/toni.susin/files/BaraffWitkinKass2001.pdf)) ❗️
* Brian Brian, [Impulse-based Dynamic Simulation of Rigid Body Systems (PhD Thesis)](https://people.eecs.berkeley.edu/\~jfc/mirtich/thesis/mirtichThesis.pdf)

## Contact and Friction

* Le Lidec et al, [Contact Models in Robotics: a Comparative Analysis](https://arxiv.org/abs/2304.06372v2)

## Collisions

* Christer Ericson, [Real-Time Collision Detection](http://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf) ❗️
* Gilbert et al, [A Fast Procedure for Computing the Distance Between Complex Objects in Three-Dimensional Space](https://graphics.stanford.edu/courses/cs448b-00-winter/papers/gilbert.pdf)
* Van den Bergen, [Proximity Queries and Penetration Depth Computation on 3D Game Objects](https://graphics.stanford.edu/courses/cs468-01-fall/Papers/van-den-bergen.pdf)
* Pan et al, [FCL: A General Purpose Library for Collision and Proximity Queries](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf)

## Other

* Yoon et al, [Comparative Study of Physics Engines for Robot Simulation with Mechanical Interaction](https://www.mdpi.com/2076-3417/13/2/680)

# Tutorials

* <https://winter.dev/articles/physics-engine>
* <https://winter.dev/articles/gjk-algorithm>