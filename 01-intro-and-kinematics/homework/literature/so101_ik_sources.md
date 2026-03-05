# SO-100 / SO-101 inverse kinematics — related sources

Curated list of public sources on SO-100/SO-101 inverse kinematics (with short descriptions). Use these for deriving analytical IK and for implementation references.

## Official docs and repos

- **XLeRobot SO101 Guide**: https://xlerobot.readthedocs.io/en/latest/software/getting_started/SO101.html  
  EE Cartesian control and IK usage; `FollowerEndEffector` for inverse kinematics.

- **LeRobot SO-101**: https://huggingface.co/docs/lerobot/en/so101  
  Setup and teleop; kinematics via bus servo control, custom IK for pose targets.

- **LeRobot Issue #678 (SO-100 IK)**: https://github.com/huggingface/lerobot/issues/678  
  Confirms no built-in analytical IK; points to implementing your own.

## Student / academic derivations

- **Georgia Tech ECE4560 Assignment 8 (SO-101 IK)**: https://maegant.github.io/ECE4560/assignment8-so101/  
  Geometric IK implementation guide; full 6-DOF analytic solution from DH (students code it). **Best starting point for formulas.**

- **Georgia Tech ECE4560 Assignment 6 (SO-101 FK)**: https://maegant.github.io/ECE4560/assignment6-so101/  
  DH table and link lengths for FK; needed to invert to IK.

## Other technical sources

- **LeRobot Issue #1462**: https://github.com/huggingface/lerobot/issues/1462  
  SO101 pose / IK accuracy on real arm.

- **LeRobot SO101 FollowerEndEffector Issue #1966**: https://github.com/huggingface/lerobot/issues/1966  
  EE IK follower class for teleop.

- **XLeRobot Enhancement #115**: https://github.com/Vector-Wangel/XLeRobot/issues/115  
  Teleop and IK readability; code snippets.

- **Chinese blog (2D IK cosines)**: https://blog.csdn.net/gitblog_00282/article/details/154719974  
  Simplified 2D IK with measured lengths (e.g. l1=115.9 mm, l2=135 mm).

- **Seeed Studio LeRobot SO100M**: https://wiki.seeedstudio.com/lerobot_so100m_new/  
  Hardware integration; basic kinematics for calibration.

- **LinkedIn LeRobot SO-101 project**: https://www.linkedin.com/posts/sanjayacharjee_robotics-lerobot-mujoco-activity-7419880395548110848-g6-e  
  Forward and inverse kinematics code for simulation.

For analytical derivation and formulas, the Georgia Tech ECE4560 assignments (6 and 8) are the main reference.
