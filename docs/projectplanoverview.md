# Project Scope and Core Approach

This project aims to develop a **Blender + Python–based automated pipeline** that analyzes and restructures human/pedestrian assets according to **ASAM OpenMATERIAL Human object principles** in a rule-based manner.

The pipeline will automatically classify skeleton structures, programmatically construct the ASAM node hierarchy, and normalize joint/limb semantics in alignment with behavior models (locomotion, joint DOF, gait cycle). This will prepare assets for direct integration into behavior models and simulation environments.

The primary objective is to replace manual asset fixing with a **principle-based, repeatable, and scalable automation system**.

---

## Phase 1 – Research and Standards Review

In the first phase, the following standards and documentation will be studied in detail to extract minimum compliance requirements and structural principles:

- **ASAM OpenMATERIAL 3D – Human object class and node hierarchy**  
  https://asam-ev.github.io/OpenMATERIAL-3D/asamopenmaterial/latest/specification/07_geometry/object-human/human-index.html

- **OpenX ecosystem context (OpenSCENARIO, OSI integration)**  
  https://www.asam.net/standards/detail/openmaterial/  
  https://www.asam.net/standards/detail/openscenario/v200/  
  https://www.asam.net/standards/detail/osi/

The goal of this phase is to extract the required node structures and naming conventions for the human object, clarify skeleton and articulated part requirements, and identify structural assumptions necessary for behavior model integration.

---

## Phase 2 – Automated Asset Analysis with Blender + Python

After the research phase, an **asset analyzer** will be developed using Blender Python to:

- Automatically extract skeleton presence and bone hierarchy
- Detect bone lengths and kinematic chains (leg/arm chains)
- Analyze material slots and mesh–bone bindings
- Extract object and collection hierarchy

These analysis outputs will serve as inputs to the rule-based mapping layer.

---

## Phase 3 – Rule-Based Skeleton Classification and ASAM Node Auto-Builder

Using a rule-based system, bones will be automatically classified based on names, hierarchy, and orientation into:

- Pelvis / root
- Upper and lower extremities
- Joint types (knee, hip, ankle, shoulder, etc.)

Following classification, the ASAM OpenMATERIAL Human node hierarchy will be programmatically constructed within Blender, and classified bones will be automatically mapped to this structure. This will significantly reduce or eliminate the need for manual node and hierarchy construction.

---

## Phase 4 – Behavior-Aware Normalization (Locomotion, Joint DOF, Gait)

The skeleton will be normalized to be behavior-model–ready:

- Automatic DOF assignment based on joint semantic class (e.g., knee = hinge, hip = ball joint)
- Axis alignment and motion limit assignment using Blender constraints
- Detection of leg chains and extraction of left/right phase relationships for gait cycles
- Standardization of pelvis/root reference for root motion and locomotion

These parameters will enable the generation of different pedestrian types (e.g., elderly, agile, child) and allow assets to be directly connected to OpenSCENARIO/OSI-based pedestrian behavior models.

---

## Phase 5 – Pedestrian–Vehicle Interaction and Edge Case Behaviors

The project will also include test scenarios to evaluate interactions between pipeline-generated pedestrian assets and vehicles, such as:

- Road crossing behavior
- Waiting at crosswalks and yielding to vehicles
- Hesitation or retreat when vehicles approach
- Unusual or edge case behaviors (sudden running, indecision, direction changes, etc.)

These scenarios will be defined via OpenSCENARIO events that trigger pedestrian behavior models, enabling evaluation of the pipeline not only structurally but also in terms of **behavioral interaction validity**.

---

## Phase 6 – Automation and Repeatability

One of the main project deliverables will be a Blender Python–based automation toolkit consisting of:

- Asset analyzer
- Rule-based skeleton classifier
- ASAM node auto-builder
- Behavior-ready skeleton normalizer

This pipeline will enable largely automatic and repeatable transformation of human assets from diverse sources into ASAM OpenMATERIAL–compliant, behavior-ready pedestrian assets.
