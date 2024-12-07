
**The first lecture starts at 8:45 AM, Feb 17, 2025.**

- [1. Important links](#1-important-links)
- [2. Course description](#2-course-description)
- [3. Learning objectives](#3-learning-objectives)
- [4. Course team](#4-course-team)
- [5. :dart: Grading policy](#5-dart-grading-policy)
  - [5.1. Homework](#51-homework)
  - [5.2. Group projects](#52-group-projects)
- [6. Detailed schedule](#6-detailed-schedule)

This repository contains the materials of the **MSc on Distributed Deep Learning Systems** course running in Spring 2025 at UniNE.

##  1. Important links

- [Project description](project.md)
- [Lab and assigments](lab/README.md)

##  2. Course description

Machine learning systems are often conventionally designed for centralized processing in that they first collect data from distributed sources and then execute algorithms on a single server. Due to the limited scalability of processing large amount of data and the long latency delay, there is a strong demand for a paradigm shift to distributed or decentralized ML systems which execute ML algorithms on multiple and in some cases even geographically dispersed nodes.

The aim of this  course is to let students learn how to design and build distributed ML systems via paper reading, presentation, and discussion; We provide a broad overview on the design of the state-of-the-art distributed ML systems, with a strong focus on the scalability, resource efficiency, data requirements, and robustness of the solutions. We will present an array of methodologies and techniques that can efficiently scale ML analysis to a large number of distributed nodes against all operation conditions, e.g., system failures and malicious attacks. The specific course topics are listed below.

The course materials will be based on a mixture of classic and recently published papers.

##  3. Learning objectives
- To understand design principles of distributed and federated learning systems
- To analyze distributed and federated ML in terms of the scalability and accuracy-performance tradeoff
- To understand and implement horizontal and vertical federated learning systems
- To understand and implement federated learn systems on different  models, e.g., classification and generative models
- To understand and analyze vulnerabilities and threat to federated learning systems, e.g., data poison attacks and freerider attacks
- To design and implement defense strategies against adversarial clients in federated systems

##  4. Course team

This course will be mainly taught by [Prof. Lydia Y Chen](https://lydiaychen.github.io/).
TAs are Abel Malan, Aditya Shankar, and Bart Cox, who will run the lab and grade homework.

Lydia is the responsible instructor of this course and can be reached at **lydiaychen@ieee.org**.

##  5. :dart: Grading policy

This grade of this course is largely determined through three components:

1. Lab assignment (30%): 3 individual lab assignment, due in week 4, 8, 12.

2. Group project (70%): group project report (60%) and presentation (10%). The goal is to reproduce a paper and propose an algorithm to extend the paper.
   - There will be an initial proposal in week 5,
   - The interim discussion with each team in week 9.
   - The final report will be due in the last week, and 20 minutes presentation in the last week as well.

**All assessment items (homework, and projects reports) have to be submitted via ILIAS.**

###  5.1. Homework
- Homework 1: due in week 4
- Homework 2: due in week 8
- Homework 3: due in week 11

Students are given additional 48 hours grace period for late submission and will not receive any grade penalty. However, submissions after 48 hours grace period will not be considered and students will loose 15 points of their final grade.

###  5.2. Group projects

The objective is to reproduce and improve the performance of a paper from the course list. The students need to hand in a final project report in the style of a short scientific paper, stating their individual contribution to the overall system performance. There are four milestones associated with this project. See the [project description](project.md) for more information.

- Group size: 1-2 students
- Schedule: initial proposal (week 5), interim meeting (week 9), report due (week 14), and presentation/interview (week 14).



At the end of each project phase we will conduct a short interview (20 minutes per group) about the group project and its connection to the course content. Based on the project report and the interview, each member of the group receives a grade.

##  6. Detailed schedule

**Week**|**Lecture Topic**|**Lab Topic**|**Assignment Due**
:-----|:-----|:-----|:-----
Week 1 (Feb 17) | Distributed Machine Learning I | No Lab
Week 2 (Feb 24) | Memory and Acceleration Technology | Setup & Intro
Week 3 (Mar 3) | Federated Learning I (Horizontal) | FedSGD & FedAvg
Week 4 (Mar 10) | Federated Learning II (Vertical) | HW1 | HW1
Week 5 (Mar 17) | Heterogeneous and Multi-modality FL | VFL | Project proposal
Week 6 (Mar 24) | Federated Generative AI |  Generative Modelling
Week 7 (Mar 31) | Hyper-parameter Tuning | Q/A about HW2|
Week 8 (Apr 7)  |  Robust Distributed Learning  | Attacks | HW2|
Week 9 (Apr 14) | Attacks and Defenses I (online) | Defenses |  Project midterm
Week 10 (Apr 21) | *No Lecture* | *No Lab* |  |
Week 11 (Apr 28) | Attacks and Defenses II | Q/A about HW3| |
Week 12 (May 5 | Privacy Enhancing Technology for Federated Learning | | HW3|
Week 13 (May 12) | Distributed Inference 
Week 14 (May 19) | *No Lecture* | *No Lab* |
Week 15 (May 27) | Project presentation | *No Lab* | Project report
