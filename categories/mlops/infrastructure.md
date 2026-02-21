# Section 10: Infrastructure and Tooling for MLOps

This section examines the fundamental facilities and systems required to support the sustainable development and maintenance of machine learning systems. It categorizes infrastructure into four distinct layers: **Storage and Compute**, **Resource Management**, **ML Platform**, and **Development Environment**.

## I. The Infrastructure Hierarchy
Machine learning infrastructure is best understood as a set of layers, each building upon the other to support the end-to-end ML lifecycle.

> "In the ML world, infrastructure is the set of fundamental facilities that support the development and maintenance of ML systems... we will examine the following four layers: Storage and compute... Resource management... ML platform... Development environment."

### **The Scale of Infrastructure Needs**
The investment required for infrastructure depends heavily on the production scale and the complexity of the applications.

> "At one end of the spectrum, you have companies that use ML for ad hoc business analytics... These companies probably won’t need to invest in any infrastructure... In the middle of the spectrum are the majority of companies, those who use ML for multiple common applications... at reasonable scale."

---

## II. Storage and Compute Layer
This foundational layer provides the essential resources needed for any ML project. While storage has become largely commoditized, the compute layer remains a complex area of optimization.

### **Compute Metrics and Units**
Compute units are generally characterized by their memory capacity and their operational speed, often measured in FLOPS.

> "A compute unit is mainly characterized by two metrics: how much memory it has and how fast it runs an operation... The most common metric is FLOPS—floating point operations per second."

### **Public Cloud vs. Private Data Centers**
While many companies start on the public cloud due to its elasticity, the long-term costs often lead to "cloud repatriation"—moving workloads back to private data centers to improve margins.

> "Sarah Wang and Martin Casado... estimated that 'across 50 of the top public software companies currently utilizing cloud infrastructure, an estimated $100B of market value is being lost among them due to cloud impact on margins—relative to running the infrastructure themselves.'"

---

## III. Development Environment
The "dev environment" is where ML engineers spend most of their time writing code, running experiments, and interacting with production systems.

### **Integrated Development Environments (IDEs) and Notebooks**
Notebooks are particularly favored for their statefulness, allowing engineers to retain data in memory between runs.

> "Notebooks have a nice property: they are stateful—they can retain states after runs. If your program fails halfway through, you can rerun from the failed step instead of having to run the program from the beginning."

### **Standardization and Containers**
Standardizing the dev environment ensures that code runs consistently across different machines. Containers (like Docker) are the primary technology used to move code from development to production.

> "With Docker, you create a Dockerfile with step-by-step instructions on how to re-create an environment in which your model can run... These instructions allow hardware anywhere to run your code."

---

## IV. Resource Management
This layer focuses on scheduling and orchestrating workloads to maximize resource utilization and cost-effectiveness.

### **Schedulers and Orchestrators**
Schedulers manage the "when" of a task (handling dependencies), while orchestrators manage the "where" (provisioning machines).

> "Schedulers are cron programs that can handle dependencies... orchestrators are concerned with where to get those resources... Schedulers deal with job-type abstractions such as DAGs... Orchestrators deal with lower-level abstractions like machines, instances, clusters."

### **Workflow Management Tools**
Tools like Airflow, Argo, and Metaflow help manage the complex Directed Acyclic Graphs (DAGs) inherent in ML pipelines.

> "Workflow management tools manage workflows. They generally allow you to specify your workflows as DAGs... Each step in a workflow is called a task."

---

## V. The ML Platform
An ML platform provides a shared set of tools across multiple business use cases to manage the deployment, storage, and feature consistency of models.

### **Model Store**
A model store must track significantly more than just model binaries; it must capture the context of the model's creation for debugging and lineage.

> "To help with debugging and maintenance, it’s important to track as much information associated with a model as possible. Here are eight types of artifacts that you might want to store... Model definition... Model parameters... Featurize and predict functions... Dependencies... Data... Model generation code... Experiment artifacts... Tags."

### **Feature Store**
Feature stores solve the three core problems of feature management, transformation, and consistency.

> "At its core, there are three main problems that a feature store can help address: feature management, feature transformation, and feature consistency."

---

## VI. Build Versus Buy Decisions
Deciding whether to build infrastructure in-house or buy a vendor solution is a critical strategic choice for engineering leadership.

> "If it’s something we want to be really good at, we’ll manage that in-house. If not, we’ll use a vendor."

> "Building means that you’ll have to bring on more engineers to build and maintain your own infrastructure. It can also come with future cost: the cost of innovation. In-house, custom infrastructure makes it hard to adopt new technologies available because of the integration issues."
