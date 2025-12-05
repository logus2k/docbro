# **ROS 2 Communication Glossary**

A structured glossary of terms relevant to ROS 2 communications, DDS middleware, RMW implementations, discovery, QoS, and node structure.

---

## **Core ROS Terminology**

### **ROS — Robot Operating System**

A framework for building modular, distributed robotic systems.
ROS 2 provides:

* a message-passing communication system
* tools for debugging, logging, and visualization
* a component lifecycle
* abstraction over different middleware implementations

ROS itself is *not* an operating system; it sits on top of Linux/macOS/Windows.

---

### **ROS 2**

The second generation of ROS — redesigned with:

* a DDS-based communication layer
* real-time capabilities
* support for multi-platform, multi-robot, and high-performance systems
* modular middleware via the RMW abstraction layer

---

### **ROS Node**

A running process participating in the ROS graph, containing:

* publishers
* subscribers
* services
* action servers/clients
* parameters

Nodes never communicate directly; everything flows through the middleware.

---

### **ROS Graph**

The conceptual network of:

* nodes
* topics
* services
* actions

Introspection tools (e.g., `ros2 node list`, `ros2 topic list`) query this graph via the underlying DDS discovery.

---

### **Message**

A structured data type published and subscribed over topics. Defined in `.msg` files and translated to language-specific classes.

---

### **Service**

A synchronous request–reply communication pattern in ROS 2. Implemented over DDS Request/Reply.

---

### **Action**

An asynchronous long-running goal interface built on top of services + topics. Used for tasks like navigation, manipulation, etc.

---

## **ROS Client Libraries**

### **RCL – ROS Client Library**

The language-agnostic ROS interface layer.
`rclcpp` and `rclpy` wrap this API.

Responsibilities:

* creating nodes
* configuring QoS profiles
* interacting with RMW
* managing message lifecycles

---

### **rclcpp**

C++ client library providing high-performance ROS 2 interfaces.

### **rclpy**

Python client library for ROS 2, wrapping RCL in Pythonic APIs.

---

### **Executor**

The component responsible for:

* spinning callbacks
* scheduling execution
* managing callback queues

Executors tie together node activity and underlying DDS events.

---

### **Callback Group**

A mechanism to control concurrency of callbacks. Useful for threading, composition, and deterministic execution.

---

## **Middleware Abstraction Layer**

### **RMW — ROS Middleware Interface**

A stable C API that sits between RCL and the underlying DDS.
Purpose:

* isolate ROS from specific DDS vendors
* allow switching middleware at runtime
* enable vendor-specific enhancements under a unified interface

---

### **RMW Implementation**

A specific backend plugin that implements the RMW API for a given DDS library.

Examples:

* `rmw_fastrtps_cpp` — Fast DDS
* `rmw_cyclonedds_cpp` — Cyclone DDS
* `rmw_connextdds` — RTI Connext
* `rmw_gurumdds_cpp` — GurumDDS

Selected via:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

---

## **DDS and RTPS Concepts**

### **DDS — Data Distribution Service**

A standardized publish–subscribe middleware used as ROS 2’s transport layer.
DDS provides:

* discovery
* serialization
* reliability
* liveliness tracking
* QoS policy enforcement
* real-time communication
* scalable distributed messaging

Defined by the OMG (Object Management Group).

---

### **DDS Vendor / Implementation**

A concrete implementation of the DDS specification.

Common ROS-compatible implementations:

* **Fast DDS** (eProsima) – default in ROS 2
* **Cyclone DDS** (Eclipse Foundation)
* **RTI Connext DDS** (commercial)
* **GurumDDS** (commercial)

Each differs in performance, discovery stability, configurability, and platform compatibility.

---

### **RTPS — Real-Time Publish-Subscribe Protocol (DDSI-RTPS)**

The standardized wire protocol that DDS vendors use to communicate.
Defines:

* network packet structure
* discovery announcements
* DATA / HEARTBEAT messages
* matching rules for endpoints

Runs over:

* UDP multicast (primary)
* UDP unicast
* shared memory (intra-host)

---

### **Domain / DDS Domain / ROS Domain**

A communication namespace.
Only nodes sharing the same domain can discover each other.

Configured through:

```bash
export ROS_DOMAIN_ID=N
```

Allows multiple independent ROS systems on the same network.

---

### **Participant**

A DDS entity representing an entire process within a domain.
A ROS node creates at least one participant.

---

### **Writer / DataWriter**

DDS entity representing a publisher endpoint.
Responsible for:

* serializing outgoing data
* enforcing QoS
* sending RTPS messages

---

### **Reader / DataReader**

DDS entity representing a subscriber endpoint.
Responsible for:

* receiving data
* deserializing
* applying QoS rules
* delivering samples to RMW

---

### **Topic (DDS layer)**

Defined by:

* name
* data type
* QoS

A ROS topic name maps to a DDS topic name via ROS naming conventions and namespace rules.

---

## **Quality of Service (QoS)**

QoS influences how data is delivered.

### **Reliability**

* *best_effort* – may drop samples
* *reliable* – retry and guarantee delivery

### **Durability**

Controls sample persistence:

* *volatile* – no history
* *transient_local* – late subscribers receive last sample

### **History**

How many past samples to store:

* *keep_last*
* *keep_all*

### **Deadline**

Maximum allowed time between samples. Missed deadlines trigger events.

### **Liveliness**

Ensures publishers signal that they are still alive.

### **Lifespan**

How long a message remains valid before being discarded.

QoS mismatches prevent publishers and subscribers from matching.

---

## **ROS 2 CLI Support**

### **`ros2 daemon`**

A background service used by CLI tools (`ros2 topic`, `ros2 node`, etc.) to accelerate graph queries.

Commands:

* `ros2 daemon start`
* `ros2 daemon stop`
* `ros2 daemon status`

A frozen daemon usually indicates middleware issues (e.g., Fast DDS failing under WSL2).

---

## **Middleware Implementations in Detail**

### **Fast DDS (formerly Fast RTPS)**

Default DDS implementation in many ROS 2 distros.

Pros:

* full-featured
* robust QoS
* good performance

Cons:

* discovery instability on WSL2
* shared memory transport often fails in virtualized environments

---

### **Cyclone DDS**

Lightweight, stable, and often more reliable in constrained environments.

Pros:

* excellent stability on WSL2/Docker
* simple discovery model
* often “just works”

Cons:

* fewer advanced configuration knobs than Fast DDS

---

### **RTI Connext DDS**

Commercial, high-performance DDS with strong real-time capabilities.
Less commonly used in open-source ROS deployments due to licensing constraints.

---

### **GurumDDS**

Another commercial DDS implementation supported by ROS 2.

---

# **RMW Selection Workflow**

To switch middleware:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

To verify:

```bash
ros2 doctor --report
```

---
