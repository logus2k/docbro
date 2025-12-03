# **Glossary of ROS 2 Terms**

This glossary covers essential terminology related to ROS 2’s communication stack, DDS, RMW implementations, discovery, QoS, and middleware behavior.

---

## **ROS 2 Abstractions**

### **ROS Node**

A process containing publishers, subscribers, services, actions, and other ROS interfaces. Nodes communicate through the middleware, not through direct sockets.

### **RCL (ROS Client Library)**

The language-specific API used by developers:

* `rclcpp` (C++)
* `rclpy` (Python)

All DDS details are hidden behind RCL → RMW.

---

## **Middleware Abstraction Layer**

### **RMW – ROS Middleware Interface**

A thin abstraction layer that bridges RCL with the actual middleware implementation. It defines a uniform API so that ROS 2 can switch middleware without changing user code.

### **RMW Implementation**

A plugin implementing the RMW API for a specific middleware. Examples:

* `rmw_fastrtps_cpp` – Fast DDS
* `rmw_cyclonedds_cpp` – Eclipse Cyclone DDS
* `rmw_connextdds` – RTI Connext
* `rmw_gurumdds_cpp` – GurumDDS (commercial)

Selected via environment variable:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

---

## **DDS (Data Distribution Service)**

### **DDS**

A standardized publish–subscribe middleware for distributed systems, defined by the Object Management Group (OMG). ROS 2 uses DDS because it supports real-time behavior, QoS policies, and scalable discovery.

DDS is responsible for:

* endpoint discovery
* matching publishers to subscribers
* reliability (if enabled)
* deadline & latency constraints
* durability (persistent data)
* message serialization
* transport (UDP, shared memory)

### **DDS Vendor / DDS Implementation**

A concrete implementation of the DDS standard. Each has its own performance characteristics, features, and behaviors:

* **Fast DDS** (default in ROS 2)
* **Cyclone DDS** (lightweight, reliable)
* **RTI Connext DDS** (commercial)
* **GurumDDS** (commercial)

---

## **RTPS and Transport**

### **RTPS – Real-Time Publish-Subscribe Protocol (DDSI-RTPS)**

The standardized *wire protocol* used by many DDS implementations. RTPS defines:

* packet formats
* discovery announcements
* how participants, readers, and writers communicate

RTPS runs over:

* UDPv4 / UDPv6 (most common)
* shared memory (for intra-host)

### **Transport Layer**

Where messages physically travel:

* UDP multicast
* UDP unicast
* Shared memory
* Loopback interface
  Transport behavior strongly affects discovery reliability.

---

## **DDS Entities & Discovery**

### **Participant**

A DDS process (ROS node infrastructure) that joins a domain and participates in discovery.

### **Publisher / Writer**

DDS "Writer" corresponds to a ROS publisher. Responsible for sending samples.

### **Subscriber / Reader**

DDS "Reader" corresponds to a ROS subscriber. Responsible for receiving samples.

### **Topic**

The *data channel* identified by name + type + QoS. A matching publisher and subscriber must agree on all of these.

### **Discovery**

The mechanism by which DDS participants learn about other participants, their topics, QoS, and endpoints.

Types of discovery:

* **Participant discovery** (who exists)
* **Endpoint discovery** (what topics exist)

Fast-DDS and CycloneDDS differ significantly in discovery performance.

---

## **QoS (Quality of Service)**

DDS QoS policies define communication behavior. ROS 2 exposes these through QoS profiles.

Key QoS terms:

### **Reliability**

* **Best effort** – no guarantees
* **Reliable** – ensures delivery

### **Durability**

Controls whether late-joining subscribers get past messages.

### **Deadline**

Maximum interval between expected messages.

### **Liveliness**

How publishers announce that they are still alive.

### **History**

How many past messages a reader/writer stores.

QoS mismatches can cause nodes to fail to connect even if names match.

---

## **Domains**

### **ROS Domain / DDS Domain**

A communication isolation mechanism. Only nodes in the same domain can see each other.

Selected via:

```bash
export ROS_DOMAIN_ID=<number>
```

Useful for multi-robot or multi-process separation.

---

## **Fast DDS (formerly Fast RTPS)**

### **Fast DDS**

Open-source DDS implementation by eProsima. Default for ROS 2 distros such as Jazzy.

Pros:

* Feature-rich
* Good QoS support
* Widely used in ROS

Cons:

* Shared memory and multicast can fail on WSL2, VPNs, Docker, or restrictive networks
* Sometimes more complex to configure

### **Fast DDS Shared Memory Transport**

High-performance local communication using `/dev/shm`.
Broken in some virtualized environments (WSL2, containers).

---

## **Cyclone DDS**

### **Cyclone DDS**

Open-source DDS implementation by Eclipse Foundation. Known for:

* Simple, stable behavior
* Strong compatibility with restricted or virtualized environments
* Good performance in practice
* Light footprint

Often works “out of the box” when Fast-DDS fails.

---

## **ROS 2 Daemon**

### **`ros2 daemon`**

A background process used by ROS CLI tools to speed up discovery queries.

Commands:

* `ros2 daemon start`
* `ros2 daemon stop`
* `ros2 daemon status`

A frozen daemon is usually caused by middleware initialization failures (very common with Fast-DDS on WSL2).

---

## **RMW Selection Workflow**

To switch middleware:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Check:

```bash
ros2 doctor --report
```

---

## **When to Change DDS / RMW**

### Use Fast DDS when:

* Native Linux
* Shared memory is needed
* Complex QoS tuning
* Multicast works reliably

### Use Cyclone DDS when:

* WSL2
* Docker
* VPN / corporate / unusual networks
* Discovery hangs
* CLI commands freeze
* Lightweight footprint is desirable

---
