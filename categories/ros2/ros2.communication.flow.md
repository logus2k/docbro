# **ROS 2 Communication Flow: Talker → Listener**

This document explains and visualizes how a message travels from a **publisher node (talker)** to a **subscriber node (listener)** in ROS 2. It breaks down each layer involved—from the user-facing ROS APIs all the way down to DDS, RTPS, and the OS network transport.

The goal is to clarify the roles of:

* **RCL** (ROS Client Libraries: rclcpp / rclpy)
* **RMW** (ROS Middleware Interface)
* **DDS** (Data Distribution Service implementations like Fast DDS, Cyclone DDS)
* **RTPS** (Real-Time Publish-Subscribe protocol)
* **Transport** (UDP, multicast, shared memory)

This layered architecture is what makes ROS 2 middleware-agnostic and allows nodes to communicate transparently, regardless of which DDS implementation is selected.

---

# **End-to-End Diagram: Talker → Listener**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          TALKER  (Publisher Node)                           │
└────────────────────────────────────────────────────────────────────────────┘
                |
                |  (1) Your code calls publisher->publish(msg)
                ▼
        ┌─────────────────────────────────────┐
        │        ROS Client Library (RCL)     │
        │     rclcpp::Publisher / rclpy       │
        └─────────────────────────────────────┘
                |
                |  (2) RCL converts ROS message into RMW format
                ▼
        ┌─────────────────────────────────────┐
        │      RMW (ros middleware interface) │
        │   e.g. rmw_cyclonedds_cpp, rmw_fastrtps_cpp
        └─────────────────────────────────────┘
                |
                |  (3) RMW forwards data to DDS Writer
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    DDS Publisher (DataWriter)                               │
│  - Serializes message to DDS/IDLC format                                    │
│  - Applies QoS (reliability, history, deadlines)                            │
│  - Sends data to matching DDS Subscribers                                   │
└────────────────────────────────────────────────────────────────────────────┘
                |
                |  (4) DDS sends message using RTPS protocol
                ▼
        ┌─────────────────────────────────────┐
        │       RTPS (DDSI-RTPS) Protocol     │
        │ - discovery packets                 │
        │ - DATA / HEARTBEAT submessages      │
        └─────────────────────────────────────┘
                |
                |  (5) Transport: UDP Multicast / Unicast / Shared Memory
                ▼
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║                         Operating System Transport                        ║
  ║                      UDPv4 / UDPv6 / Shared Memory                        ║
  ║    - Network routing, loopback, WSL2 virtualization layer (if applicable) ║
  ╚══════════════════════════════════════════════════════════════════════════╝
                |
                |  (6) RTPS packets received by DDS Subscriber
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       DDS Subscriber (DataReader)                           │
│ - Deserializes DDS message                                                  │
│ - Applies QoS filtering (deadline, lifespan, history)                       │
│ - Hands sample to RMW                                                       │
└────────────────────────────────────────────────────────────────────────────┘
                |
                |  (7) RMW delivers ROS message representation to RCL
                ▼
        ┌─────────────────────────────────────┐
        │          RCL (Python/C++)           │
        │   Calls user-defined callback       │
        └─────────────────────────────────────┘
                |
                |  (8) Your callback executes
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          LISTENER (Subscriber Node)                         │
│            e.g.    void callback(const std_msgs::msg::String& msg)         │
└────────────────────────────────────────────────────────────────────────────┘
```
---
