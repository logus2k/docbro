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

```mermaid
flowchart TD
    A["TALKER<br/>(Publisher Node)"]
    B["ROS Client Library<br/>(RCL)<br/>rclcpp::Publisher / rclpy"]
    C["RMW<br/>(ros middleware interface)<br/>e.g. rmw_cyclonedds_cpp, rmw_fastrtps_cpp"]
    D["DDS Publisher<br/>(DataWriter)<br/>- Serializes message to DDS/IDLC format<br/>- Applies QoS<br/>- Sends data to matching DDS Subscribers"]
    E["RTPS<br/>(DDSI-RTPS) Protocol<br/>- discovery packets<br/>- DATA / HEARTBEAT submessages"]
    F["Operating System Transport<br/>UDPv4 / UDPv6 / Shared Memory<br/>- Network routing, loopback"]
    G["DDS Subscriber<br/>(DataReader)<br/>- Deserializes DDS message<br/>- Applies QoS filtering<br/>- Hands sample to RMW"]
    H["RCL<br/>(Python/C++)<br/>Calls user-defined callback"]
    I["LISTENER<br/>(Subscriber Node)<br/>e.g. void callback"]
    
    A -->|1. Your code calls<br/>publisher->publish| B
    B -->|2. RCL converts ROS<br/>message into RMW format| C
    C -->|3. RMW forwards data<br/>to DDS Writer| D
    D -->|4. DDS sends message<br/>using RTPS protocol| E
    E -->|5. Transport: UDP Multicast<br/>/ Unicast / Shared Memory| F
    F -->|6. RTPS packets received<br/>by DDS Subscriber| G
    G -->|7. RMW delivers ROS<br/>message to RCL| H
    H -->|8. Your callback<br/>executes| I
```
