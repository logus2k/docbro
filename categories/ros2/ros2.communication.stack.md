# **1. Diagram — ROS 2 Communication Stack (Text-based / Markdown)**

```
+----------------------------------------------------------+
|                  ROS 2 Node (Python/C++)                 |
|----------------------------------------------------------|
|     Publishers, Subscribers, Services, Actions, etc.     |
+---------------------------▲------------------------------+
                            |
          ROS Client Libraries (RCL)  →  rclpy / rclcpp
                            |
+---------------------------▼------------------------------+
|               ROS Middleware Interface (RMW)             |
|  - Thin abstraction layer                                |
|  - Selects middleware via $RMW_IMPLEMENTATION            |
|  - Examples: rmw_fastrtps_cpp, rmw_cyclonedds_cpp        |
+---------------------------▲------------------------------+
                            |
                   DDS Implementation
                            |
+---------------------------▼------------------------------+
|                   DDS Middleware Layer                   |
|  (e.g., Fast DDS, Cyclone DDS, GurumDDS, RTI Connext)    |
|  Handles:                                                 |
|    - Discovery (who exists?)                              |
|    - QoS enforcement                                      |
|    - Serialization                                        |
|    - Reliability, durability, deadlines                   |
|    - Matching pubs/subs                                   |
+---------------------------▲------------------------------+
                            |
               RTPS (DDSI-RTPS) Wire Protocol
                            |
+---------------------------▼------------------------------+
|        Transport Layer (UDPv4 / UDPv6 / Shared Memory)   |
|        OS Networking, multicast, loopback, etc.          |
+----------------------------------------------------------+
```

This shows how a ROS node uses RCL → RMW → DDS → RTPS → Transport to communicate.

---
