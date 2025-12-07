## Webots Physics Guide

Webots uses a powerful physics library (ODE - Open Dynamics Engine) to calculate how objects move and interact. Setting the physics parameters correctly is the key to creating realistic simulations. 

---

### **1. Core Body Properties: Mass & Inertia**

These properties define how difficult it is to move or rotate an object. They are typically set on the **Solid** or **SolidReference** nodes that define the physical components of your robot.

| Physics Term | Where to Set in Webots | Practical Use for Beginners |
| :--- | :--- | :--- |
| **Mass** ($m$) | Found in the **`mass`** field of the **Solid** node (or calculated automatically if a **`PBRAppearance`** node is used with density). | **Crucial:** Setting the correct mass (in kg) determines the robot's **weight** and how much force is needed for **acceleration**. A massless object will float away! |
| **Inertia** ($I$) | Found in the **`inertia`** field of the **Solid** node. | **Important:** For simple shapes like boxes or cylinders, Webots calculates this automatically. For complex shapes, you might need to manually set the **`inertiaMatrix`**. It defines how hard it is to **rotate** the body. |

---

### **2. Forces of Motion: Torque, Velocity, and Gravity**

These are the forces that drive the robot's movement or are exerted upon it by the environment.

| Physics Term | How to Apply/View in Webots | Practical Use for Beginners |
| :--- | :--- | :--- |
| **Torque** ($\tau$) | Applied through **Motor** or **RotationalMotor** nodes. You control this by setting the motor's **`velocity`** and **`maxForce`** (which is actually a torque limit). | **Motor Control:** The motor applies **torque** to the joint to achieve the desired rotational **velocity**. Setting **`maxForce`** too low means the motor won't be strong enough to lift the robot's **weight**. |
| **Velocity** ($v$) | Can be read through the sensor functions (e.g., `wb_motor_get_velocity()`) or observed directly in the simulation. | **Feedback:** Use the robot's **velocity** to implement control logic, such as a PID controller to maintain a constant speed. |
| **Gravity** ($g$) | Set globally in the **WorldInfo** node under the **`gravity`** field. | **Environmental Setup:** If you are simulating on Earth, leave it at the default ($9.81$). For a moon simulation, you would lower this value significantly. |

---

### **3. Interaction Properties: Friction & Momentum**

These define how the robot interacts with the floor and other objects, especially during collisions.

| Physics Term | Where to Set in Webots | Practical Use for Beginners |
| :--- | :--- | :--- |
| **Friction** ($f$) | Set within the **Physics** node of the ground object using the **`coulombFriction`** parameter. | **Traction:** **Friction** is essential for wheels to roll without slipping. High friction prevents slipping; low friction (like ice) causes sliding. This often requires experimentation! |
| **Weight** ($W$) | Not set directly; it's the result of **Mass** $\times$ **Gravity**. | **Collision:** The robot's **weight** determines the strength of the normal force between its wheels and the ground, which in turn affects the maximum possible **friction** force. |
| **Restitution** | Set within the **Physics** node using the **`restitution`** parameter. | **Bounciness:** This defines how much **Kinetic Energy** is conserved during a collision. A value of $0$ means no bounce (perfectly plastic collision), and $1$ means a perfectly elastic collision (full bounce). |

By focusing on these nodes and parameters, you'll be able to quickly adjust your robot's physical behavior to match your simulation goals!

| Term | Symbol/Unit | Definition & Relevance to Webots |
| :--- | :--- | :--- |
| **Mass** | $m$ (kg) | A measure of the amount of matter in an object. In Webots, the **mass** of a robot or its links affects its **inertia**, how it responds to **forces**, and how it interacts in **collisions**. |
| **Force** | $F$ (N) | An interaction that, when unopposed, will change the motion of an object. In Webots, forces like motor **torque**, **gravity**, and contact **forces** drive the robot's movement and interactions. |
| **Momentum** | $p$ ($kg \cdot m/s$) | The product of an object's **mass** and its **velocity** ($p = m \cdot v$). It represents the quantity of motion. It is conserved in Webots collisions, influencing post-collision velocities. |
| **Inertia** | $I$ ($kg \cdot m^2$) | An object's resistance to a change in its state of motion (linear) or rotation (rotational inertia or **moment of inertia**). Higher **inertia** means a robot is harder to accelerate or turn. |
| **Gravity** | $g$ ($m/s^2$) | The acceleration experienced by an object due to gravitational attraction. In Webots, it's typically set to the standard Earth value ($9.81 \, m/s^2$) and is the primary force pulling objects downward. |
| **Weight** | $W$ (N) | The **force** exerted on a **mass** by **gravity** ($W = m \cdot g$). This is the force pulling the robot onto the ground surface in a simulation. |
| **Torque** | $\tau$ ($N \cdot m$) | A twisting **force** that causes rotation. In Webots, **torque** is the key parameter for driving robot **joints** and **motors**, causing wheels to turn or arms to swing. |
| **Friction** | $f$ (N) | A **force** that opposes motion when two surfaces are in contact. **Friction** is critical in Webots for realistic wheel grip (traction) and for stopping objects from sliding indefinitely. |
| **Velocity** | $v$ ($m/s$) | The rate of change of an object's position with respect to time (includes speed and direction). It determines the object's movement in the simulation. |
| **Acceleration** | $a$ ($m/s^2$) | The rate of change of an object's **velocity** with respect to time. It is caused by an unopposed net **force** acting on the object ($F = m \cdot a$). |
| **Kinetic Energy** | $E_k$ (J) | The energy possessed by an object due to its motion ($E_k = \frac{1}{2} m v^2$). This energy is exchanged during collisions. |

---
