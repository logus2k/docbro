# Comprehensive Research Report on Decision Patterns
**Focus:** Origins, Taxonomy, Composition, and Implementation of Decision Patterns in Systems and Business Architecture

---

## 1. Executive Summary
Decision patterns are standardized, reusable templates and models used to solve recurring decision-making problems. In modern enterprise environments, "decision patterns" span two tightly coupled disciplines: **Architectural Decision Patterns** (how technical and structural choices are made and documented) and **Decision Management Patterns** (how business logic and operational decisions are modeled and executed). This report provides a detailed examination of how these patterns originated, lists the most relevant patterns in industry use today, and explains the mechanics of their composition and technical implementation.

---

## 2. The Origins of Decision Patterns

The concept of formalizing decisions into patterns originates from a convergence of cognitive psychology, software engineering, and business process optimization.

### 2.1 Cognitive and Theoretical Origins
The foundational roots of decision patterns lie in **Decision Theory** and **Bounded Rationality** (introduced by Herbert Simon in the 1950s). Recognizing that humans cannot process infinite variables, cognitive science identified that human minds use "heuristics"—mental patterns—to make choices. Over time, as organizations grew complex, these internal cognitive heuristics needed to be externalized into mathematical and logical models to ensure consistency.

### 2.2 Origins in Software Architecture
In the 1990s, the "Gang of Four" (GoF) introduced software *design patterns*—reusable solutions to coding problems. However, design patterns only captured the *result* of a decision, not the *rationale*. To capture the "why," software architects developed **Architectural Decision Records (ADRs)** and **Architectural Decision Patterns**. These originated from the need to preserve institutional memory, prevent repeated debates over the same trade-offs, and standardize the evaluation of structural system choices.

### 2.3 Origins in Business Rules and DMN
Simultaneously, business operations suffered from "hardcoded" logic. When business rules were buried inside application code, non-technical stakeholders could not verify or change decisions. This led to the creation of Business Rule Management Systems (BRMS) in the 2000s, which eventually evolved into the **Decision Model and Notation (DMN)** standard published by the Object Management Group (OMG) in 2015. DMN established a universal, standardized set of visual and logical patterns to isolate decision-making from process execution.

---

## 3. Identification and List of Relevant Decision Patterns

Decision patterns can be categorized into two primary domains: **Architectural/Structural Patterns** and **Operational/Logic Patterns (DMN)**.

### Category 1: Architectural Decision Patterns
These patterns govern how systems are built, scaled, and maintained. They dictate the trade-offs between competing technical paradigms.

1.  **Declarative vs. Imperative Pattern:** 
    *   *Declarative:* Defining the desired end-state (what should happen) and letting the system figure out how to achieve it.
    *   *Imperative:* Defining the exact sequential steps (how it should happen).
2.  **Mutable vs. Immutable Pattern:** 
    *   Deciding whether an asset (like a server or database state) can be updated in-place (mutable) or if it must be entirely replaced with a new instance upon change (immutable).
3.  **Event-Driven vs. Request-Response Pattern:**
    *   Determining communication flow. Event-driven reacts to state changes asynchronously, while request-response requires synchronous, two-way communication.
4.  **Downselecting Pattern:**
    *   A mathematical/architectural pattern used to select a highly synergistic subset of entities from a massive candidate pool (e.g., choosing a specific stack of microservices that integrate well while minimizing capability redundancy).

### Category 2: Decision Management & Logic Patterns (DMN)
These patterns dictate how business rules are executed and evaluated at runtime. They are visually represented and mathematically executed.

1.  **The Decision Table Pattern:** A tabular representation of decision logic. It utilizes specific "Hit Policies" to determine the outcome:
    *   *Unique (U):* Only one rule (row) can match the input.
    *   *Any (A):* Multiple rules can match, but they must all yield the same output.
    *   *Priority (P):* Multiple rules can match; the output is selected based on a predefined priority list.
    *   *Collect (C):* Aggregates outputs from multiple matching rules (e.g., Sum, Min, Max, Count).
2.  **Decision Requirements Graph (DRG) / Cascading Decision Pattern:**
    *   A structural pattern where complex decisions are decomposed into smaller, chained sub-decisions. The output of a lower-level decision becomes the input for a higher-level decision.
3.  **Business Knowledge Model (BKM) Pattern:**
    *   A pattern for encapsulating reusable decision logic. Instead of rewriting a tax-calculation rule in ten different tables, the logic is stored in a BKM and invoked via parameters by multiple higher-level decisions.
4.  **Literal / Boxed Expression Pattern:**
    *   Used for highly complex mathematical or conditional logic that does not fit neatly into a table, often utilizing FEEL (Friendly Enough Expression Language) to define intervals, date mathematics, and list filtering.

---

## 4. Composition of Decision Patterns

Composition refers to how the components of a pattern are structured and linked together to form a complete, coherent decision unit.

### 4.1 Composing Architectural Decisions (ADRs)
An architectural decision pattern is composed as a structured document, typically containing the following components:
*   **Title:** A clear, concise statement of the decision.
*   **Context/Forces:** The business drivers, technical constraints, and environmental factors necessitating a decision.
*   **Alternatives:** A matrix of the available options (e.g., Option A vs. Option B).
*   **Decision:** The final choice selected.
*   **Consequences (Trade-offs):** A quantified explanation of what is gained and what is sacrificed (e.g., "We gain horizontal scalability but incur network latency").

### 4.2 Composing Business Decisions (DMN)
DMN patterns are composed visually and logically using a Computation Independent Model approach:
*   **Input Data Nodes:** Representing the raw facts or variables entering the system (e.g., "Customer Age", "Current Season").
*   **Decision Nodes:** The logic boxes that evaluate the inputs.
*   **Knowledge Requirements (Arrows):** The directional flow composing the network, linking inputs to decisions, and sub-decisions to main decisions.
*   **Underlying Logic (Boxed Expressions):** Inside every Decision Node is a composed table or FEEL expression. The composition requires strict alignment of input data types (e.g., ensuring a string like "Summer" is not evaluated against an integer).

---

## 5. Implementation of Decision Patterns

Implementation refers to how these composed patterns are physically executed, stored, and managed in a production environment.

### 5.1 Implementing Architectural Patterns
*   **Version Control & Repo Integration:** ADRs are implemented as Markdown (`.md`) or AsciiDoc files stored directly in the source code repository (e.g., in a `/docs/adr` directory). This ensures the decision history travels with the code.
*   **Infrastructure as Code (IaC):** Patterns like *Declarative Infrastructure* are implemented using tools like Terraform or Kubernetes manifests. The pattern is codified so that the infrastructure engine continuously enforces the decision (the "desired state").

### 5.2 Implementing Operational/DMN Patterns
*   **Decision Engines (Rule Engines):** DMN patterns are serialized into XML files. These XML files are deployed to specialized software called Decision Engines (such as Camunda, Red Hat Drools, or Trisotech). 
*   **Direct Execution:** Unlike old business requirements that had to be translated into Java or Python by a programmer, DMN XML files are *directly executable*. The engine reads the XML, parses the tables, and exposes the decision pattern as an API endpoint (e.g., a REST API).
*   **Runtime Evaluation via FEEL:** During execution, a client application sends a JSON payload of input data to the engine. The engine uses FEEL (Friendly Enough Expression Language) to evaluate the inputs against the composed rules, navigating down the Decision Requirements Graph, and instantly returning the final computed decision.

---

## 6. Conclusion
Decision patterns provide a necessary bridge between chaotic, unstructured human choices and scalable, predictable system behaviors. By leveraging Architectural Decision Records, organizations maintain the institutional memory of system trade-offs. Simultaneously, by implementing Decision Model and Notation (DMN) patterns, enterprises can separate their operational logic from underlying code, allowing business rules to be visible, agile, and automatically executable by decision engines.