# Section 3: Data Engineering Fundamentals

This section explores the critical role of data engineering in ML systems, covering how data is collected, formatted, stored, and transported across different services.

## I. Data Sources
ML systems draw data from varied sources, each requiring different handling and processing strategies.

> "Understanding the sources your data comes from can help you use your data more efficiently."

*   **User Input:** Explicit data like text or images provided by users. It is often malformatted and requires heavy validation.
*   **System-Generated Data:** Logs and system outputs (like model predictions) used for debugging and monitoring.
*   **Internal Databases:** Managed assets like inventory and customer relationships used to provide context for ML predictions.
*   **Third-Party Data:** Data collected by other companies, which has become more restricted due to privacy regulations like the move toward opt-in advertiser IDs.

## II. Data Formats and Storage
The choice of data format significantly impacts system performance, storage costs, and access speed.

### **Row-Major vs. Column-Major**
Row-major (e.g., CSV) is optimized for writing new data, while column-major (e.g., Parquet) is optimized for reading specific features across many records.

> "Row-major formats are better when you have to do a lot of writes, whereas column-major ones are better when you have to do a lot of column-based reads."

> "Column-major formats allow flexible column-based reads, especially if your data is large with thousands, if not millions, of features."

### **Text vs. Binary**
Text formats (JSON, CSV) are human-readable but consume more space than binary formats (Parquet, Avro, Protobuf), which are more compact and faster for machines to process.

## III. Data Models
Data models define how information is represented and structured, which dictates the types of queries the system can efficiently handle.

### **Relational vs. NoSQL**
The relational model uses strict schemas and normalization to reduce redundancy, whereas NoSQL models offer more flexibility.

> "Relational models are among the most persistent ideas in computer science... In this model, data is organized into relations; each relation is a set of tuples."

*   **Document Model:** Best for self-contained data with rare relationships between items.
*   **Graph Model:** Best when the priority is the relationship between data items (e.g., social networks).

### **Data Warehouse vs. Data Lake**
*   **Data Warehouse:** For structured data that has been processed into ready-to-use formats.
*   **Data Lake:** For raw, unstructured data before processing, allowing for "fast arrival" of data.

## IV. Storage Engines and Processing
Databases are optimized for different types of workloads: transactional (OLTP) or analytical (OLAP).

> "Transactional databases are designed to process online transactions and satisfy the low latency, high availability requirements."

> "Analytical databases are designed for [queries that allow you to look at data from different viewpoints]... This type of processing is known as online analytical processing (OLAP)."

### **ETL vs. ELT**
ETL processes data before loading it into storage, while ELT loads raw data first and transforms it later.

> "ETL stands for extract, transform, and load... This process refers to the general purpose processing and aggregating of data into the shape and the format that you want."

## V. Modes of Dataflow
Data must be passed between processes that do not share memory, using one of three main modes:

1.  **Passing Through Databases:** Simple but slow; requires shared access.
2.  **Passing Through Services:** Uses request-driven architectures like REST or RPC APIs.
3.  **Passing Through Real-Time Transport:** Uses brokers like Kafka or Kinesis for event-driven, asynchronous communication.

## VI. Batch vs. Stream Processing
These two paradigms handle historical data versus data that is currently streaming in.

> "Batch features—features extracted through batch processing—are also known as static features."

> "Streaming features—features extracted through stream processing—are also known as dynamic features."

While many believe stream processing is less efficient, modern technologies allow for scalable, stateful computation that can unify both pipelines.

> "It’s easier to make a stream processor do batch processing than to make a batch processor do stream processing."
