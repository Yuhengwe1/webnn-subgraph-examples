# WebNN Subgraph Examples

This repository contains small JavaScript pseudocode examples for discussing how WebNN could represent high-level operations as configurable subgraphs. It supports the discussion in [WebNN issue #907: Composite operators / subgraphs](https://github.com/webmachinelearning/webnn/issues/907).

The subgraph is independent of the WebNN backend. A WebNN implementation using ONNX Runtime, TFLite, Core ML, or another backend could lower the primitive body or select a compatible optimized implementation, subject to its capabilities.