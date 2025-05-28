# Game-Theoretic Clustering Algorithm

This repository provides an implementation of a game-theoretic clustering algorithm with penalty-based payoff design. The code supports experiments on multiple graph datasets, parameter sensitivity analysis, and dynamic graph scenarios.

## 🧪 How to Run Parameter Experiments

To conduct parameter experiments on a specific dataset:

1. Uncomment the **setup code block** for the desired graph (among the four datasets).
   - ⚠️ Only uncomment the lines **up to (but not including)** the `clustering_algorithm` function call.
2. Then, uncomment the **parameter test block** located below the dataset setup.

In the parameter test block, there are two options for the `predicted_labels` variable.  
→ **Only uncomment one of them**, depending on the type of evaluation you wish to perform (refer to the comment next to each line for guidance).

---

## 🧭 How to Run Clustering for Each Graph

If you want to perform full clustering on a specific graph, simply **uncomment the entire code block**, including the call to the `clustering_algorithm` function.

---

## 🔄 How to Run Dynamic Graph Experiments

To evaluate the algorithm on dynamic graphs (e.g., node addition or removal), **scroll to the bottom of the script** and execute the **`dynamic graph test` block** by uncommenting it.

---

For any questions or issues, feel free to contact the maintainer or open an issue.
