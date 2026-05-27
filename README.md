# Postgraduate Program in Software Engineering and Applied AI 🚀

This repository is dedicated to storing practical examples, labs, and projects developed throughout the **Postgraduate Program in Software Engineering and Applied AI**.

---

## 📂 Repository Structure

The repository is organized by practical examples and projects that cover everything from machine learning fundamentals to complex software architectures integrated with AI:

| Directory | Description | Key Technologies |
| :--- | :--- | :--- |
| **[`example-00`](file:///Users/andrecoelho/Documents/software-engineering-and-applied-ai/example-00)** | Sequential Neural Network for user profile classification. | Node.js, `@tensorflow/tfjs-node` |

---

## 🔬 Module Details

### 🔹 [Example 00: Multiclass Classification with TensorFlow.js](file:///Users/andrecoelho/Documents/software-engineering-and-applied-ai/example-00)

A practical demonstration of a complete machine learning workflow using JavaScript on the backend (Node.js). The project covers data preparation, network architecture definition, supervised training, and inference.

* **Neural Network Architecture**:
  * **Input Layer**: 7 neurons (Normalized age + One-Hot encoded favorite colors + One-Hot encoded cities).
  * **Hidden Layer**: 80 neurons with **ReLU** activation (filtering values and allowing the network to learn complex non-linear relationships).
  * **Output Layer**: 3 neurons with **Softmax** activation (generating probabilities for the `premium`, `medium`, and `basic` classes).
* **Training Process**:
  * Compiled with the **Adam** optimizer and **Categorical Crossentropy** loss function.
  * Trained for 100 epochs (`epochs: 100`) with active shuffling (`shuffle: true`) to prevent bias.
  * Real-time feedback of loss evolution at the end of each epoch.
* **Practical Demonstration**:
  * Includes data normalization and encoding for new test users, displaying the estimated probabilities in sorted order.

---

## ⚙️ How to Run the Projects

Each folder contains its own pre-configured environment. Here is how to run the projects locally:

### Prerequisites
* **Node.js** (version 18 or higher recommended)
* **npm** or your preferred package manager.

### Step-by-Step Instructions (`example-00`)

1. **Navigate to the project folder**:
   ```bash
   cd example-00
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the script in watch mode**:
   ```bash
   npm start
   ```
   * *Note: The project is configured to use Node.js's native `--watch` mode, automatically reloading the file when changes are detected.*

---

> [!NOTE]
> **Contributions and Studies:** This is an academic and practical repository. Feel free to clone it, experiment with neural network parameters (such as the number of neurons in the hidden layer, number of epochs, and learning rate), and create new examples.

> [!TIP]
> **Engineering Best Practices:** When adding new projects, ensure you organize their dependencies in a dedicated `package.json` file and document their purpose in the main table.
