<p align="center">
    <img src="hag.png" align="center" width="30%">
</p>
<p align="center"><h1 align="center">HAG</h1></p>
<p align="center">
	<em>Experiments on bio inspired plasticity to improve reservoir computing</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Finebouche/HAG?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/languages/top/Finebouche/HAG?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://zenodo.org/badge/doi/10.1109/ijcnn54540.2023.10191230.svg" alt="repo-language-count">
	<img src="https://zenodo.org/badge/doi/10.1109/rivf60135.2023.10471845.svg" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [📚 Publications](#-publications)
- [👾 Features](#-features)
- [🚀 Setup](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)

- [🎗 License](#-license)

---

## 📍 Overview

HAG introduces an innovative, biologically-inspired approach to improve Reservoir Computing networks. Grounded in Hebbian plasticity principles, HAG dynamically constructs and optimizes reservoir architectures to enhance the adaptability and efficiency of time-series prediction and classification tasks. By autonomously forming and pruning connections between neurons based on Pearson correlation, HAG tailors reservoirs to the specific demands of each task, aligning with biological neural network principles and Cover’s theorem.

---
## 📚 Publications

### **2025**

- **[Reshaping reservoirs with unsupervised Hebbian adaptation](https://doi.org/10.1038/s41467-025-67137-1)**  
  *Tanguy Cazalets, Joni Dambre*  
  *Nature Communications*  
  This paper introduces HAG.

### **2023**

- **[A Bio-Inspired Model for Audio Processing](https://doi.org/10.1109/ijcnn54540.2023.10191230)**  
  *Tanguy Cazalets, Joni Dambre*  
  *Presented at RIVF 2023*  
  This paper introduces a biologically-inspired approach to audio processing, emphasizing homeostatic mechanisms and plasticity for efficient neural network performance.

- **[A Homeostatic Activity-Dependent Structural Plasticity Algorithm for Richer Input Combination](https://doi.org/10.1109/rivf60135.2023.10471845)**  
  *Tanguy Cazalets, Joni Dambre*  
  *Presented at IJCNN 2023*  
  This work explores an innovative algorithm for structural plasticity, enhancing neural network adaptability to diverse input combinations.

---
## 👾 Features

|     | Feature         | Summary                                                                                                                                                                                        |
|:----| :---:           |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🔬️ | **Dynamic Reservoirs**  | Dynamically generates connectivity matrices using Hebbian-inspired rules, ensuring optimized, task-specific reservoir properties for enhanced linear separability and efficiency.         |
| 🧩  | **Structural Plasticity**    | Implements biologically plausible mechanisms to create or prune connections based on activity levels and correlations, enabling the reservoir to self-organize around task requirements.  |
| ⚡️  | **Performance Boost**   | Outperforms traditional Echo State Networks (ESNs) across various benchmarks, offering higher accuracy in classification and reduced error in time-series prediction tasks.                    |
| 📊  | **Comprehensive Metrics**  | Evaluates reservoirs with advanced metrics including Pearson Correlation, Spectral Radius, and Cumulative Explained Variance to ensure enriched dynamics and decorrelated feature representations. |


---
## 🚀 Setup

### ☑️ Prerequisites

Before getting started with HAG, ensure your runtime environment meets the following requirements:

- <code>reservoirPy</code> for reservoir computing training and inference
- <code>optuna</code> for hyperparameter optimization
- <code>librosa</code> for time-series preprocessing

### ⚙️ Installation

Install HAG using one of the following methods:

**Build from source:**

1. Clone the HAG repository:
```sh
❯ git clone https://github.com/Finebouche/HAG
```

2. Navigate to the project directory:
```sh
❯ cd HAG
```

3. Install the project dependencies:


**Using `conda`** &nbsp; [<img align="center" src="https://img.shields.io/badge/conda-342B029.svg?style={badge_style}&logo=anaconda&logoColor=white" />](https://docs.conda.io/)

```sh
❯ conda env create -f environment.yml
```



## 🎗 License

This project is protected under the [MIT License ](https://choosealicense.com/licenses/mit/) License.

---

## 🙌 Acknowledgments

- This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 860949

---
