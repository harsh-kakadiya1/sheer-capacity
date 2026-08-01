Integrating Artificial Intelligence (AI) and Machine Learning (ML) into your existing research on the high-temperature performance of composite deck shear connectors is a highly strategic move. Your current dataset—encompassing mechanical properties, ISO 834 fire exposures, connector geometries, and non-linear load-slip curves—forms the perfect foundational matrix for training advanced predictive models.

Recent literature from 2021 to 2026 demonstrates that AI can effectively act as a high-speed "surrogate model," bypassing computationally heavy Finite Element Analysis (FEA) to predict highly non-linear thermo-structural behaviors.

Here is a detailed, step-by-step roadmap on how to integrate machine learning into your project, including the methodology and the specific results you need to generate.

Part 1: How to Execute the AI/ML Integration
Step 1: Data Augmentation via Automated FEA (Synthetic Data Generation) Machine learning requires massive amounts of data. Since high-temperature push-out tests are expensive and time-consuming, your experimental data will likely suffer from statistical discreteness.


The Approach: Use your validated experimental load-slip graphs to calibrate a non-linear Finite Element Model (FEM) in software like ABAQUS.


Automation: Utilize Python scripting via the Abaqus API to automate a massive parametric study. You can program the script to automatically alter the inputs (e.g., connector height, concrete grade, temperature curve) across thousands of permutations, run the analysis, and extract the resulting load-slip data. This creates a robust, synthetic training dataset.

Step 2: Feature Engineering (Defining Inputs and Outputs)

You must structure your dataset so the algorithm understands the physics.

Input Features: Temperature, exposure time, specific heat capacity, concrete compressive strength, steel yield strength, and the precise geometric dimensions of the connectors (e.g., stud diameter, channel web thickness).

Target Output: The algorithm should be trained to predict the ultimate shear capacity and the specific coordinate points of the non-linear load-slip curve.

Step 3: Algorithm Selection and Training

Because the load-slip response of shear connectors under fire is violently non-linear, simple linear regression will fail. You should employ advanced algorithms:

Tree-Based Ensembles: Algorithms like Extreme Gradient Boosting (XGBoost), Random Forest (RF), and Decision Trees (DT) have proven exceptionally accurate at capturing non-linear patterns and feature interactions in structural fire engineering.

Hybrid Neural Networks: You can also deploy a Multi-Layer Perceptron (MLP) optimized by Particle Swarm Optimization (PSO). The PSO algorithm prevents the neural network from getting trapped in local minima during training, allowing it to highly accurately predict slip responses at elevated temperatures.


Step 4: Explainable AI (XAI) Implementation

A major criticism of AI in civil engineering is that it acts as a "black box." To maintain academic rigor, you must implement SHAP (SHapley Additive exPlanations) analysis. SHAP values will explicitly quantify how much each specific parameter (e.g., changing the concrete grade vs. changing the connector height) influenced the algorithm's final prediction, thereby linking the AI's mathematical weights back to actual structural physics.

Part 2: Required Results to Generate
To publish a high-quality analysis or complete your report, your AI integration must generate and present the following specific results:

1. Statistical Performance Metrics

You must prove the accuracy of your predictive models by calculating and reporting specific statistical errors for both your training and testing datasets. Required metrics include:

Coefficient of Determination ($R^2$): Target values above 0.90 to prove the model explains the variance in the shear capacity.

Root Mean Square Error (RMSE) and Mean Absolute Error (MAE): To quantify the average prediction error in ultimate load capacity (measured in kN).

2. AI-Predicted vs. FEA/Experimental Load-Slip Curves

Generate graphical plots that overlay the entire non-linear load-slip curves predicted instantaneously by the AI surrogate model against the actual curves obtained from your physical furnace tests and ABAQUS models. This visually validates that the AI accurately captures the initial stiffness, peak load, and the plastic post-peak degradation phases.

3. Feature Importance Ranking (SHAP Summary Plots) Generate a SHAP dependency plot that ranks your influencing parameters from most to least critical. For instance, your results might definitively show that at 600°C, the geometric height of a channel connector dictates survivability more heavily than increasing the concrete grade.


4. A Predictive Design Tool (Optional but Highly Recommended)

As a final output, translate your trained XGBoost or MLP-PSO model into a simple graphical user interface (GUI) or script. This tool should allow structural engineers to input a desired concrete grade, fire temperature, and connector type, and instantly receive the predicted residual shear capacity and expected slip without needing to run an extensive FEA simulation themselves.