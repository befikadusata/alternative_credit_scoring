# Model Serving Strategy

The primary goal of the model serving strategy is to deploy new models into production safely and reliably, without compromising system stability or performance. This is achieved using a **Champion-Challenger** framework.

## 1. Champion-Challenger Framework

The Champion-Challenger model is a deployment pattern that allows for the live testing of a new model (the "Challenger") against the current production model (the "Champion").

-   **Champion:** The model currently in the `Production` stage in the MLflow Model Registry. It handles the majority of the live traffic.
-   **Challenger:** A model in the `Staging` stage. It is a candidate for promotion and receives a small, configurable percentage of live traffic to test its performance in a real-world environment.

### Traffic Routing Logic

Incoming Request
       │
       ▼
   Is Testing?
       │
   ┌───┴────┐
   │        │
  NO       YES
   │        │
   ▼        ▼
Champion  Random(0,1)
(100%)       │
         ┌───┴────┐
         │        │
      <0.90    >=0.90
         │        │
         ▼        ▼
     Champion  Challenger
      (90%)     (10%)

### Implementation Overview

A `ChampionChallengerRouter` class is implemented within the prediction service to manage this logic.

-   **Initialization:**
    -   On startup, the router connects to the MLflow Tracking Server.
    -   It loads the latest model version from the `Production` stage to serve as the **Champion**.
    -   It also loads the latest model version from the `Staging` stage to serve as the **Challenger**. If no model is in Staging, only the Champion is served.

-   **Prediction Flow:**
    1.  When a prediction request arrives, the router sends the input features to **both** the Champion and Challenger models to get their respective predictions. This is crucial for comparing their performance on the exact same data.
    2.  A random number is generated to decide which model's prediction to return to the user, based on a pre-configured weight (e.g., 90% Champion, 10% Challenger).
    3.  A detailed log entry is created that records the request, the predictions from both models, and which model was chosen to serve the response.
    4.  The final response sent to the client includes the prediction from the chosen model.

-   **Benefits:**
    -   **Safe Deployment:** New models are tested on live data with minimal risk. Any issues with the Challenger model affect only a small subset of users.
    -   **Data-Driven Decisions:** The logged predictions from both models can be analyzed offline to compare their performance on key metrics (e.g., AUC, precision, latency). This allows for a data-driven decision on whether to promote the Challenger.
    -   **Zero-Downtime Promotion:** When a Challenger is promoted, the router can seamlessly switch to the new Champion without any service interruption.

## 2. Model Hot-Reloading

The prediction service is designed to periodically check the MLflow Model Registry for new model versions in the `Production` or `Staging` stages. When a new version is detected (e.g., after a model has been promoted), the router automatically downloads and loads the new model into memory, ensuring the service always uses the latest approved versions without requiring a server restart.
