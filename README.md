# Enhanced Credit Risk Probability Model for Alternative Data

This is an improved version of the 10 Academy Week 5 project for building, deploying, and automating a credit risk model using eCommerce transaction data.

## Credit Scoring Business Understanding

1. **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**  
   Basel II mandates accurate credit risk quantification for capital requirements, pushing for interpretable models (e.g., logistic regression) to enable regulatory validation, transparency in decision-making, and thorough documentation for audits and compliance.

2. **Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**  
   A proxy (e.g., via RFM clustering) is essential to enable supervised learning on alternative data for predicting defaults. Risks include inaccurate risk signals leading to faulty loan decisions, increased bad debts, lost revenue from denied good customers, and regulatory penalties if the proxy misaligns with actual default behaviors.

3. **What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**  
   Simple models provide explainability, easier regulatory approval, and lower costs but may sacrifice accuracy on complex data. Complex models offer superior predictive power and handle non-linear patterns but are opaque, raising explainability issues, bias risks, and challenges in regulated audits.

## Project Improvement for Week 12
I chose the Week 5 "Credit Risk Probability Model for Alternative Data" project from 10 Academy, building an end-to-end credit scoring model for Bati Bank using eCommerce RFM data to predict defaults, assign scores, and recommend loans, compliant with Basel II. In the original week, I finished business summary in README.md, EDA, feature engineering with RFM clustering for proxies, model training (Logistic Regression, Random Forest) via MLflow, tests, and FastAPI/Docker deployment with basic CI/CD. I aim to improve it for finance reliability by adding modularity, tests, CI/CD auto-deploy, SHAP explainability, Streamlit dashboard, and Optuna tuning to demonstrate 20-30% lending efficiency gains. Week plan: Wed (13 Aug) - Refactor code, update README; Thu (14 Aug) - Add tests/CI/CD; Fri (15 Aug) - Add SHAP, start dashboard; Sat (16 Aug) - Finish dashboard/tuning; Sun (17 Aug) - Interim report/fixes.