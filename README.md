# Breast Cancer Detection

A machine learning pipeline to classify breast cancer ultrasound images as benign or malignant.

- Extracts features from 6 image regions — tumor, peritumoral rings, posterior area, skin, and background tissue
- Handcrafted texture features: intensity stats, GLCM, LBP, edge density
- Started with ConvNeXt for deep embeddings — hit a performance plateau
- Switched to UltraSAM (pretrained on ultrasound data) — better domain-specific features
- Trained and compared 5 classifiers: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM
- Evaluated on AUC, accuracy, sensitivity, specificity, and F1-score
- Dataset provided by AIIMS
- Built with Python, PyTorch, scikit-learn, XGBoost, LightGBM
