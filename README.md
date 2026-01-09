# 🔐 Password Intelligence System (Multi-Model ML App)

This project is a **multi-model machine learning–based password analysis system** that evaluates password security, improves weak passwords, and provides intelligent recommendations.
All decisions are made using **trained machine learning models only**.

## 🧠 Machine Learning Models Used

The system integrates **multiple independent ML models**, each responsible for a specific task in password intelligence.

### 1️⃣ Password Strength Prediction Model

**Files used**

- `enhanced_password_model_1.pkl`
- `tfidf_vectorizer_1.pkl`

**Purpose**
Predicts the **overall strength of a password**.

**How it works**

- The password is converted into numerical features using **TF-IDF vectorization**
- The trained classification model predicts:

  * `Weak`
  * `Medium`
  * `Strong`

**Output example**

```
Input: koyel@123!RAM
Predicted Strength: Medium
```

### 2️⃣ Password Improvement Model

**File used**

- `password_improvement_model.pkl`

**Purpose**
Generates a **stronger version of the input password**.

**How it works**

- Uses the same vectorized representation of the password
- Predicts an improved password with:

  * Higher complexity
  * Better character distribution
  * Increased resistance to attacks

**Output example**

```
Original Password: koyel
Improved Password: Ekoyel?,^2u/
```

### 3️⃣ Password Length Recommendation Model

**Files used**

- `length_recommender_model.pkl`
- `length_vectorizer.pkl`

**Purpose**
Predicts the **optimal password length** for better security.

**How it works**

- Analyzes the structure and patterns of the input password
- Outputs a **recommended length** that improves password strength

**Output example**

```
Current Length: 7
Recommended Length: 14
```

### 4️⃣ Advanced Password Recommendation Model

**Files used**

- `new_recommender_model.pkl`
- `new_vectorizer.pkl`

**Purpose**
Generates **additional strong password suggestions**.

**How it works**

- Uses a separate ML pipeline trained on strong-password patterns
- Produces multiple alternative secure passwords

**Output example**

```
ML Suggestions:
- jmR6%MNAQM'S
- |I-83o6JJ^Wk
- ig7//tiQ+J2h
```

## 🔄 How All Models Work Together

1. User enters a password
2. Password is processed by **multiple ML pipelines**
3. The system returns:

   * Predicted password strength
   * Improved password
   * Recommended password length
   * Additional ML-generated suggestions

All outputs are generated **only using trained machine learning models**.

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Flask
* Joblib
* TF-IDF Vectorization
* GitHub Codespaces

