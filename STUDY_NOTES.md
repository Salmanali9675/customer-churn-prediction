# ?? Machine Learning & Customer Churn ? Complete Study Notes
*(Asaan Roman Urdu Mein Complete Revision Guide)*

---

## ?? Hissa 1: Bunyadi ML Concepts

### 1. Customer Churn Kya Hota Hai?
* **Churn** ka matlab hai: **"Chhor kar chale jana"** ya subscription cancel kar dena.
* *Example:* Jazz/Zong ki sim band karke kisi doosri company pe chale jana.
* **Company ko kyun fikar hoti hai?** Naya customer dhoond kar lana, purane customer ko rokay rakhne se **5 se 7 guna zyada mehnga** hota hai.

### 2. Features ($X$) vs Target ($y$)
* **Features ($X$):** Woh saari input maloomat jo hum computer ko dete hain taake woh andaza lagaye.
  * *Example:* Customer ki umar, mahana bill, kitne maheenay se sath hai (Tenure), contract type.
* **Target ($y$):** Woh aakhri nateeja jo model ko guess karna hota hai.
  * *Example:* `Churn` (Kya banda chhor kar gaya? 1 ya 0).
* **Misaal (Sebon Ki Tokri):** Tokri mein 10 seb hain. 1 Laal seb hamara Target ($y$) hai, baqi 9 Sabz seb hamare Features ($X$) hain!

### 3. Classification vs Regression
* **Regression:** Jab hum koi continuous number predict karte hain (jaise ghar ki qeemat ya kal ka temperature).
* **Classification:** Jab hum kisi cheez ko categories/groups mein baant te hain (jaise: Churn: Yes/No, ya Email: Spam/Not Spam).
* Hamara project ek **Classification Problem** hai!

---

## ?? Hissa 2: Python Aur Data Ki Safaai (Pandas & Data Cleaning)

### 1. Zaroori Commands
* `pd.read_csv('file.csv')`: CSV file ko parh kar DataFrame (Table) banata hai.
* `df.shape`: Batata hai kitni **Rows (Customers)** aur kitne **Columns (Details)** hain.
* `df.head()`: Table ki pehli **5 rows** screen par dikhata hai.
* `df.isnull().sum()`: Har column mein **khali dabbe (Missing Values)** ginta hai.
* `df.dropna(inplace=True)`: Missing values wali rows ko delete karta hai.

### 2. Data Types (`dtypes`)
* **Numeric (`int`, `float`):** Numbers (jaise Age = 25, Bill = $65.5).
* **Text (`object` ya `str`):** Alfaz (jaise "Male", "Month-to-month").
* **Asli Detective Masla (TotalCharges Case):**
  * Data enter karne walon ne naye customers ke bill mein space `" "` daba diya tha.
  * Is wajah se computer numbers ko text samajh raha tha.
  * Humne `pd.to_numeric(df['TotalCharges'], errors='coerce')` use karke space ko NaN banaya aur drop kiya.

---

## ?? Hissa 3: Exploratory Data Analysis (EDA)

### 1. Class Imbalance
* Hamare data mein:
  * **73%** customers rukay huay thay (Stay - 0).
  * **27%** customers chhor kar gaye thay (Churn - 1).
* Jab ek class bohot zyada ho aur doosri kam, isay **Class Imbalance** kehte hain.

### 2. The Accuracy Paradox (Dhokaybaaz Accuracy)
* Agar koi aalsi model har customer ko andha ho kar keh de: *"Koi nahi jayega (No)"*, tab bhi uski accuracy **73%** aayegi!
* Lekin woh un 27% bhaagne walon ko pakar nahi sakega.
* **Sabaq:** Churn jaise projects mein sirf Accuracy dekhna dhoka ho sakta hai!

### 3. Wajah Jo Data Ne Batayi:
* **Contract:** Month-to-month walay **42% log bhaag rahe thay**, jabke 2-year contract walay sirf **3%**.
* **Monthly Charges:** Churn karne walon ka bill (\$74) aam logon (\$61) se zyada tha.

---

## ?? Hissa 4: Data Preprocessing

1. **Text to Numbers (Encoding):**
   * Computer sirf math/binary samajhta hai.
   * `df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})`
   * `pd.get_dummies()`: Har category ka alag column (0 aur 1) bana deta hai (One-Hot Encoding).
2. **Fuzool Columns Hatana:**
   * `customerID` kisi kaam ka nahi tha, is liye `df.drop('customerID', axis=1)` kiya. (`axis=1` = Column).
3. **Feature Scaling (`StandardScaler`):**
   * Tenure 0 se 72 tak tha, aur TotalCharges $20 se $8000 tak!
   * Scaling saare features ko ek barabar tarazu (scale) par le aati hai taake bara number chhote number par zulm na kare.

---

## ?? Hissa 5: Train / Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
* **80% Data (Train):** Model ko sikhane ke liye (Homework/Practice).
* **20% Data (Test):** Model ka imtehan lene ke liye (Unseen Final Exam).
* **Overfitting:** Jab model sirf ratta maar le lekin naye imtehan mein fail ho jaye.
* **Data Leakage:** Jab exam ka paper pehle hi practice mein leak ho jaye!

---

## ?? Hissa 6: Machine Learning Models

Har model ke 2 jaduai functions:
* **`.fit(X_train, y_train)`:** Model ko parhana (Train karna).
* **`.predict(X_test)`:** Model se imtehan lena (Prediction karwana).

### Models Ka Taaruf:
1. **Logistic Regression:**
   * Probability (0% se 100% chance) batata hai.
   * 50% se zyada chance ho toh Churn (1), warna Stay (0).
2. **Decision Tree:**
   * If-Else ke flowchart ki tarah faisla karta hai. (Asaan hai par ratta bohot maarta hai).
3. **Random Forest:**
   * 100 Decision Trees ki Panchayat (Panel of judges)! Majority voting se faisla karta hai.

---

## ?? Hissa 7: Model Evaluation (Metrics)

### 1. Confusion Matrix (2x2 Scoreboard)
* **True Positive (TP):** Banda chhor kar ja raha tha aur model ne bilkul sahi pakra! ??
* **True Negative (TN):** Banda rukne wala tha aur model ne sahi kaha ke rukay ga.
* **False Positive (FP):** Jhoota shak! Kaha jayega par nahi gaya.
* **False Negative (FN) ??:** **Sab se khatarnak ghalti!** Model ne kaha nahi jayega, par banda chhor kar bhaag gaya!

### 2. Evaluation Metrics:
* **Recall:** Bhaagne walon ko pakadne ki taqat (Asal churners mein se kitno ko pakra). Churn mein Recall sab se ahem hai!
* **Precision:** Model ke shak mein kitni sachaai thi.
* **F1-Score:** Precision aur Recall ka darmiana (harmonic mean) score.
* **ROC-AUC:** Model ki farq karne ki taqat (0.5 = tukka, 1.0 = perfect).

### Kyun Logistic Regression Jeeta?
Saboot ne bataya ke Logistic Regression ne Recall (57%) aur F1-Score (0.61) mein Random Forest ko peechay chhor diya, is liye humne isay final model banaya.

---

## ?? Hissa 8: Model Saving & Deployment

* **Model Save:** `joblib.dump(bundle, 'model/churn_model.pkl')`
  * Model ko bar bar train karne ki zaroorat nahi parti.
* **Streamlit App (`app.py`):**
  * Frontend UI jahan user details enter karta hai.
  * Model background mein calculation karke live result dikhata hai: **High Churn Risk** ya **Likely to Stay**!

---

## ?? Hissa 9: Professional Folder Structure

```text
customer-churn-prediction/
?
??? data/              # telco_churn.csv
??? notebooks/         # EDA aur experiments
??? src/               # train.py
??? model/             # churn_model.pkl
??? app.py             # Streamlit web app
??? requirements.txt   # Libraries list
??? README.md          # Project front cover
??? STUDY_NOTES.md     # Yeh complete guide!
```

---
*Created with ?? by Antigravity for your Learning & Career Portfolio.*
