# Fraud Detection Dataset - Analysis & Explanation

## Dataset Overview

**File:** `fraud_detection_dataset_lstm_v3.csv`  
**Size:** 4.3+ Million transactions  
**Fraud Rate:** 0.98% (~42,000 fraud cases)  
**Features:** 17 columns

---

## Dataset Columns Explained

| # | Column | Type | Description | Example |
|---|--------|------|-------------|---------|
| 1 | `transaction_id` | String | Unique identifier for each transaction | `TXN_000000043` |
| 2 | `user_id` | String | Unique identifier for the account holder | `USER_000012` |
| 3 | `timestamp` | DateTime | When the transaction occurred | `2024-06-26 09:28:41` |
| 4 | `amount` | Float | Transaction amount in currency units | `92976.69` |
| 5 | `merchant_category` | String | Type of merchant (ATM, Restaurant, Electronics, etc.) | `Online Shopping` |
| 6 | `device_id` | String | Device used to make the transaction | `Mobile_Android` |
| 7 | `location` | String | Geographic location of transaction | `Chennai`, `Pune` |
| 8 | `time_gap_from_previous_hrs` | Float | Hours since user's last transaction | `0.7` hours |
| 9 | `amount_deviation_from_avg_pct` | Float | % difference from user's average transaction amount | `250.1%` (much higher) |
| 10 | `transactions_last_1hr` | Integer | Number of transactions in past hour | `1` (is this suspicious?) |
| 11 | `transactions_last_24hr` | Integer | Number of transactions in past 24 hours | `2`, `5` |
| 12 | `account_age_days` | Integer | How long the account has existed | `817` days |
| 13 | `avg_transaction_amount` | Float | User's typical transaction amount | `26559.93` |
| 14 | `transaction_frequency` | Integer | How many transactions user typically makes | `39` transactions |
| 15 | `most_used_device` | String | Device user usually transacts with | `Mobile_Android` |
| 16 | `most_used_location` | String | Location user usually transacts from | `Chennai` |
| 17 | `is_fraud` | Integer | Target label (0=Legitimate, 1=Fraud) | `1` = Fraud |

---

## Why Transactions Are Marked as FRAUD

Transactions are flagged as fraud based on **behavioral anomalies** - unusual patterns that deviate significantly from a user's normal activity. Here are the **key fraud indicators**:

### 1. **UNUSUALLY HIGH TRANSACTION AMOUNT** ⚠️⚠️⚠️ (STRONGEST SIGNAL)
- **Fraud transactions are 175% HIGHER than legitimate ones**
  - Fraud average: **49,208.92**
  - Legitimate average: **17,858.47**

- **Amount deviation from user average:**
  - Fraud: **159.9% above** user's typical amount (3,528% difference!)
  - Legit: **4.4% above** user's typical amount
  
**Why?** Fraudsters typically try to transfer large amounts quickly before detection.

---

### 2. **RAPID-FIRE TRANSACTIONS (Very Quick Succession)** ⚠️⚠️⚠️ (VERY STRONG)
- **Fraud has 92.9% SMALLER time gaps between transactions**
  - Fraud: **1.0 hour** average between transactions
  - Legit: **14.07 hours** average between transactions

**Real Example:** A fraud transaction occurs only **0.1-0.7 hours** after the previous one (6-42 minutes!)

**Why?** Fraudsters rush to make multiple unauthorized charges before the account holder notices.

---

### 3. **SUDDEN BURST OF TRANSACTIONS** ⚠️⚠️⚠️ (VERY STRONG)
- **Fraud has 751% MORE transactions in the last 1 hour**
  - Fraud: **0.61** transactions per hour
  - Legit: **0.07** transactions per hour

- **Fraud has 137% MORE transactions in the last 24 hours**
  - Fraud: **3.85** transactions per day
  - Legit: **1.62** transactions per day

**Why?** When a card is compromised, the fraudster attempts multiple quick purchases.

---

### 4. **DIFFERENT DEVICE USAGE** ⚠️⚠️ (MODERATE)
- **64.7% of fraud uses a DIFFERENT device** than the user's usual device
- **Only 10.7% of legitimate transactions** use a different device

**Real Example from Dataset:**
- User normally transacts via `Mobile_Android` → Fraud uses `Mobile_Web`
- This suggests **account compromise or unauthorized access**

---

### 5. **DIFFERENT LOCATION** ⚠️⚠️ (MODERATE)
- **71.7% of fraud occurs from DIFFERENT locations**
- **Only 17.2% of legitimate transactions** from different locations

**Real Example:**
- User normally transacts from `Chennai` → Fraud from `Pune` or other cities
- Suggests **account takeover or stolen credentials**

---

### 6. **SUSPICIOUS MERCHANT CATEGORY** ⚠️ (WEAK SIGNAL)
- **ATM Withdrawals dominate fraud: 37.4%** of all fraud
- Legitimate transactions are evenly distributed (~6-7% per category)

**Why?** Cash withdrawals are harder to trace and recover. Fraudsters target ATMs.

**Fraud Categories (Top):**
1. ATM Withdrawal - 37.4%
2. Travel - 10.2%
3. Electronics - 9.7%
4. Online Shopping - 9.0%

**Legitimate Categories (Evenly spread):**
- Pharmacy, Restaurant, Education, etc. - all ~6.7% each

---

## Real Example: Transaction #44 Marked as FRAUD

```
Amount:                    53,738.17 (avg was 28,104.50)
Deviation from average:    91.2% ABOVE normal
Time since last txn:       0.4 HOURS (24 minutes!)
Txns in last 1 hour:       1 (suspicious burst)
Txns in last 24 hours:     3 (higher than normal)
Category:                  Electronics (high-value item)
Device:                    DIFFERENT from usual
Location:                  DIFFERENT from usual
Status:                    ✓ MARKED AS FRAUD
```

**Why it's fraud:**
- Amount is nearly 2x the user's average
- Happened 24 minutes after previous transaction
- Using unfamiliar device & location
- Electronics purchases are common in fraud

---

## Statistical Summary: Fraud vs Legitimate

| Feature | Fraud | Legitimate | Difference |
|---------|-------|-----------|------------|
| **Amount** | 49,209 | 17,858 | **+175.5%** |
| **Time Gap (hours)** | 1.0 | 14.07 | **-92.9%** |
| **Amount Deviation %** | 159.9% | 4.4% | **+3,528%** |
| **Txns/1hr** | 0.61 | 0.07 | **+751%** |
| **Txns/24hr** | 3.85 | 1.62 | **+137%** |
| **Different Device** | 64.7% | 10.7% | **6.0x more** |
| **Different Location** | 71.7% | 17.2% | **4.2x more** |

---

## How the Model Detects Fraud

### **XGBoost Model** (Precision: 30.2%, Recall: 78.49%)
- Processes all 17 features simultaneously
- Learns feature importance automatically
- Top features: amount deviation, transaction frequency, time gaps

### **LSTM Model** (Precision: 89.9%, Recall: 95%+)
- Analyzes **sequence of transactions** (last 15 transactions)
- Captures **temporal patterns** (0.1-hour gaps are suspicious)
- Better at detecting fraud patterns that build up over time

### **Hybrid Ensemble** (Both XGBoost + LSTM)
- Combines both approaches
- Achieves better balance between precision and recall

---

## Why This Dataset is Realistic

1. **Subtle Fraud:** Only ~60% of fraud has elevated amounts (not all are obvious)
2. **Behavioral Patterns:** Fraud is defined by **unusual behavior**, not just amount
3. **Real-world Imbalance:** 0.98% fraud rate matches actual financial institutions
4. **Temporal Dimension:** Time gaps and burst patterns are critical signals
5. **Multi-device World:** People use phones, laptops, tablets - fraudsters just pick the wrong one

---

## Key Takeaway

**Fraud is detected by combining multiple weak signals:**

```
(High Amount) + (Quick Succession) + (Burst Traffic)
   + (Different Device) + (Different Location)
   = High Fraud Probability
```

No single feature is definitive, but **the combination is highly suspicious!**

---

## High-Confidence Fraud Cases (Compared to Past Transactions)

To make the explanation more convincing, these cases were selected using strict criteria:
- amount deviation > 150%
- time gap < 2 hours
- transactions in last 24h > 2x user normal daily rate
- device mismatch
- location mismatch

Each case below has **5/5 fraud flags** and is compared against that user's **last 5 legitimate transactions**.

### Case 1 — TXN_000457921 (Gas Station)
- User: USER_004579
- Fraud amount: 84,524.29 vs last-5 normal average: 19,767.69 (**4.28x higher**)
- Fraud time gap: 1.47h vs last-5 normal average gap: 14.00h (**9.5x faster**)
- Fraud 24h count: 2 vs last-5 normal average: 1.20 (**1.67x higher**)
- Device: Desktop_Windows vs usual Desktop_Mac (mismatch)
- Location: Kolkata vs usual Nagpur (mismatch)
- Why fraud: extreme amount jump + rushed timing + behavior shift in both device and location.

### Case 2 — TXN_000609122 (ATM Withdrawal)
- User: USER_006091
- Fraud amount: 77,275.83 vs last-5 normal average: 20,275.10 (**3.81x higher**)
- Fraud time gap: 0.13h (7.8 minutes) vs last-5 normal average gap: 21.46h (**165x faster**)
- Fraud 24h count: 2 vs last-5 normal average: 1.00 (**2.0x higher**)
- Device: Desktop_Windows vs usual Mobile_iOS (mismatch)
- Location: Jaipur vs usual Mumbai (mismatch)
- Why fraud: near-immediate repeat transaction + large amount spike + complete context switch.

### Case 3 — TXN_000306021 (Insurance)
- User: USER_003060
- Fraud amount: 115,677.55 vs last-5 normal average: 22,684.66 (**5.10x higher**)
- Fraud time gap: 1.48h vs last-5 normal average gap: 7.02h (**4.7x faster**)
- Fraud 24h count: 4 vs last-5 normal average: 2.80 (**1.43x higher**)
- Device: Tablet_Android vs usual Desktop_Windows (mismatch)
- Location: Hyderabad vs usual Indore (mismatch)
- Why fraud: very large amount escalation with compressed timing and identity-context mismatch.

### Case 4 — TXN_000609124 (Electronics)
- User: USER_006091
- Fraud amount: 90,498.73 vs last-5 normal average: 20,275.10 (**4.46x higher**)
- Fraud time gap: 1.94h vs last-5 normal average gap: 21.46h (**11.1x faster**)
- Fraud 24h count: 4 vs last-5 normal average: 1.00 (**4.0x higher**)
- Device: Desktop_Mac vs usual Mobile_iOS (mismatch)
- Location: Coimbatore vs usual Mumbai (mismatch)
- Why fraud: burst behavior and large value transaction from a new device and city.

### Case 5 — TXN_000100030 (Travel)
- User: USER_001000
- Fraud amount: 61,009.16 vs last-5 normal average: 13,976.41 (**4.37x higher**)
- Fraud time gap: 1.87h vs last-5 normal average gap: 11.55h (**6.2x faster**)
- Fraud 24h count: 4 vs last-5 normal average: 1.60 (**2.5x higher**)
- Device: Web_Browser vs usual Tablet_Android (mismatch)
- Location: Chandigarh vs usual Chennai (mismatch)
- Why fraud: amount and velocity sharply exceed personal baseline with full context mismatch.

### Case 6 — TXN_000471731 (Utilities)
- User: USER_004717
- Fraud amount: 63,950.60 vs last-5 normal average: 17,065.85 (**3.75x higher**)
- Fraud time gap: 1.68h vs last-5 normal average gap: 18.27h (**10.9x faster**)
- Fraud 24h count: 3 vs last-5 normal average: 1.60 (**1.88x higher**)
- Device: Tablet_Android vs usual Web_Browser (mismatch)
- Location: Pune vs usual Nagpur (mismatch)
- Why fraud: unusually high spend compressed in time, with abrupt device/location switch.

### Why these are convincing
- In all 6 cases, amount is ~3.75x to 5.10x the user's recent normal baseline.
- In all 6 cases, the transaction happens much faster than normal (from ~4.7x to 165x faster).
- All cases include device + location mismatch, which is rare in genuine continuity behavior.
- This is exactly the pattern fraud systems look for: **amount spike + velocity spike + identity/context shift**.
