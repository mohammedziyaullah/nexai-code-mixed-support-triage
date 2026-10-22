# Intent Expansion Proposal for NexAI / CodeMix Support AI

**Document Version:** 1.0  
**Status:** PROPOSED & ISOLATED (Not added to the 27-intent production classifier)  
**Governance Invariant:** In accordance with Phase 4 hackathon rules, all proposed intents remain strictly documented in this separate specification. The 27 production intents are preserved without silent class-space modifications.

---

## 1. Executive Motivation

The existing 27-intent Bitext customer support taxonomy covers core e-commerce transactions (orders, cancellations, refunds, payments, accounts, invoices, complaints, and general support). However, Indian e-commerce, D2C, and consumer electronics ecosystems frequently encounter four distinct customer workflows that cannot be accurately classified into the existing 27 classes without causing semantic drift:

1. **`warranty_claim`** — Claiming or verifying manufacturer or extended warranty coverage.
2. **`track_repair_service`** — Tracking technician visits, device service requests, or repair progress.
3. **`store_locator`** — Inquiring about physical service centers, offline pickup kiosks, and experience hubs.
4. **`feedback_suggestion`** — Constructive platform improvement ideas and feature requests (non-complaint, non-product review).

---

## 2. Detailed Intent Specifications

### Intent 1: `warranty_claim`

- **Definition:** The customer is seeking to register, verify, or claim a product warranty against manufacturer defects after the standard 30-day return window.
- **Distinction from Existing Intents:**
  - *Differs from `check_refund_policy`:* Refund policy deals with merchant return windows (e.g., 30 days) and financial reimbursements; warranty deals with long-term brand coverage (e.g., 1-year or 2-year repairs/replacements).
  - *Differs from `complaint`:* `complaint` is an expression of dissatisfaction about poor service; `warranty_claim` is a standard post-sales entitlement workflow.
- **Ambiguity Notes:** If a customer mentions an item is broken AND asks for money back, the intent is `complaint` or `get_refund`. If they explicitly ask about warranty registration or claim procedures, it is `warranty_claim`.

#### Candidate Examples (10 Diverse Multilingual Cases)
1. **[English]** "How do I claim the 1-year manufacturer warranty for my boAt earphones?"
2. **[English]** "Is extended warranty covered if the screen cracks accidentally?"
3. **[English]** "Please share the warranty registration portal link for my microwave."
4. **[Hinglish]** "Laptop ka 1 saal ka warranty claim kaise register karein?"
5. **[Hinglish]** "Kya yeh product brand warranty ke under free me repair hoga?"
6. **[Hinglish]** "Warranty card par stamp nahi laga hai, claim reject toh nahi hoga?"
7. **[Tanglish]** "Indha mobile-ku 1 year warranty claim panna enna procedure?"
8. **[Tanglish]** "Brand warranty activate aagiducha nu eppadi check panradhu?"
9. **[Tanglish]** "Earphones sound work aagala, warranty claim raise panna mudiyuma?"
10. **[Hinglish]** "Bill copy kho gayi hai, kya serial number se warranty claim milega?"

---

### Intent 2: `track_repair_service`

- **Definition:** Monitoring the operational progress of an ongoing physical product repair, technician visit, or service center job sheet.
- **Distinction from Existing Intents:**
  - *Differs from `track_order`:* `track_order` strictly monitors warehouse logistics, courier transit, and parcel dispatch from seller to buyer. `track_repair_service` tracks technician inspection, part replacements, and turnaround times for owned items.
- **Ambiguity Notes:** Queries mentioning *"Where is my order / parcel?"* are `track_order`. Queries mentioning *"Technician kab aayega?"* or *"Repair job sheet status"* are `track_repair_service`.

#### Candidate Examples (10 Diverse Multilingual Cases)
1. **[English]** "When will the service technician visit my house for the washing machine repair?"
2. **[English]** "Can you check the current repair status of job sheet number JS-9821?"
3. **[English]** "Has the motherboard part arrived at the service center for my laptop?"
4. **[Hinglish]** "Technician visit kab schedule hua hai? Subah se wait kar raha hoon."
5. **[Hinglish]** "Service center me repair ke liye diya tha mobile, kab tak ready hoga?"
6. **[Hinglish]** "Job sheet number 8821 ka repair status check karke bataiye."
7. **[Tanglish]** "AC repair-ku technician eppo varuvaaru nu timing sollunga."
8. **[Tanglish]** "Service center-la kudutha laptop repair mudinjiducha nu theriyanum."
9. **[Tanglish]** "TV panel replacement spare part vandhurucha repair-ku?"
10. **[Hinglish]** "Inspection ke baad repair estimate kab share karenge?"

---

### Intent 3: `store_locator`

- **Definition:** Locating physical brick-and-mortar touchpoints, authorized service centers, experience retail stores, or self-service drop-off kiosks.
- **Distinction from Existing Intents:**
  - *Differs from `delivery_options`:* `delivery_options` refers to e-commerce shipping speeds (Standard, Express, COD). `store_locator` specifically asks for geographic directions, addresses, and operating hours of physical offline stores.
  - *Differs from `contact_customer_service`:* Focuses on physical locations rather than digital/phone contact channels.
- **Ambiguity Notes:** If the user asks for a helpline number, it is `contact_customer_service`. If they ask for the nearest physical center in their city, it is `store_locator`.

#### Candidate Examples (10 Diverse Multilingual Cases)
1. **[English]** "Where is the nearest authorized service center in Bangalore Koramangala?"
2. **[English]** "Do you have a physical offline experience store in Mumbai?"
3. **[English]** "What are the Sunday working hours for the Chennai service hub?"
4. **[Hinglish]** "Delhi me sabse paas wala service center kahan hai?"
5. **[Hinglish]** "Kya main directly store par jakar product check kar sakta hoon?"
6. **[Hinglish]** "Indore me authorized repair center ka exact address aur timing batao."
7. **[Tanglish]** "Chennai T. Nagar-la unga official store irukka, address sollunga."
8. **[Tanglish]** "Coimbatore-la authorized service center enga irukku nu share pannunga."
9. **[Tanglish]** "Direct-a store-la poyi return drop panna physical location irukka?"
10. **[Hinglish]** "Pincode 110001 ke paas kaunsa store location open hai?"

---

### Intent 4: `feedback_suggestion`

- **Definition:** Proactive, non-hostile recommendations, user interface ideas, or feature requests intended to improve the platform or product catalog.
- **Distinction from Existing Intents:**
  - *Differs from `complaint`:* `complaint` expresses dissatisfaction with a failed service, broken item, or delayed refund. `feedback_suggestion` is constructive and neutral.
  - *Differs from `review`:* `review` is evaluating a specific purchased item (ratings, star reviews, size fit). `feedback_suggestion` targets platform features (e.g. app UI, search filters, payment options).
- **Ambiguity Notes:** Harsh criticism without actionable feedback maps to `complaint`. Ideas framed as *"It would be great if your app had..."* map to `feedback_suggestion`.

#### Candidate Examples (10 Diverse Multilingual Cases)
1. **[English]** "It would be great if your mobile app supported Dark Mode in the next release."
2. **[English]** "Please consider adding regional language audio search on the home screen."
3. **[English]** "A price alert notification feature for wishlist items would be really helpful."
4. **[Hinglish]** "App me Tamil aur Hindi voice search ka option add karo toh accha hoga."
5. **[Hinglish]** "Wishlist ke items par price drop alert aana chahiye, feature request hai."
6. **[Hinglish]** "Checkout page par UPI pin verification fast karne ke liye ek suggestion hai."
7. **[Tanglish]** "App-la filter option innum clear-aa irundha searching romba easy-aa irukkum."
8. **[Tanglish]** "Delivery slot time select panra feature add panna nalla irukkum."
9. **[Tanglish]** "Product comparison table app-la kudutha choose panna convenient-aa irukkum."
10. **[Hinglish]** "Return pickup time slot choose karne ka option hona chahiye."

---

## 3. Governance Checklist for Future Integration

Before any of these proposed intents can be integrated into the production model:
- [ ] Annotate at least 500 verified real-world samples per proposed intent across English, Hinglish, and Tanglish.
- [ ] Retrain model on the expanded (31-intent) taxonomy and verify zero F1 degradation on the existing 27 core intents.
- [ ] Update Streamlit UI label mappings, suggested draft reply templates, and urgency heuristics.
- [ ] Update benchmark test suites and regression tests.
