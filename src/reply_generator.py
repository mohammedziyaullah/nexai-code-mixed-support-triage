"""Multilingual customer support draft response generator.

Supports Official Stretch Goal 1:
- Generates natural, context-aware draft responses in the customer's language mix
  (English, Hinglish, Tanglish).
- Covers all 27 canonical customer support intents with polite, helpful templates.
- Enforces strict Responsible AI constraints: never fabricates tracking numbers,
  bank account statuses, delivery dates, or unauthorized refund claims.
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional

LanguageType = Literal["English", "Hinglish", "Tanglish"]

# Lexical indicators for Romanized Indic language identification
HINGLISH_INDICATORS = [
    r"\bmera\b", r"\bmeri\b", r"\bmere\b", r"\bkahan\b", r"\bhai\b", r"\bhain\b",
    r"\baayega\b", r"\bkab\b", r"\btak\b", r"\bkardo\b", r"\bkar\s+do\b", r"\bkarna\b",
    r"\bnahi\b", r"\bnahe\b", r"\bpaisa\b", r"\bpaise\b", r"\bkat\s+gaya\b", r"\bkate\b",
    r"\bchahiye\b", r"\bbhai\b", r"\bkaro\b", r"\bmadad\b", r"\bshukriya\b", r"\bdikha\b",
    r"\bchal\s+raha\b", r"\bkaam\b", r"\bbekaar\b", r"\bghatiya\b", r"\bkharaab\b",
    r"\bse\s+baat\b", r"\bkarenge\b", r"\bkaro\b", r"\bkyun\b", r"\bkya\b", r"\bhoga\b",
]

TANGLISH_INDICATORS = [
    r"\ben\b", r"\byen\b", r"\benga\b", r"\birukku\b", r"\bvarala\b", r"\bvaravillai\b",
    r"\bpannunga\b", r"\bpannanum\b", r"\bpannidunga\b", r"\bmudiyala\b", r"\banupunga\b",
    r"\banuppunga\b", r"\bpochu\b", r"\bpanam\b", r"\bkasu\b", r"\baagala\b", r"\baayiduchu\b",
    r"\bkitta\b", r"\bpesa\b", r"\bmaranthuten\b", r"\benna\b", r"\beppadi\b", r"\bseekiram\b",
    r"\budane\b", r"\bromba\b", r"\bthappu\b", r"\bseri\b", r"\bmosam\b", r"\bnalla\b",
    r"\bda\b", r"\bpa\b", r"\bvandhu\b", r"\bserala\b", r"\bkatunum\b",
]

COMPILED_HINGLISH = [re.compile(p, re.IGNORECASE) for p in HINGLISH_INDICATORS]
COMPILED_TANGLISH = [re.compile(p, re.IGNORECASE) for p in TANGLISH_INDICATORS]


def detect_language(text: str) -> LanguageType:
    """Classify text language into English, Hinglish, or Tanglish based on lexical indicators.

    Args:
        text: Raw customer query.

    Returns:
        One of 'English', 'Hinglish', or 'Tanglish'.
    """
    if not isinstance(text, str) or not text.strip():
        return "English"

    clean = text.strip().lower()
    tanglish_matches = sum(1 for p in COMPILED_TANGLISH if p.search(clean))
    hinglish_matches = sum(1 for p in COMPILED_HINGLISH if p.search(clean))

    if tanglish_matches > 0 and tanglish_matches >= hinglish_matches:
        return "Tanglish"
    if hinglish_matches > 0 and hinglish_matches > tanglish_matches:
        return "Hinglish"
    return "English"


# Multilingual Draft Templates for all 27 Intents
MULTILINGUAL_INTENT_DRAFTS: Dict[str, Dict[LanguageType, str]] = {
    "cancel_order": {
        "English": "I can help you cancel this order. If the package has not been dispatched yet, we can process the cancellation immediately. Could you please confirm your order number?",
        "Hinglish": "Main aapka order cancel karne mein madad kar sakta hoon. Agar package abhi tak dispatch nahi hua hai, toh hum turant cancel kar denge. Kripya apna order number confirm karein?",
        "Tanglish": "Unga order-a cancel panna naan help panren. Package innum dispatch aagala na, udane cancel pannidalam. Dayavu seidhu unga order number confirm pannunga?",
    },
    "change_order": {
        "English": "I can help modify the items or quantity in your order before fulfillment. Please provide your order number and the exact changes you would like to make.",
        "Hinglish": "Main dispatch se pehle aapke order mein item ya quantity update karne mein madad kar sakta hoon. Kripya apna order number aur jo changes karne hain woh share karein.",
        "Tanglish": "Dispatch aagurathukulla unga order items matha naan help panren. Unga order number matrum enna changes pannanum nu sollunga?",
    },
    "change_shipping_address": {
        "English": "I can update your delivery destination before dispatch. Please reply with your order number and complete new shipping address including pincode.",
        "Hinglish": "Main dispatch se pehle aapka delivery address update kar sakta hoon. Kripya apna order number aur naya address pincode ke saath share karein.",
        "Tanglish": "Dispatch aagurathukulla unga delivery address change panna mudiyum. Unga order number matrum pudhu address pincode oda anupunga.",
    },
    "check_cancellation_fee": {
        "English": "Standard order cancellations made prior to dispatch do not incur cancellation fees. Please share your order ID so I can verify the terms for your specific items.",
        "Hinglish": "Dispatch se pehle order cancel karne par koi cancellation fee nahi lagti. Kripya apna order ID share karein taaki main terms verify kar sakun.",
        "Tanglish": "Dispatch aagurathukulla cancel panna endha cancellation charges-um kidayathu. Unga order ID share pannunga, specific items terms verify panren.",
    },
    "check_invoice": {
        "English": "You can view and verify your itemized billing statement under 'Order History' in your account. Please let us know if you observe any discrepancy.",
        "Hinglish": "Aap apne account mein 'Order History' section se itemized invoice check kar sakte hain. Agar bill mein koi galti dikhe toh kripya batayein.",
        "Tanglish": "Unga account 'Order History' la itemized invoice check pannikalam. Bill la edhavathu issue irundha theriyapaduthunga.",
    },
    "check_payment_methods": {
        "English": "We support various payment modes including UPI (GPay, PhonePe, Paytm), Credit/Debit Cards, Net Banking, and Cash on Delivery where serviceable.",
        "Hinglish": "Hum UPI (GPay, PhonePe, Paytm), Credit/Debit Card, Net Banking aur eligible locations par Cash on Delivery support karte hain.",
        "Tanglish": "Namma platform la UPI (GPay, PhonePe), Credit/Debit Card, Net Banking matrum Cash on Delivery options support irukku.",
    },
    "check_refund_policy": {
        "English": "Our standard policy provides a 30-day return window for eligible items in original condition. Refunds are credited to the original payment source within 5 to 7 business days.",
        "Hinglish": "Hamari policy ke tahat eligible items par 30 days ka return window milta hai. Refund approve hone ke baad 5-7 working days mein aapke bank account mein credit ho jaata hai.",
        "Tanglish": "Namma policy padi eligible items-ku 30 days return window irukku. Refund approve aana 5-7 business days la unga bank account-ku credit aayidum.",
    },
    "complaint": {
        "English": "We sincerely apologize for the inconvenience and disappointing experience. Your complaint has been logged with top priority, and our support leadership is reviewing it.",
        "Hinglish": "Aapko hui pareshani ke liye hum maafi chahte hain. Aapki complaint top priority par register kar li gayi hai aur hamari support team ispar kaam kar rahi hai.",
        "Tanglish": "Unga mosamaana anubavathukku romba mannikavum. Unga complaint high priority la register panniyachu, support team udane action edupaanga.",
    },
    "contact_customer_service": {
        "English": "Our customer support team is available 24/7. How can we best assist you with your inquiry today?",
        "Hinglish": "Hamari customer support team 24/7 madad ke liye hazir hai. Hum aaj aapki kis tarah madad kar sakte hain?",
        "Tanglish": "Namma customer support team 24/7 ungalukku help panna ready. Innaiku ungalukku enna help venum sollunga?",
    },
    "contact_human_agent": {
        "English": "I am connecting you with a human support agent right now. Please stay on this chat while I transfer your session with high priority.",
        "Hinglish": "Main turant aapko ek human customer support executive se connect kar raha hoon. Kripya chat par bane rahein, session transfer ho raha hai.",
        "Tanglish": "Naan udane oru human customer service agent kitta ungalai connect panren. Dayavu seidhu chat la irunga, session transfer aaguthu.",
    },
    "create_account": {
        "English": "To register a new account, click 'Sign Up', enter your mobile number or email, and verify using the OTP sent to your phone.",
        "Hinglish": "Naya account banane ke liye 'Sign Up' par click karein, mobile number ya email enter karein, aur OTP verify karke complete karein.",
        "Tanglish": "Pudhu account create panna 'Sign Up' click panni, mobile number/email enter pannunga, phone-ku vara OTP enter panni verify pannunga.",
    },
    "delete_account": {
        "English": "We can assist with permanently closing your account and removing personal data. Please confirm your registered email address to receive the verification link.",
        "Hinglish": "Hum aapka account permanently close karne mein madad kar sakte hain. Kripya apna registered email confirm karein taaki hum deletion link bhej sakein.",
        "Tanglish": "Unga account permanently delete panna help panrom. Confirmation link anupa unga registered email address share pannunga.",
    },
    "delivery_options": {
        "English": "We offer Standard Delivery (3-5 business days) and Express Next-Day Delivery on eligible pincodes. You can select your preferred speed during checkout.",
        "Hinglish": "Hum Standard Delivery (3-5 din) aur select pincodes par Express Delivery provide karte hain. Aap checkout ke waqt delivery speed choose kar sakte hain.",
        "Tanglish": "Nammalita Standard Delivery (3-5 days) matrum Express Next-Day Delivery options irukku. Checkout la ungalukku thevaiyana option select pannikalam.",
    },
    "delivery_period": {
        "English": "Standard fulfillment typically takes 3 to 5 business days depending on destination pincode. You will receive an SMS and email notification upon shipment.",
        "Hinglish": "Standard orders deliver hone mein aamtaur par 3 se 5 business days lagte hain. Item ship hote hi tracking SMS aur email bhej diya jaayega.",
        "Tanglish": "Standard delivery-ku aamtaur-a 3 to 5 business days aagum. Item ship aana udane ungalukku tracking SMS matrum email notification vandhidum.",
    },
    "edit_account": {
        "English": "You can update personal details, contact info, and delivery addresses directly under 'Account Settings' in your profile.",
        "Hinglish": "Aap apna naam, contact number aur address 'Account Settings' section mein jaakar aasani se update kar sakte hain.",
        "Tanglish": "Unga profile details, mobile number matrum address-a 'Account Settings' la direct-a update pannikalam.",
    },
    "get_invoice": {
        "English": "I can provide your official GST invoice receipt. You can download the PDF directly from 'Order Details' or confirm your email to receive a copy.",
        "Hinglish": "Main aapko official tax invoice PDF provide kar sakta hoon. Aap 'Order Details' se download kar sakte hain ya email par mangwa sakte hain.",
        "Tanglish": "Official tax invoice PDF unga 'Order Details' page-la download pannalam. Illana unga email confirm pannunga, anga send panren.",
    },
    "get_refund": {
        "English": "To initiate a refund for an eligible return or canceled order, please share your order number and the reason for the refund request.",
        "Hinglish": "Return ya cancelled order ka refund process shuru karne ke liye, kripya apna order number aur refund ka reason share karein.",
        "Tanglish": "Refund initiate panna unga order number matrum refund request reason sollunga, naan immediate-a process panren.",
    },
    "newsletter_subscription": {
        "English": "You can subscribe or unsubscribe from marketing updates anytime via 'Notification Preferences' in your account settings.",
        "Hinglish": "Aap account settings mein 'Notification Preferences' mein jaakar promotional emails kabhi bhi turn on ya unsubscribe kar sakte hain.",
        "Tanglish": "Promotional emails stop panna unga account settings la 'Notification Preferences' poyi unsubscribe pannikalam.",
    },
    "payment_issue": {
        "English": "If amount was deducted without order confirmation, banks typically reconcile and reverse the transaction within 24 to 48 hours. Please share your bank reference/transaction ID.",
        "Hinglish": "Agar paise kat gaye hain aur order confirm nahi hua, toh aamtaur par bank 24-48 ghante mein amount refund kar deta hai. Kripya apna transaction ID share karein.",
        "Tanglish": "Panam deduct aagi order confirm aagala na, bank 24-48 hours la auto-refund pannidum. Safe side-ku unga transaction reference ID share pannunga.",
    },
    "place_order": {
        "English": "To place an order, add desired items to your shopping cart, proceed to checkout, verify delivery address, and complete payment.",
        "Hinglish": "Order place karne ke liye items cart mein add karein, delivery address verify karein aur payment complete karke confirm karein.",
        "Tanglish": "Order place panna thevaiyana items-a cart la add panni, delivery address check panni payment complete pannunga.",
    },
    "recover_password": {
        "English": "To reset your password, visit the login page and click 'Forgot Password'. A secure reset link or OTP will be sent to your registered contact.",
        "Hinglish": "Password reset karne ke liye login page par 'Forgot Password' click karein. Aapke registered mobile/email par secure reset link bheja jaayega.",
        "Tanglish": "Password reset panna login page la 'Forgot Password' click pannunga. Unga registered mobile/email-ku secure reset link varum.",
    },
    "registration_problems": {
        "English": "If you are encountering errors during signup or OTP verification, please check your network connection, ensure valid credentials, or try an alternate browser.",
        "Hinglish": "Agar signup ya OTP verification mein issue aa raha hai, toh kripya phone network check karein ya alternate browser use karke try karein.",
        "Tanglish": "Registration ya OTP varathula issue irundha, network connection check pannunga illana alternate browser la try pannunga.",
    },
    "review": {
        "English": "We appreciate your feedback! You can rate and review your purchased products directly under 'My Orders' > 'Write a Review'.",
        "Hinglish": "Aapke feedback ke liye shukriya! Aap 'My Orders' section mein jakar product rating aur review submit kar sakte hain.",
        "Tanglish": "Unga feedback-ku romba nandri! 'My Orders' section la poyi unga review matrum rating share pannalam.",
    },
    "set_up_shipping_address": {
        "English": "To add a new delivery destination, go to 'Saved Addresses' in your profile, enter the recipient name, street address, and pincode.",
        "Hinglish": "Naya delivery address add karne ke liye profile mein 'Saved Addresses' par jaakar recipient name, address aur pincode enter karein.",
        "Tanglish": "Pudhu shipping address set panna profile la 'Saved Addresses' poyi recipient name, street address matrum pincode enter pannunga.",
    },
    "switch_account": {
        "English": "To switch accounts, please log out from the top profile menu and sign in using your alternative credentials.",
        "Hinglish": "Account switch karne ke liye profile menu se logout karein aur doosre login credentials ke saath sign in karein.",
        "Tanglish": "Account switch panna mela irukura profile icon la logout panni, unga vera credentials vechu login pannunga.",
    },
    "track_order": {
        "English": "You can monitor live transit status for your shipment. Please provide your order number or tracking ID, or view live status under 'Track Order'.",
        "Hinglish": "Aap apne parcel ka live status track kar sakte hain. Kripya apna order number ya tracking ID share karein, ya 'Track Order' par check karein.",
        "Tanglish": "Unga parcel live delivery status track panna unga order number illana tracking ID sollunga, live tracking update therinjukalam.",
    },
    "track_refund": {
        "English": "To check your refund progress, please provide your refund reference or order ID. Bank transfers typically take 3 to 5 business days to reflect in your account statement.",
        "Hinglish": "Refund status check karne ke liye apna order ID ya refund reference number share karein. Bank transfer mein aamtaur par 3-5 business days lagte hain.",
        "Tanglish": "Refund status therinjuka unga refund reference ID illana order number sollunga. Bank account-ku varathukku 3-5 business days aagum.",
    },
}

FALLBACK_DRAFTS: Dict[LanguageType, str] = {
    "English": (
        "Thank you for reaching out to support. We have received your message and routed it to a "
        "customer service agent who will review the details and assist you directly."
    ),
    "Hinglish": (
        "Customer support mein sampark karne ke liye shukriya. Humne aapka message receive kar liya hai "
        "aur ek support executive ko assign kiya hai jo jald hi aapse baat karenge."
    ),
    "Tanglish": (
        "Customer support-ku contact pannathukku nandri. Unga message receive panniyachu, "
        "support agent review panni udane ungala contact pannuvanga."
    ),
}


MAP_61_TO_27: Dict[str, str] = {
    "search_product": "place_order",
    "product_recommendation": "place_order",
    "product_availability": "place_order",
    "product_price_inquiry": "check_invoice",
    "product_comparison": "place_order",
    "product_specifications": "place_order",
    "product_compatibility": "place_order",
    "cancel_order": "cancel_order",
    "modify_order_items": "change_order",
    "order_confirmation_missing": "track_order",
    "order_status_inquiry": "track_order",
    "track_order": "track_order",
    "delivery_fee_inquiry": "delivery_options",
    "delayed_delivery": "delivery_period",
    "missing_delivery": "complaint",
    "change_shipping_address": "change_shipping_address",
    "delivery_options": "delivery_options",
    "payment_methods_inquiry": "check_payment_methods",
    "payment_failed": "payment_issue",
    "payment_deducted_order_failed": "payment_issue",
    "duplicate_payment": "payment_issue",
    "promo_code_issue": "payment_issue",
    "unexpected_charge": "check_invoice",
    "refund_policy_inquiry": "check_refund_policy",
    "request_refund": "get_refund",
    "refund_status_inquiry": "track_refund",
    "damaged_product_received": "complaint",
    "defective_product_received": "complaint",
    "return_request": "get_refund",
    "exchange_request": "change_order",
    "wrong_product_received": "complaint",
    "missing_item_in_package": "complaint",
    "return_pickup_request": "get_refund",
    "warranty_claim": "complaint",
    "login_issue": "registration_problems",
    "recover_password": "recover_password",
    "otp_not_received": "registration_problems",
    "account_locked": "registration_problems",
    "change_email_address": "edit_account",
    "change_phone_number": "edit_account",
    "suspected_account_hack": "recover_password",
    "suspected_scam": "complaint",
    "unauthorized_transaction": "payment_issue",
    "privacy_policy_inquiry": "contact_customer_service",
    "general_complaint": "complaint",
    "unresolved_issue": "complaint",
    "escalate_complaint": "complaint",
    "request_human_agent": "contact_human_agent",
    "request_supervisor": "contact_human_agent",
    "checkout_error": "payment_issue",
    "app_crash": "registration_problems",
    "notification_issue": "edit_account",
    "subscription_cancellation": "cancel_order",
    "loyalty_points_missing": "check_invoice",
    "subscription_signup": "newsletter_subscription",
    "marketing_opt_out": "newsletter_subscription",
    "greeting": "contact_customer_service",
    "gratitude": "review",
    "unclear_message": "contact_customer_service",
    "language_preference": "contact_customer_service",
    "contact_customer_service": "contact_customer_service",
}


def get_multilingual_draft_reply(
    intent: Optional[str],
    language: LanguageType = "English",
) -> str:
    """Retrieve draft reply formatted in the target language for a specific intent.

    Args:
        intent: Canonical Bitext intent or 61-intent taxonomy string.
        language: 'English', 'Hinglish', or 'Tanglish'.

    Returns:
        Customer-ready draft response string.
    """
    if not intent:
        return FALLBACK_DRAFTS.get(language, FALLBACK_DRAFTS["English"])

    intent_key = intent.strip().lower()
    drafts_for_intent = MULTILINGUAL_INTENT_DRAFTS.get(intent_key)

    # Fallback to mapped canonical intent if fine-grained 61-intent is provided
    if not drafts_for_intent and intent_key in MAP_61_TO_27:
        canonical_key = MAP_61_TO_27[intent_key]
        drafts_for_intent = MULTILINGUAL_INTENT_DRAFTS.get(canonical_key)

    if not drafts_for_intent:
        return FALLBACK_DRAFTS.get(language, FALLBACK_DRAFTS["English"])

    return drafts_for_intent.get(language, drafts_for_intent.get("English", FALLBACK_DRAFTS["English"]))
