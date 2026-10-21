# Data Directory

This directory is designated for managing local dataset caches and test evaluation datasets for the Code-Mixed Support Triage Assistant.

## Dataset Management Guidelines

1. **Do Not Commit Raw or Large Datasets Unnecessarily:**
   - Raw dataset files, downloaded corpora, and large intermediate caches must not be committed to the repository.
   - Any local data files downloaded during training or evaluation should be ignored by Git or stored outside version control.

2. **Official Bitext Dataset:**
   - The official Bitext Customer Support dataset (`bitext/Bitext-customer-support-llm-chatbot-training-dataset` or specified identifier on Hugging Face) must be loaded dynamically using the Hugging Face `datasets` library during later implementation steps rather than committed as static binary/CSV files.

3. **Team-Curated Test Set:**
   - In subsequent implementation phases, curated Tanglish and Hinglish test cases (minimum 30 team-written messages with ground truth labels) will be maintained in a structured format (e.g., CSV or JSON) for transparent evaluation.
