# this is My branchs Tree 
.
├── .env
├── .gitignore
├── README.md
├── SUBMISSION.md
├── agents
│   ├── .gitignore
│   ├── car_agent.py
│   ├── country_agent.py
│   └── math_agent.py
├── data
│   ├── car_metadata.json
│   └── country_metadata.json
├── index
│   ├── car_faiss.index
│   └── country_faiss.index
├── langgraph_pipeline
│   └── langgraph_runner.py
├── main.ipynb
├── requirements.txt
├── source_data
│   ├── production_data
│   │   ├── car_faiss.index
│   │   ├── car_metadata.json
│   │   ├── country_faiss.index
│   │   └── country_metadata.json
│   └── raw_data
│       ├── car_chunks.jsonl
│       ├── cars_dataset.csv
│       ├── country_chunks.jsonl
│       ├── country_data.md
│       └── data_cleaning.ipynb
└── utils
    ├── __init__.py
    └── router.py

# you can test with main.ipynb

# OpnAI 3.5 key needed to implement 

# Task and building Processing
# 1. Data analyzing and processing + data transformation to faiss.idex and json form 
# 2. Builded 3 Agents cars , Country , Math
# 3. Rooting
# 4. Building Langraph_pipeline
# 5. Testing and optimizing

