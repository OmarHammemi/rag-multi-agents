### **Building a RAG Bot for Specialized Question-Answering**

#### **Problem Statement:**

You are tasked with creating a **Retrieval-Augmented Generation (RAG) Bot** that can answer questions based on a set of predefined data dumps. Your system will leverage **specialized agents** to handle different types of questions, retrieve relevant data from the appropriate sources, and generate accurate answers.

### **Data Dump:**

You are provided with two data dumps that are imaginary and created by us with the following categories:

1. **Car Data** (information about car models, brands, specifications, etc.)  \[Artificially curated\] \[Structured data\]  
2. **Country Data** (information about countries, capitals, population, language, etc.) \[Artificially curated\]\[Unstructured data\]

### **Task:**

You Can complete this task on a Jupyter NoteBook

* **Data Chunking & Storage**: You will need to chunk and organize the provided data dumps into structured formats that can be easily retrieved by your bot.  
* **Specialized Agents**: Create specialized **RAG agents** that can:  
  * Retrieve information from the **Car Data** dump when asked questions related to cars.  
  * Retrieve information from the **Country Data** dump when asked about countries.  
  * Retrieve or compute mathematical answers.  
* **Query Handling**: When a user asks a question, your system should route the query to the appropriate agent based on the nature of the question:  
  * **If the query is related to cars**: The Car Agent should fetch relevant car-related data and provide an answer.  
  * **If the query is related to countries**: The Country Agent should fetch relevant country-related data.  
  * **If the query involves solving a math problem**: The Mathematical Agent should compute and return the result. It needs to verify that the result is correct.   
  * **If the query falls outside these categories**: Return a message saying the system cannot provide an answer.

### **Requirements:**

* **Data Chunking & Storage**: You must chunk the data into manageable units for efficient retrieval and ensure it's stored in a way that allows fast querying.  
* **Retrieval-Augmented Generation (RAG)**: Use RAG to augment your system's ability to generate relevant responses based on the data.  
* **Routing & Query Handling**: Implement a system where the right agent is invoked based on the user's query type, ensuring that only relevant data is used to answer the question.  
* **Edge Cases & Error Handling**: Consider scenarios where data is missing or the query is outside the scope of the provided data dumps.
* **Use LangGraph for Agentic AI workflow**

### **Submission Instructions**

* Create your own Git branch in the format `solution/{firstname}-{lastname}`, e.g., `solution/peter-griffin`.
* We encourage you to commit often, but feel free to use whatever approach works best for you.
* Currently, the project repository has no remote setup. Please keep it that way—don’t publish or share it.
* Keep this `README` file for the instructions, and create a new `SUBMISSION.md` file to document any important decisions, trade-offs, or technical details that will help us understand your solution.
* At the end, compress the repository into a `.zip` file and upload it to the link we provided in the email. Make sure the `.zip` file contains hidden files and directories (e.g., `.git`, `.env`) but does not contain `.ipynb_checkpoints` or `__pycache__` folders.

