# Myntra-Online-Store-Database-Q-A with LLM project based on Google Palm and Langchain
This project is an intelligent system designed to interact with a MySQL database for Myntra's online cloth store. Users can ask questions in natural language, and the system translates these questions into SQL queries, executes them against the MySQL database, and returns accurate answers. The system aims to simplify data retrieval and enhance decision-making for store managers and staff.
![image](https://github.com/user-attachments/assets/3a85d672-4457-4864-8bb5-f42de26c1117)

## Aim
The project's objective is to simplify data retrieval and enhance decision-making for store managers and staff through the use of a Streamlit app.

## Dataset
Utilized Kaggle's [ Myntra Fashion Dataset](https://github.com/user-attachments/assets/3a85d672-4457-4864-8bb5-f42de26c1117)

## Project Structure

- **`main.py`**: The main script for the Streamlit application that serves as the user interface for interacting with the system.
- **`langchain_helper.py`**: Contains the Langchain code to integrate with Google Generative AI and perform SQL query generation and execution.
- **`requirements.txt`**: A list of required Python packages for the project.
- **`few_shots.py`**: Contains few shot prompts
- **`.env`**: Configuration file for storing your Google API key.
  
