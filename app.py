
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import pymysql

# Define the API key as a string
api_key = "your_api_key"

# Initialize the GoogleGenerativeAI LLM with the API key
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.1)

# Example usage of the LLM (adjust based on your actual use case)
# response = llm("Write a poem about dogs")
# print(response)

# Define your MySQL connection details
from langchain_community.utilities import SQLDatabase

db_user = "root"
db_password = "root"
db_host = "localhost"
db_name = "myntra"

# Create a connection string to SQL Database
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
#print(db.table_info)

from langchain_experimental.sql import SQLDatabaseChain

# Initialize SQLDatabaseChain with named arguments if needed
db_chain = SQLDatabaseChain.from_llm( llm, db, verbose=True )

#qstn1 = db_chain("What is the discount percentage on the Roadster Men Navy Blue Slim Fit Mid Rise Clean Look Jeans?")
#print(qstn1)

#Few Shot Learning
few_shots = [
{'Question' : "What is the best deal you have on jeans for men?",
'SQLQuery' : "SELECT myntra.Description, discount.DiscountPrice_in_Rs, discount.DiscountOffer FROM myntra JOIN discount ON myntra.Product_id = discount.Product_id WHERE myntra.category_by_Gender = 'Men' AND myntra.Individual_category = 'jeans' ORDER BY discount.DiscountPrice_in_Rs LIMIT 1",
'SQLResult': "Result of the SQL query",
'Answer' : "herenow men blue slim fit mid rise clean look ankle length stretchable jeans"},
{'Question': "How can I find out more about a specific product?",
'SQLQuery':"SELECT * FROM myntra WHERE Product_id = 2296012",
'SQLResult': "Result of the SQL query",
'Answer': "https://www.myntra.com/jeans/roadster/roadster-men-navy-blue-slim-fit-mid-rise-clean-look-jeans/2296012/buy , Roadster , Bottom Wear , jeans , Men , roadster men navy blue slim fit mid rise clean look jeans , 1499.00 , 28, 30, 32, 34, 36 , 3.9 , 999"},
{'Question': "What are the available sizes for the Roadster men’s shirt?" ,
'SQLQuery' : "SELECT SizeOption FROM myntra WHERE Product_id = 11895958",
'SQLResult': "Result of the SQL query",
'Answer': "38, 40, 42, 44, 46, 48"} ,
{'Question' : "What is the discount percentage on the Roadster Men Navy Blue Slim Fit Mid Rise Clean Look Jeans?" ,
'SQLQuery': "SELECT DiscountOffer FROM discount WHERE Product_id = 2296012",
'SQLResult': "Result of the SQL query",
'Answer' : "45% OFF"}
]

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#e = embeddings.embed_query("What is the discount percentage on the Roadster Men Navy Blue Slim Fit Mid Rise Clean Look Jeans?")
#print(e[:5])

to_vectorize = [" ".join(example.values()) for example in few_shots]
#print(to_vectorize)

from langchain.vectorstores import Chroma
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

from langchain.prompts import SemanticSimilarityExampleSelector
example_selector = SemanticSimilarityExampleSelector(
vectorstore=vectorstore,
k=2,
)

selected_examples = example_selector.select_examples({"Question": "What are the available sizes for the Roadster men’s shirt?"})

# Print the selected examples for debugging
#print(selected_examples)

# Execute a sample query using the selected examples
response = db_chain(selected_examples)
#print(response)

### my sql based instruction prompt
mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""

from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
#print(_mysql_prompt)

#print(PROMPT_SUFFIX)

from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)

from langchain.prompts import FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
)

new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
new_chain("What is the discount percentage on the Roadster Men Navy Blue Slim Fit Mid Rise Clean Look Jeans?")
