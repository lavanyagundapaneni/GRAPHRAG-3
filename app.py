from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase
from dotenv import load_dotenv
import boto3
import os

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Neo4j connection settings
uri = "bolt://localhost:7687"
user = "neo4j"
password = "NEO4J_PASSWORD"

# Initialize Neo4j driver
try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
except Exception as e:
    print(f"Error initializing Neo4j driver: {e}")
    raise

# AWS credentials and Bedrock client setup
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Define a function to call Amazon Bedrock using the Converse API
def call_bedrock(prompt):
    try:
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        response = bedrock_client.converse(
            modelId='mistral.mistral-7b-instruct-v0:2',
            messages=conversation,
            inferenceConfig={"maxTokens": 8192, "temperature": 0.7, "topP": 0.7},
            additionalModelRequestFields={"top_k": 50}
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Function to clean the generated Cypher query
def clean_cypher_query(cypher_query):
    # Remove unwanted prefixes like "Cypher Query:" or any other extra text
    cleaned_query = cypher_query.replace("Cypher Query:", "").strip()
    
    # Further cleaning to remove any unexpected characters or symbols
    unwanted_patterns = ["```", "/", "`",]
    for pattern in unwanted_patterns:
        cleaned_query = cleaned_query.replace(pattern, "")
    
    # Ensure only valid Cypher syntax is present
    return cleaned_query

# Function to generate Cypher query based on the user's question
def generate_cypher_query(question):
    prompt = f"""
    You are an expert at writing Cypher queries for a Neo4j database. Based on the following question, generate a well-formed Cypher query without any special symbols like ```, /, `, and Cypher Query: etc., to retrieve the relevant data from the Neo4j knowledge graph.
    Use examples as reference and generate correct queries
    Your schema includes:
 
    - `Student` nodes with properties like `name`.
    - `School` nodes with properties like `name`.
    - `University` nodes with properties like `name`.
    -  Relationships like `[:ENROLLED_IN]`, `[:WANTS_TO_PURUSE]`, and `[:AT_UNIVERSITY]` connecting `Student`, `Degree`, and `University`.

    Here are some examples:
    - Question: "how many students are pursuing Bachelor of Technology?"
      MATCH (s:Student)-[:WANTS_TO_PURUSE]->(d:Degree {{name: 'Bachelor of Technology'}})
      RETURN count(s) AS number_of_students

    - Question: "how many students are studying in Harvard University?"
      MATCH (s:Student)-[:WANTS_TO_PURUSE]->(d:Degree)-[:AT_UNIVERSITY]->(u:University) WHERE u.name = 'University of California, Berkeley' RETURN count(s) AS numberOfStudents
   
    - Question: "how many schools are there?"
      MATCH (school:School)
      RETURN count(school) AS numberOfSchools

    - Question: "Tell me about student1?"
      WITH 'Student1' AS studentName
      MATCH (s:Student {{name: studentName}})-[r]->(n)
      RETURN type(r) AS relationship_type, labels(n) AS node_labels, properties(n) AS node_properties

    - Question: "Which students are studying at the University of California, Berkeley?"
      MATCH (s:Student)-[:WANTS_TO_PURUSE]->(d:Degree)-[:AT_UNIVERSITY]->(u:University) WHERE u.name = 'University of California, Berkeley' RETURN s.name AS student_name, d.name AS degree_name

    - Question: "How many students are studying in hyderabad international school?"
      MATCH (s:Student)-[:ATTENDS_SCHOOL]->(school:School {{name: 'Hyderabad International School'}})
      RETURN count(s) AS number_of_students

    - Question: "Name of the schools ?"
      MATCH (school:School)
      RETURN school.name AS schoolname

    - Question: "name of the students who are studying in University of california,Berkely?"
      MATCH (s:Student)-[:WANTS_TO_PURUSE]->(d:Degree)-[:AT_UNIVERSITY]->(u:University)
      WHERE u.name = 'University of California, Berkeley'
      RETURN s.name AS studentname



    Your Question: {question}

    Generate the Cypher Query:
    """
    cypher_query = call_bedrock(prompt)
    
    # Clean the generated Cypher query
    cleaned_query = clean_cypher_query(cypher_query)
    print(f"Generated Cypher Query: {cleaned_query}")  # Print cleaned Cypher query
    return cleaned_query

# Function to fetch data from Neo4j using the cleaned Cypher query
def fetch_data_from_neo4j(query, parameters=None):
    try:
        with driver.session() as session:
            result = session.run(query, parameters)
            data = [record.data() for record in result]
            print(f"Fetched Data: {data}")  # Print fetched data
            return data
    except Exception as e:
        print(f"Error fetching data from Neo4j: {e}")
        return []

# Function to generate the final answer based on the fetched data
def generate_graph_rag(question, graph_data):
    prompt = f"""
    You are a language model interacting with a Neo4j graph database. 
    The following is data retrieved from the graph database. 
    Answer the question using only this data. Do not generate or infer any information that is not directly present in the data.

    Graph Data: {graph_data}

    Question: {question}

    Answer the question using only the provided graph data in meaningful format by combining whole fetched data.
    """
    response = call_bedrock(prompt)
    return response

# Flask route for the chatbot interface
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question")

        # Step 1: Generate the Cypher query based on the user's question
        cypher_query = generate_cypher_query(question)

        # Step 2: Fetch data from Neo4j using the cleaned Cypher query
        data = fetch_data_from_neo4j(cypher_query)

        # Step 3: Convert the fetched data to a string format for the LLM
        graph_data = "\n".join([f"{key}: {value}" for record in data for key, value in record.items()])

        # Step 4: Generate the final answer using the Bedrock model and the fetched data
        answer = generate_graph_rag(question, graph_data)

        return jsonify({"question": question, "answer": answer, "cypher_query": cypher_query})

    return render_template("index.html")

# Flask route for /ask
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

    # Step 1: Generate the Cypher query based on the user's question
    cypher_query = generate_cypher_query(question)

    # Step 2: Fetch data from Neo4j using the cleaned Cypher query
    data = fetch_data_from_neo4j(cypher_query)

    # Step 3: Convert the fetched data to a string format for the LLM
    graph_data = "\n".join([f"{key}: {value}" for record in data for key, value in record.items()])

    # Step 4: Generate the final answer using the Bedrock model and the fetched data
    answer = generate_graph_rag(question, graph_data)

    return jsonify({"question": question, "answer": answer})

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
