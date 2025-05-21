import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from datetime import datetime
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing import Literal, Dict
from langchain_core.tools import tool

# Initialize all the necessary components
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
# Graph State
class State(TypedDict):
    file_path: str
    document_type: Literal["PDF", "DOCX"]
    document_content: dict 
    grade: int 
    feedback: str 

# Node Definitions

## Node: Determine Document Type
### Output Class/Tool
class DocumentType(BaseModel):
    document_type: Literal["PDF", "DOCX"] = Field(None,description="Type of the document")
    
### Augment LLM
llm_doc_type = llm.with_structured_output(DocumentType)

### LLM Node
def determine_document_type(state: State):
    """Determine a document type (PDF or DOCX)"""
    msg = llm_doc_type.invoke(f"Given the file path {state['file_path']}, can you determine what type of document this is? PDF or DOCX?")
    return {"document_type": msg.document_type}

## Node: Resume Reader
### Output Class/Tool
class Resume(BaseModel):
    name: str = Field(None,description="Name of the person")
    experience: str = Field(None,description=f"work experience with company, years and details. Today's date is {datetime.now().strftime('%Y-%m-%d')}"
    )
    city: str = Field(None,description="City where the person lives")
    education: str = Field(None,description="education with school, years and degree")
    skills: str = Field(None,description="Skills of the person")
    certifications: str = Field(None,description="Certifications of the person")

def read_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_content = ""
    for page in pages:
        text_content += page.page_content + "\n"
    return text_content

def read_docx(docx_path):
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()
    return documents[0].page_content

### Augment LLM
llm_read = llm.with_structured_output(Resume)

### LLM Node
def read_document(state: State):
    """Read the document and extract its content"""
    doc_type = state['document_type']
    if doc_type.upper() == "PDF":
        content = read_pdf(state['file_path'])
    elif doc_type.upper() == "DOCX":
        content = read_docx(state['file_path'])
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    # Invoke LLM to extract structured information with a more specific prompt
    prompt = """
    Parse the following resume and extract information in this exact format:
    - Name of the person
    - City where they live
    - Experience as a dictionary where:
        - Each key is a company name
        - Each value is a dictionary with 'years' and 'details' keys
    - Education as a dictionary where:
        - Each key is a school name
        - Each value is a dictionary with 'years' and 'degree' keys
    - List of skills (as a string)
    - List of certifications (as a string)

    Resume content: {content}
    """
    
    msg = llm_read.invoke(prompt.format(content=content))
    
    return {"document_content": msg.model_dump()}

## Node: Grade Resume
### Output Class/Tool
class Grading(BaseModel):
    grade: int = Field(None,description="Grade of the resume on a scale of 1-10")
    feedback: str = Field(None,description="Feedback on the resume")

### Augment LLM
llm_grade = llm.with_structured_output(Grading)

### LLM Node
def grade_resume(state: State):
    """Grade the resume based on the extracted information"""
    msg = llm_grade.invoke(f"Grade the resume based on the following information: {state['document_content']}") 
    return {"grade": msg.grade, "feedback": msg.feedback}

def main():
    # Load the document
    file_path = "test_resume.pdf"

    # Build workflow
    resume_reviewer_builder = StateGraph(State)

    # Add the nodes
    resume_reviewer_builder.add_node("determine_document_type", determine_document_type)
    resume_reviewer_builder.add_node("read_document", read_document)
    resume_reviewer_builder.add_node("grade_resume", grade_resume)

    # Add edges to connect nodes in sequence
    resume_reviewer_builder.add_edge(START, "determine_document_type")
    resume_reviewer_builder.add_edge("determine_document_type", "read_document")
    resume_reviewer_builder.add_edge("read_document", "grade_resume")
    resume_reviewer_builder.add_edge("grade_resume", END)

    # Compile the workflow
    resume_reviewer_workflow = resume_reviewer_builder.compile()

    # Save and show the workflow diagram
    workflow_image = resume_reviewer_workflow.get_graph().draw_mermaid_png()
    
    # Save the image to a file
    with open("workflow_diagram.png", "wb") as f:
        f.write(workflow_image)

    # Invoke the workflow with initial state
    state = resume_reviewer_workflow.invoke({"file_path": file_path})
    
    # Save the results to a text file
    with open("AI_Review.txt", "w") as f:
        f.write(f"Resume Review for {file_path}\n")
        f.write(f"Resume Grade: {state['grade']}/10\n")
        f.write(f"Feedback: {state['feedback']}\n")

    # Print the results
    print(f"Resume Grade: {state['grade']}/10")
    print(f"Feedback: {state['feedback']}")

if __name__ == "__main__":
    main()
