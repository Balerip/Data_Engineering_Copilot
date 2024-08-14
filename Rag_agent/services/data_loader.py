import os
import requests
from bs4 import BeautifulSoup
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.core import Document
import bs4
# from llama_index import Document, WebBaseLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

class DataLoader:
    def __init__(self, directory=os.environ.get("DATA_STORAGE_DIRECTORY")):
        self.directory = directory
        print(f"DataLoader directory: {self.directory}")
    
    def convert_webpages_to_text_documents(self, urls: list) -> None:
        """Converts multiple webpages to text files and saves them to the specified directory."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        for i, url in enumerate(urls):
            try:
                loader = WebBaseLoader(
                web_paths=(urls)
                )
 
                docs = loader.load()
 
                text_content = ""
                # for doc in docs:
                # # Adjust based on actual attribute or method for accessing text
                #     text_content += getattr(doc, 'content', '') + "\n"
                #     # response = requests.get(url)
                #     # response.raise_for_status()
                #     # soup = BeautifulSoup(response.text, 'html.parser')
                #     # print('Taking HTML format document')
                #     # text = soup.get_text(separator='\n')
                #     # print('Converting the document to text')
                #    
                #     text_content = ""
                for doc in docs:
    # Check and extract the correct text content
                    if hasattr(doc, 'content'):
                        text_content += doc.content + "\n"
                    elif hasattr(doc, 'text'):
                        text_content += doc.text + "\n"
                    elif hasattr(doc, 'get_text'):
                        text_content += doc.get_text() + "\n"
                    else:
        # Fallback or debugging information
                         text_content += str(doc) + "\n"
                filename = f"webpage_{i+1}.txt"
                file_path = os.path.join(self.directory, filename)
                print('Creating file Path')
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(text_content)
                print('opening and writing to file')
                print(f"Saved text to {file_path}")
        
# Save the text to a .txt file
                # out = "output.txt"
                # with open(out, "w", encoding="utf-8") as file:
                #         file.write(text_content)

 
            except requests.exceptions.RequestException as e:
                print(f"Error converting {url} to text document: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
    def load_documents(self, urls: list = None):
        if urls:
            self.convert_webpages_to_text_documents(urls)
        if not os.path.exists(self.directory):
            print(f"Directory does not exist: {self.directory}")
            raise Exception("Directory does not exist")
        documents = SimpleDirectoryReader(self.directory).load_data()
        return documents