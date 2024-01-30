from setuptools import find_packages, setup

setup(
    name = 'Medical Chatbot',
    version= '0.0.0',
    author= 'narendra rawat',
    author_email= 'rawatnarenddra009@gmail.com',
    packages= find_packages(),
    install_requires = ["langchain","streamlit","pypdf","python-dotenv","llama-index"]

)