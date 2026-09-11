import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

from models import Base


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL) #Conecta com o banco 


def create_tables():
    Base.metadata.create_all(engine) #chama a nossa arquitetura de banco de dados criada no arquivo models.py e cria as tabelas no banco de dados, representadas pelas classes do python.

    print("Tabelas criadas com sucesso!")


if __name__ == "__main__":
    create_tables()